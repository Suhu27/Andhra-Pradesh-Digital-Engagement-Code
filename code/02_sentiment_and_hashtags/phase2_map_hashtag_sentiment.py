# ============================================================
# phase2_map_hashtag_sentiment.py
# Maps the final hashtag dictionary onto all 24 post datasets.
# Adds per-post hashtag sentiment columns to each file.
#
# Input  : Master_Hashtag_Dictionary_Final.csv
#          24 x  *_Sentiment.xlsx  (caption sentiment already done)
# Output : 24 x  *_FullAnalysis.xlsx  (caption + hashtag sentiment)
#          _PHASE2_SUMMARY.xlsx       (cross-dataset rollup)
#
# New columns added per post:
#   hashtag_total_count    — total hashtags extracted for the post
#   hashtag_matched_count  — how many were found in the dictionary
#   hashtag_neg_count      — count of Negative hashtags in post
#   hashtag_neu_count      — count of Neutral hashtags in post
#   hashtag_pos_count      — count of Positive hashtags in post
#   hashtag_sentiment_score — weighted mean (-1=Neg, 0=Neu, +1=Pos)
#   hashtag_dominant_label — Negative / Neutral / Positive / NoHashtags
#   hashtag_avg_confidence — mean confidence of matched hashtags
#   hashtag_charged_tags   — pipe-separated list of Neg/Pos hashtags only
# ============================================================

import os
import re
import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION  ← edit these two paths only
# ─────────────────────────────────────────────
DICT_FILE       = r"C:\Users\Suhru\Desktop\Sentiment Analysis\Master_Hashtag_Dictionary_Final.csv"
DATASETS_FOLDER = r"C:\Users\Suhru\Desktop\Sentiment Analysis\sentiment_output"
OUTPUT_FOLDER   = r"C:\Users\Suhru\Desktop\Sentiment Analysis\full_analysis"

DATASET_FILES = [
    "ncbn_Facebook_JanMay2024_Sentiment.xlsx",
    "ncbn_Facebook_JunAug2023_Sentiment.xlsx",
    "ncbn_Facebook_SepDec2023_Sentiment.xlsx",
    "ncbn_Instagram_JanMay2024_Sentiment.xlsx",
    "ncbn_Instagram_JunAug2023_Sentiment.xlsx",
    "ncbn_Instagram_SepDec2023_Sentiment.xlsx",
    "tdp_Facebook_JanMay2024_Sentiment.xlsx",
    "tdp_Facebook_JunAug2023_Sentiment.xlsx",
    "tdp_Facebook_SepDec2023_Sentiment.xlsx",
    "tdp_Instagram_JanMay2024_Sentiment.xlsx",
    "tdp_Instagram_JunAug2023_Sentiment.xlsx",
    "tdp_Instagram_SepDec2023_Sentiment.xlsx",
    "ysjagan_Facebook_JanMay2024_Sentiment.xlsx",
    "ysjagan_Facebook_JunAug2023_Sentiment.xlsx",
    "ysjagan_Facebook_SepDec2023_Sentiment.xlsx",
    "ysjagan_Instagram_JanMay2024_Sentiment.xlsx",
    "ysjagan_Instagram_JunAug2023_Sentiment.xlsx",
    "ysjagan_Instagram_SepDec2023_Sentiment.xlsx",
    "ysrcp_Facebook_JanMay2024_Sentiment.xlsx",
    "ysrcp_Facebook_JunAug2023_Sentiment.xlsx",
    "ysrcp_Facebook_SepDec2023_Sentiment.xlsx",
    "ysrcp_Instagram_JanMay2024_Sentiment.xlsx",
    "ysrcp_Instagram_JunAug2023_Sentiment.xlsx",
    "ysrcp_Instagram_SepDec2023_Sentiment.xlsx",
]

LABEL_SCORE = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}

# ─────────────────────────────────────────────
# LOAD DICTIONARY
# ─────────────────────────────────────────────
print(f"[STEP 1] Loading dictionary: {DICT_FILE}")
dict_df = pd.read_csv(DICT_FILE)
print(f"         Rows: {len(dict_df):,}")

# Build lookup: hashtag_raw → {label, confidence, score_neg, score_neu, score_pos}
lookup = {}
for _, row in dict_df.iterrows():
    lookup[row["hashtag_raw"]] = {
        "label":      row["sentiment_label"],
        "confidence": float(row["sentiment_confidence"]),
        "score_neg":  float(row["score_negative"]),
        "score_neu":  float(row["score_neutral"]),
        "score_pos":  float(row["score_positive"]),
    }

neg_tags = {h for h, v in lookup.items() if v["label"] == "Negative"}
pos_tags = {h for h, v in lookup.items() if v["label"] == "Positive"}
print(f"         Negative: {len(neg_tags):,}  |  Positive: {len(pos_tags):,}  "
      f"|  Total: {len(lookup):,}")

# ─────────────────────────────────────────────
# HASHTAG PARSER
# Handles all common storage formats:
#   ['#tag1', '#tag2']   Python list string
#   ["#tag1","#tag2"]    JSON list
#   #tag1, #tag2         comma-separated
#   #tag1 #tag2          space-separated
#   tag1, tag2           no # prefix
# ─────────────────────────────────────────────
def parse_hashtags(raw) -> list:
    if pd.isna(raw) or str(raw).strip() in ('', 'nan', '[]', 'None'):
        return []
    s = str(raw).strip()

    # Try Python/JSON list
    if s.startswith('['):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                tags = [str(t).strip().lower() for t in parsed if str(t).strip()]
                # Normalise: ensure each starts with #
                return [t if t.startswith('#') else f'#{t}' for t in tags]
        except Exception:
            pass
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                tags = [str(t).strip().lower() for t in parsed if str(t).strip()]
                return [t if t.startswith('#') else f'#{t}' for t in tags]
        except Exception:
            pass

    # Plain string — extract all #word patterns
    found = re.findall(r'#\w+', s.lower())
    if found:
        return found

    # No # prefix — try comma or space split
    if ',' in s:
        parts = [p.strip().lower() for p in s.split(',') if p.strip()]
    else:
        parts = [p.strip().lower() for p in s.split() if p.strip()]
    return [p if p.startswith('#') else f'#{p}' for p in parts if p]


# ─────────────────────────────────────────────
# POST-LEVEL AGGREGATION
# ─────────────────────────────────────────────
def aggregate_hashtag_sentiment(hashtags: list) -> dict:
    """
    Given a list of hashtag strings for one post, return
    aggregated sentiment fields using the dictionary.
    """
    result = {
        "hashtag_total_count":     len(hashtags),
        "hashtag_matched_count":   0,
        "hashtag_neg_count":       0,
        "hashtag_neu_count":       0,
        "hashtag_pos_count":       0,
        "hashtag_sentiment_score": np.nan,
        "hashtag_dominant_label":  "NoHashtags" if len(hashtags) == 0 else "Unmatched",
        "hashtag_avg_confidence":  np.nan,
        "hashtag_charged_tags":    "",
    }
    if not hashtags:
        return result

    scores   = []
    confs    = []
    charged  = []

    for tag in hashtags:
        if tag in lookup:
            entry = lookup[tag]
            result["hashtag_matched_count"] += 1
            lbl = entry["label"]
            if lbl == "Negative":
                result["hashtag_neg_count"] += 1
                charged.append(tag)
            elif lbl == "Neutral":
                result["hashtag_neu_count"] += 1
            elif lbl == "Positive":
                result["hashtag_pos_count"] += 1
                charged.append(tag)
            scores.append(LABEL_SCORE[lbl])
            confs.append(entry["confidence"])

    if scores:
        result["hashtag_sentiment_score"] = round(float(np.mean(scores)), 4)
        result["hashtag_avg_confidence"]  = round(float(np.mean(confs)), 4)

        # Dominant label: majority vote among matched hashtags
        neg = result["hashtag_neg_count"]
        neu = result["hashtag_neu_count"]
        pos = result["hashtag_pos_count"]
        mx  = max(neg, neu, pos)
        if neg == neu == pos:
            result["hashtag_dominant_label"] = "Neutral"   # tie → Neutral
        elif mx == neg:
            result["hashtag_dominant_label"] = "Negative"
        elif mx == pos:
            result["hashtag_dominant_label"] = "Positive"
        else:
            result["hashtag_dominant_label"] = "Neutral"
    else:
        result["hashtag_dominant_label"] = "Unmatched"

    result["hashtag_charged_tags"] = " | ".join(charged) if charged else ""
    return result


# ─────────────────────────────────────────────
# PROCESS FILES
# ─────────────────────────────────────────────
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
summary_rows = []

print(f"\n[STEP 2] Processing {len(DATASET_FILES)} files...")

for fname in DATASET_FILES:
    fpath = os.path.join(DATASETS_FOLDER, fname)
    if not os.path.exists(fpath):
        print(f"  [SKIP]  {fname} — file not found")
        continue

    print(f"\n  ── {fname}")
    df = pd.read_excel(fpath)
    print(f"     Rows: {len(df):,}  |  Cols: {len(df.columns)}")

    # Detect hashtag column (handles minor naming variations)
    ht_col = None
    for candidate in ["Extracted_Hashtags", "extracted_hashtags",
                       "hashtags", "Hashtags", "tags"]:
        if candidate in df.columns:
            ht_col = candidate
            break
    if ht_col is None:
        print(f"     [WARN] No hashtag column found — skipping {fname}")
        continue
    print(f"     Hashtag column: '{ht_col}'")

    # Aggregate sentiment for each row
    agg_rows = []
    for raw_val in df[ht_col]:
        tags = parse_hashtags(raw_val)
        agg_rows.append(aggregate_hashtag_sentiment(tags))

    agg_df = pd.DataFrame(agg_rows)
    for col in agg_df.columns:
        df[col] = agg_df[col].values

    # Save
    out_name = fname.replace("_Sentiment.xlsx", "_FullAnalysis.xlsx")
    out_path = os.path.join(OUTPUT_FOLDER, out_name)
    df.to_excel(out_path, index=False)

    # Stats
    n = len(df)
    matched    = (df["hashtag_matched_count"] > 0).sum()
    no_hash    = (df["hashtag_dominant_label"] == "NoHashtags").sum()
    unmatched  = (df["hashtag_dominant_label"] == "Unmatched").sum()
    dom_counts = df["hashtag_dominant_label"].value_counts().to_dict()

    print(f"     Posts with ≥1 matched hashtag: {matched:,} / {n:,} "
          f"({matched/n*100:.1f}%)")
    print(f"     Dominant label dist: {dom_counts}")
    print(f"     Saved → {out_name}")

    # Summary row
    parts   = fname.replace("_Sentiment.xlsx", "").split("_")
    entity  = parts[0]
    platform = parts[1]
    period  = "_".join(parts[2:])
    summary_rows.append({
        "entity":        entity,
        "platform":      platform,
        "period":        period,
        "total_posts":   n,
        "posts_no_hashtags":  no_hash,
        "posts_unmatched":    unmatched,
        "posts_matched":      matched,
        **{f"dominant_{k}": v for k, v in dom_counts.items()},
        "avg_hashtag_sentiment_score":
            round(df["hashtag_sentiment_score"].dropna().mean(), 4)
            if matched > 0 else np.nan,
    })

# ─────────────────────────────────────────────
# SUMMARY EXPORT
# ─────────────────────────────────────────────
if summary_rows:
    summary_df = pd.DataFrame(summary_rows).fillna(0)
    summary_path = os.path.join(OUTPUT_FOLDER, "_PHASE2_SUMMARY.xlsx")
    summary_df.to_excel(summary_path, index=False)

    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"\n  Files processed : {len(summary_rows)}")
    print(f"  Summary saved  → {summary_path}")
    print(f"\n  Cross-entity overview:")
    print(f"\n  {'Entity':<12} {'Platform':<12} {'Period':<14} "
          f"{'Posts':<8} {'Matched%':<10} {'Avg Score'}")
    print("  " + "─"*70)
    for _, row in summary_df.iterrows():
        n   = int(row.get("total_posts", 0))
        m   = int(row.get("posts_matched", 0))
        pct = m / n * 100 if n > 0 else 0
        avg = row.get("avg_hashtag_sentiment_score", np.nan)
        print(f"  {row['entity']:<12} {row['platform']:<12} {row['period']:<14} "
              f"{n:<8,} {pct:<10.1f} {avg:.3f}")

    print(f"\n  Columns added to each FullAnalysis file:")
    print("    hashtag_total_count      — total hashtags in post")
    print("    hashtag_matched_count    — matched in dictionary")
    print("    hashtag_neg_count        — Negative hashtag count")
    print("    hashtag_neu_count        — Neutral hashtag count")
    print("    hashtag_pos_count        — Positive hashtag count")
    print("    hashtag_sentiment_score  — weighted mean (-1 to +1)")
    print("    hashtag_dominant_label   — Negative/Neutral/Positive/NoHashtags/Unmatched")
    print("    hashtag_avg_confidence   — mean dict confidence for matched tags")
    print("    hashtag_charged_tags     — pipe-separated list of Neg/Pos tags only")
    print(f"\n  Hashtag sentiment analysis complete. Caption sentiment untouched.")
    print(f"  All 24 files now carry both caption_sentiment_label and")
    print(f"  hashtag_dominant_label for downstream analysis.")
else:
    print("\n[WARN] No files were processed. Check DATASETS_FOLDER path.")

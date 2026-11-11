#!/usr/bin/env python3
"""
phase3_validate_large.py
────────────────────────
Compact validation for deberta-v3-large on 3 seeds BEFORE the 30k bulk run.
Run this first. Read the compact output. Only proceed to bulk when all gates pass.

PURPOSE:  Re-validate all thresholds and R-rules because the large model
          produces different confidence distributions than the base model.

Runtime:  ~25–35 min per seed on T4 GPU (3 seeds total ~90 min)
Output:   compact_validate_large_seed{N}.txt  (one per seed — paste to assistant)
          validate_large_seed{N}.xlsx          (full sample — keep for reference)

Usage:
  Kaggle:  upload your 24 *_FullAnalysis.xlsx to a dataset, point INPUT_DIR below
  Colab:   mount Drive, point INPUT_DIR below
"""

import os, re, time, warnings, json
import pandas as pd
import torch
from glob import glob
from transformers import pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PATHS  — edit for your environment
# ─────────────────────────────────────────────────────────────────
INPUT_DIR = (
    "/kaggle/input/fullanalysis"
    if os.path.exists("/kaggle/input")
    else "/content/drive/MyDrive/Sentiment Analysis/full_analysis"
)
OUTPUT_DIR = (
    "/kaggle/working"
    if os.path.exists("/kaggle/working")
    else "/content/drive/MyDrive/Sentiment Analysis/phase3_output"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CAPTION_COL      = "caption_for_sentiment"
SAMPLE_PER_GROUP = 25        # 25 × 24 strata = 587 rows per seed
MIN_WORDS        = 8
BATCH_SIZE       = 4         # safe for large model on T4 during validation
MODEL_ID         = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEVICE           = 0 if torch.cuda.is_available() else -1
SEEDS            = [42, 123, 250]  # run all 3 in one session

# ── THRESHOLDS (base-calibrated — we are TESTING whether these still hold) ──
NEUTRAL_RECLASSIFY_THR  = 0.50
NEEDS_REVIEW_TONE_THR   = 0.38
EMOTION_CONF_FLOOR      = 0.40
EMOTION_SECONDARY_THR   = 0.35
TONE_SECONDARY_MIN_PRI  = 0.50
TONE_SECONDARY_THR      = 0.22
TONE_DEF_SECONDARY_THR  = 0.35

# ── KEYWORD PATTERNS (tightened — word boundaries) ──────────────
LEGAL_KEYWORDS = (
    r"\barrest\b|illegal detain|court order|case filed|\bFIR\b|\bbail\b|"
    r"\bcustody\b|ED raid|\bCBI\b|\bremand\b|\bjail\b|\bprison\b|"
    r"police action|\bsummons\b|\bwarrant\b|case against|criminal case|"
    r"\bframed\b|false case|quash petition|anticipatory bail"
)
DEATH_KEYWORDS = (
    r"\bdeath\b|\bdied\b|\bdemise\b|passed away|passing of|condolence|"
    r"accident victim|taken away|no more|\bmourn\b|\bfuneral\b|last rites|"
    r"death anniversary|rest in peace|\bRIP\b|\buntimely\b|\bgrief\b|"
    r"bereavement|no longer with us"
)

# ─────────────────────────────────────────────────────────────────
# HYPOTHESES  (v8c.1 — identical to validated base-model version)
# ─────────────────────────────────────────────────────────────────
TONE_HYPOTHESES = [
    "This text criticizes, blames, condemns, or attacks a political opponent "
    "for corruption, incompetence, broken promises, or criminal behavior",

    "This text announces government schemes, makes election promises, or appeals "
    "to voters with reasoned arguments about development or welfare",

    "This text uses first-person language where the speaker explicitly denies "
    "or contests a specific named legal case, criminal arrest, court order, or "
    "formal criminal charge filed against themselves or their party — NOT a "
    "comparison of governance records between administrations, NOT an attack on "
    "an opponent's policy record, and NOT an assertion of personal development "
    "achievements or welfare delivery",

    "This text motivates supporters, celebrates or pays tribute to any political "
    "leader, freedom fighter, historical national figure, national poet, or "
    "deceased family member of a politician — living or deceased — articulates "
    "a vision for society or the nation, or rallies people around a shared "
    "cultural, political, or civic identity",

    "This text portrays the speaker, their leader, or their party as victims of "
    "political conspiracy, illegal arrests, persecution, or deliberate media "
    "bias by opponents",

    "This text is exclusively a condolence message for a person who died in a "
    "non-political context such as a road accident or illness, OR a greeting "
    "for a religious festival such as Diwali, Eid, or Christmas, OR a national "
    "public holiday greeting such as Independence Day or Republic Day — with "
    "absolutely no mention of any politician, political party, historical leader, "
    "freedom fighter, national poet, election, governance record, or development work"
]
TONE_NAMES = ["Aggressive", "Persuasive", "Defensive",
              "Inspirational", "Victim_Narrative", "Neutral"]

EMOTION_HYPOTHESES = [
    "This text conveys anger, outrage, or contempt directed at a political "
    "opponent, party, or government policy",
    "This text conveys pride in personal achievements, governance record, "
    "party legacy, or cultural identity",
    "This text conveys fear, alarm, or urgency about a danger, threat, or "
    "worsening situation affecting people",
    "This text conveys hope, optimism, or confident expectation about a "
    "positive future outcome or political victory",
    "This text conveys trust in a leader, gratitude for welfare schemes, or "
    "confidence in a government's commitment to the people"
]
EMOTION_NAMES = ["Anger", "Pride", "Fear", "Hope", "Trust"]

SUBJECTIVITY_HYPOTHESES = [
    "This text primarily reports verifiable events, statistics, government "
    "announcements, or factual information",
    "This text consists primarily of personal judgments, political attacks, "
    "sarcasm, or emotional condemnation with little or no verifiable factual "
    "content — even if names or events are mentioned in passing",
    "This text contains substantial verifiable facts or official information "
    "alongside clear personal opinions, political framing, or unverified claims "
    "— where both factual and opinionated elements are meaningfully present, "
    "not incidental"
]
SUBJECTIVITY_NAMES = ["Fact", "Opinion", "Mixed"]

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def parse_filename(fname):
    m = re.match(
        r"([a-zA-Z]+)_(Facebook|Instagram)_([A-Za-z0-9]+)_FullAnalysis\.xlsx",
        os.path.basename(fname), re.IGNORECASE
    )
    return (m.group(1).lower(), m.group(2), m.group(3)) if m \
           else ("unknown", "unknown", "unknown")

def is_insufficient(text):
    if pd.isna(text) or str(text).strip() in ["", "nan"]:
        return True
    words = [w for w in re.findall(r'\b\w+\b', str(text))
             if not w.startswith('#')]
    return len(words) < MIN_WORDS

def classify(clf, texts, hypotheses, names, multi_label=False):
    results = clf(
        texts, candidate_labels=hypotheses,
        multi_label=multi_label, hypothesis_template="{}"
    )
    if isinstance(results, dict):
        results = [results]
    return [
        {names[hypotheses.index(h)]: round(dict(zip(r["labels"], r["scores"]))[h], 4)
         for h in hypotheses}
        for r in results
    ]

def extract_tone(sd):
    ranked   = sorted(sd.items(), key=lambda x: -x[1])
    l1, c1   = ranked[0]; l2, c2 = ranked[1]
    if c1 < TONE_SECONDARY_MIN_PRI:
        return l1, round(c1, 3), "None", round(c2, 3)
    thr      = TONE_DEF_SECONDARY_THR if l2 == "Defensive" else TONE_SECONDARY_THR
    return l1, round(c1, 3), (l2 if c2 >= thr else "None"), round(c2, 3)

def extract_emo(sd):
    ranked   = sorted(sd.items(), key=lambda x: -x[1])
    l1, c1   = ranked[0]; l2, c2 = ranked[1]
    return l1, round(c1, 3), (l2 if c2 >= EMOTION_SECONDARY_THR else "None"), round(c2, 3)

def apply_rules(df, has_legal, has_death):
    df = df.copy()
    df["tone_primary"] = df["tone_primary_raw"].copy()
    cnt = {}

    # R1: Low-confidence Neutral → secondary or Persuasive
    r1       = (df["tone_primary"] == "Neutral") & (df["tone_primary_c"] < NEUTRAL_RECLASSIFY_THR)
    r1_has_s = r1 & (df["tone_secondary"] != "None")
    r1_no_s  = r1 & (df["tone_secondary"] == "None")
    df.loc[r1_has_s, "tone_primary"]   = df.loc[r1_has_s, "tone_secondary"]
    df.loc[r1_has_s, "tone_primary_c"] = df.loc[r1_has_s, "tone_secondary_c"]
    df.loc[r1_no_s,  "tone_primary"]   = "Persuasive"
    df.loc[r1_no_s,  "tone_primary_c"] = 0.30
    cnt["R1"] = r1.sum()

    # R2: Defensive + positive emotion → Persuasive (governance claim, not legal denial)
    r2 = (df["tone_primary"] == "Defensive") & \
         (df["emo_primary"].isin(["Pride", "Trust", "Hope"]))
    df.loc[r2, "tone_primary"] = "Persuasive"
    cnt["R2"] = r2.sum()

    # R4: Neutral + positive emotion → Inspirational / Persuasive
    r4i = (df["tone_primary"] == "Neutral") & \
          (df["emo_primary"].isin(["Pride", "Hope"])) & \
          (df["emo_primary_c"] >= 0.40)
    r4p = (df["tone_primary"] == "Neutral") & \
          (df["emo_primary"] == "Trust") & \
          (df["emo_primary_c"] >= 0.60)
    df.loc[r4i, "tone_primary"] = "Inspirational"
    df.loc[r4p, "tone_primary"] = "Persuasive"
    cnt["R4"] = r4i.sum() + r4p.sum()

    # R6: Neutral + Anger ≥ 0.70 → Aggressive
    r6 = (df["tone_primary"] == "Neutral") & \
         (df["emo_primary"] == "Anger") & \
         (df["emo_primary_c"] >= 0.70)
    df.loc[r6, "tone_primary"] = "Aggressive"
    cnt["R6"] = r6.sum()

    # R8: Neutral + Fear ≥ 0.55 → Aggressive
    r8 = (df["tone_primary"] == "Neutral") & \
         (df["emo_primary"] == "Fear") & \
         (df["emo_primary_c"] >= 0.55)
    df.loc[r8, "tone_primary"] = "Aggressive"
    cnt["R8"] = r8.sum()

    # R7: Neutral + Informational + no death keyword → Persuasive
    r7 = (df["tone_primary"] == "Neutral") & \
         (df["emo_primary"] == "Informational") & \
         (~has_death)
    df.loc[r7, "tone_primary"] = "Persuasive"
    cnt["R7"] = r7.sum()

    # R5: Defensive + Informational + no legal keyword → Persuasive
    r5 = (df["tone_primary"] == "Defensive") & \
         (df["emo_primary"] == "Informational") & \
         (~has_legal)
    df.loc[r5, "tone_primary"] = "Persuasive"
    cnt["R5"] = r5.sum()

    # R9 (NEW): Defensive + Fear/Anger + no legal keyword → Aggressive
    # Catches: "they are attacking us / law & order collapsed" posts
    # that surface as Defensive because of "our side is harmed" phrasing
    r9 = (df["tone_primary"] == "Defensive") & (
         ((df["emo_primary"] == "Fear")  & (df["emo_primary_c"] >= 0.60)) |
         ((df["emo_primary"] == "Anger") & (df["emo_primary_c"] >= 0.70))
    ) & (~has_legal)
    df.loc[r9, "tone_primary"] = "Aggressive"
    cnt["R9"] = r9.sum()

    # needs_review flag
    df["avg_conf"] = (
        df["tone_primary_c"] + df["emo_primary_c"] + df["subj_conf"]
    ) / 3
    df["needs_review"] = (
        (df["tone_primary_c"] < NEEDS_REVIEW_TONE_THR) |
        ((df["emo_primary"] != "Informational") &
         (df["emo_primary_c"] < 0.55)) |
        (df["subj_conf"] < 0.55)
    )
    return df, cnt

def cap(text, n=130):
    t = str(text).replace("\n", " ").strip()
    return (t[:n] + "…") if len(t) > n else t

def row_line(row):
    acct  = f"{row['_account'][:4].upper()}|{row['_platform'][:2].upper()}|{row['_timeline']}"
    e2    = f"+{row['emo_secondary']}({row['emo_secondary_c']})" \
            if row['emo_secondary'] != "None" else ""
    t2    = f"+{row['tone_secondary']}({row['tone_secondary_c']})" \
            if row['tone_secondary'] != "None" else ""
    return (f"[{acct}] "
            f"T:{row['tone_primary']}({row['tone_primary_c']}){t2} "
            f"E:{row['emo_primary']}({row['emo_primary_c']}){e2} "
            f"S:{row['subj_label']}({row['subj_conf']}) "
            f"| \"{cap(row[CAPTION_COL])}\"")

# ─────────────────────────────────────────────────────────────────
# LOAD + SAMPLE
# ─────────────────────────────────────────────────────────────────
files = sorted(glob(os.path.join(INPUT_DIR, "*_FullAnalysis.xlsx")))
print(f"Found {len(files)} source files in {INPUT_DIR}")

all_source_dfs = []
for fpath in files:
    account, platform, timeline = parse_filename(fpath)
    df = pd.read_excel(fpath, dtype=str)
    df["_account"]      = account
    df["_platform"]     = platform
    df["_timeline"]     = timeline
    df["_insufficient"] = df[CAPTION_COL].apply(is_insufficient)
    all_source_dfs.append(df)

corpus_df = pd.concat(all_source_dfs, ignore_index=True)
valid_corpus = corpus_df[~corpus_df["_insufficient"]].copy()
print(f"Total valid rows in corpus: {len(valid_corpus)}")

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL  (once — reused across all seeds)
# ─────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} on {'GPU' if DEVICE == 0 else 'CPU'} ...")
clf = pipeline(
    "zero-shot-classification",
    model=MODEL_ID, device=DEVICE, batch_size=BATCH_SIZE
)
print("Model loaded.\n")

# ─────────────────────────────────────────────────────────────────
# RUN ALL 3 SEEDS
# ─────────────────────────────────────────────────────────────────
for SEED in SEEDS:
    print(f"{'='*60}")
    print(f"  SEED {SEED}  |  sampling {SAMPLE_PER_GROUP} rows per stratum ...")
    print(f"{'='*60}")

    # Stratified sample
    sampled_parts = []
    for (acc, plat, tl), grp in valid_corpus.groupby(
            ["_account", "_platform", "_timeline"]):
        n = min(SAMPLE_PER_GROUP, len(grp))
        sampled_parts.append(grp.sample(n, random_state=SEED))
    sample_df = pd.concat(sampled_parts, ignore_index=True).reset_index(drop=True)
    sample_df["_row_id"] = sample_df.index
    total = len(sample_df)
    print(f"  Sample size: {total} rows")

    # Inference
    captions   = sample_df[CAPTION_COL].fillna("").tolist()
    all_tone   = []; all_emo = []; all_subj = []
    t_start    = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = captions[i:i + BATCH_SIZE]
        all_tone.extend( classify(clf, batch, TONE_HYPOTHESES,
                                  TONE_NAMES,         multi_label=False))
        all_emo.extend(  classify(clf, batch, EMOTION_HYPOTHESES,
                                  EMOTION_NAMES,      multi_label=True))
        all_subj.extend( classify(clf, batch, SUBJECTIVITY_HYPOTHESES,
                                  SUBJECTIVITY_NAMES, multi_label=False))
        done    = min(i + BATCH_SIZE, total)
        elapsed = time.time() - t_start
        eta     = (total - done) / (done / elapsed) / 60 if done > 0 else 0
        print(f"  [{done}/{total}] ETA: {eta:.1f}min", end="\r")

    elapsed_total = (time.time() - t_start) / 60
    print(f"\n  Inference done in {elapsed_total:.1f} min")

    # Build columns
    t_res = [extract_tone(t) for t in all_tone]
    e_res = [extract_emo(e)  for e in all_emo]
    s_res = [sorted(s.items(), key=lambda x: -x[1])[0] for s in all_subj]

    sample_df["tone_primary_raw"]  = [r[0] for r in t_res]
    sample_df["tone_primary_c"]    = [r[1] for r in t_res]
    sample_df["tone_secondary"]    = [r[2] for r in t_res]
    sample_df["tone_secondary_c"]  = [r[3] for r in t_res]
    sample_df["emo_primary_raw"]   = [r[0] for r in e_res]
    sample_df["emo_primary_c"]     = [r[1] for r in e_res]
    sample_df["emo_secondary"]     = [r[2] for r in e_res]
    sample_df["emo_secondary_c"]   = [r[3] for r in e_res]
    sample_df["emo_primary"]       = sample_df.apply(
        lambda r: "Informational" if r["emo_primary_c"] < EMOTION_CONF_FLOOR
                  else r["emo_primary_raw"], axis=1
    )
    sample_df["subj_label"] = [r[0] for r in s_res]
    sample_df["subj_conf"]  = [round(r[1], 3) for r in s_res]

    has_legal = sample_df[CAPTION_COL].str.contains(
        LEGAL_KEYWORDS, case=False, na=False, regex=True)
    has_death = sample_df[CAPTION_COL].str.contains(
        DEATH_KEYWORDS, case=False, na=False, regex=True)

    sample_df, rule_counts = apply_rules(sample_df, has_legal, has_death)

    # Save xlsx
    xlsx_path = os.path.join(OUTPUT_DIR, f"validate_large_seed{SEED}.xlsx")
    sample_df.to_excel(xlsx_path, index=False)

    # ─────────────────────────────────────────────────────────────
    # BUILD COMPACT OUTPUT
    # ─────────────────────────────────────────────────────────────
    out = []
    P   = out.append
    n   = len(sample_df)
    SEP = "=" * 65

    P(SEP)
    P(f"  COMPACT VALIDATION LARGE | seed={SEED} | {n} rows | {MODEL_ID}")
    P(f"  Rules: R1 R2 R3 R4 R5 R6 R7 R8 R9(new)")
    P(f"  KEY: Agg Per Def Ins Vic Neu | Ang Pri Fea Hop Tru Inf | Fac Opi Mix")
    P(SEP)

    # S1 — Global distribution
    P("\n[S1] GLOBAL DISTRIBUTION")
    P("-" * 45)
    for col, lbl in [("tone_primary","Tone"), ("emo_primary","Emotion"), ("subj_label","Subj")]:
        vc = sample_df[col].value_counts()
        vals = "  ".join([f"{k}={v}({v/n*100:.0f}%)" for k, v in vc.items()])
        P(f"  {lbl}: {vals}")

    nr   = sample_df["needs_review"].sum()
    inf  = (sample_df["emo_primary"] == "Informational").sum()
    t2   = (sample_df["tone_secondary"] != "None").sum()
    rc_str = "  ".join([f"{k}={v}" for k, v in sorted(rule_counts.items())])
    P(f"  Low-conf={nr}({nr/n*100:.1f}%)  Inf={inf}({inf/n*100:.1f}%)  Tone-2nd={t2}({t2/n*100:.1f}%)")
    P(f"  Rules: {rc_str}")

    # S1b — CONFIDENCE DISTRIBUTION (KEY FOR THRESHOLD CALIBRATION)
    P("\n[S1b] CONFIDENCE DISTRIBUTION (compare to base to check threshold shift)")
    P("-" * 45)
    bins = [(0.0,0.38,"<0.38"),(0.38,0.55,"0.38-0.55"),(0.55,0.70,"0.55-0.70"),
            (0.70,0.85,"0.70-0.85"),(0.85,1.01,">=0.85")]
    P(f"  {'Band':<12} {'Tone':>8} {'Emotion*':>9} {'Subj':>8}   (*excl. Informational)")
    for lo, hi, hdr in bins:
        t_n = ((sample_df["tone_primary_c"] >= lo) & (sample_df["tone_primary_c"] < hi)).sum()
        e_mask = sample_df["emo_primary"] != "Informational"
        e_n = ((sample_df.loc[e_mask, "emo_primary_c"] >= lo) &
               (sample_df.loc[e_mask, "emo_primary_c"] < hi)).sum()
        s_n = ((sample_df["subj_conf"] >= lo) & (sample_df["subj_conf"] < hi)).sum()
        P(f"  {hdr:<12} {t_n:>8} {e_n:>9} {s_n:>8}")
    p_mean_tone = sample_df["tone_primary_c"].mean()
    p_mean_emo  = sample_df.loc[sample_df["emo_primary"] != "Informational", "emo_primary_c"].mean()
    p_mean_subj = sample_df["subj_conf"].mean()
    P(f"  Mean conf — Tone:{p_mean_tone:.3f}  Emotion(excl.Inf):{p_mean_emo:.3f}  Subj:{p_mean_subj:.3f}")

    # S2 — Tone × Account cross-tab
    P("\n[S2] TONE x ACCOUNT")
    P("-" * 45)
    ct = pd.crosstab(sample_df["_account"], sample_df["tone_primary"])
    P("  " + ct.to_string().replace("\n", "\n  "))

    # S3 — Top-3 high-conf anchors per label
    P("\n[S3] TOP-3 HIGH-CONF ANCHORS PER LABEL")
    P("     Format: [ACC|PL|TL] T:x(c) E:x(c) S:x(c) | caption")
    P("-" * 45)
    P("\n  -- TONE ANCHORS --")
    for lbl in TONE_NAMES + ["Neutral"]:
        sub = sample_df[sample_df["tone_primary"] == lbl].nlargest(3, "tone_primary_c")
        if len(sub) == 0:
            continue
        P(f"\n  {lbl}(n={len(sample_df[sample_df['tone_primary']==lbl])}):")
        for _, r in sub.iterrows():
            P(f"    {row_line(r)}")

    P("\n  -- EMOTION ANCHORS --")
    for lbl in EMOTION_NAMES + ["Informational"]:
        sub = sample_df[sample_df["emo_primary"] == lbl].nlargest(3, "emo_primary_c")
        if len(sub) == 0:
            continue
        P(f"\n  {lbl}(n={len(sample_df[sample_df['emo_primary']==lbl])}):")
        for _, r in sub.iterrows():
            P(f"    {row_line(r)}")

    # S4 — Bottom-15 confidence (excl. Informational)
    P("\n[S4] BOTTOM-15 CONFIDENCE ROWS  (excl. Informational)")
    P("-" * 45)
    b15 = sample_df[sample_df["emo_primary"] != "Informational"].nsmallest(15, "avg_conf")
    for _, r in b15.iterrows():
        P(f"  avg={r['avg_conf']:.3f} | {row_line(r)}")

    # S5 — Remaining Neutral (condolences only)
    P(f"\n[S5] REMAINING NEUTRAL  (expect: condolences/accidents/festivals only)")
    P("-" * 45)
    neu = sample_df[sample_df["tone_primary"] == "Neutral"]
    P(f"  Total: {len(neu)}")
    for _, r in neu.iterrows():
        P(f"  {row_line(r)}")

    # S6 — Remaining Defensive
    P(f"\n[S6] REMAINING DEFENSIVE  (expect: named legal case denial only)")
    P("-" * 45)
    defn = sample_df[sample_df["tone_primary"] == "Defensive"]
    P(f"  Total: {len(defn)}")
    for _, r in defn.iterrows():
        P(f"  {row_line(r)}")

    # S7 — R9 verification (what did the new rule actually catch?)
    P(f"\n[S7] R9 VERIFICATION  (Defensive+Fear/Anger→Aggressive, should be attack posts)")
    P("-" * 45)
    # Re-derive R9 mask on raw labels for display
    r9_display = sample_df[
        (sample_df["tone_primary_raw"] == "Defensive") & (
            ((sample_df["emo_primary_raw"] == "Fear")  & (sample_df["emo_primary_c"] >= 0.60)) |
            ((sample_df["emo_primary_raw"] == "Anger") & (sample_df["emo_primary_c"] >= 0.70))
        ) & (~has_legal)
    ]
    P(f"  Total caught by R9: {len(r9_display)}")
    for _, r in r9_display.iterrows():
        P(f"  raw_tone:Def→Agg | {row_line(r)}")

    # QG — Quality gates
    P(f"\n[QG] QUALITY GATES  (large model | seed={SEED})")
    P("-" * 45)
    def gate(label, col, val, lo, hi, note=""):
        status = "✅ PASS" if lo <= val <= hi else "❌ FAIL"
        P(f"  {label:<12} {val:.1f}%   (expect {lo}–{hi}%)  {status}  {note}")

    agg_pct  = (sample_df["tone_primary"] == "Aggressive").sum()    / n * 100
    def_pct  = (sample_df["tone_primary"] == "Defensive").sum()     / n * 100
    ins_pct  = (sample_df["tone_primary"] == "Inspirational").sum() / n * 100
    neu_pct  = (sample_df["tone_primary"] == "Neutral").sum()       / n * 100
    opi_pct  = (sample_df["subj_label"]   == "Opinion").sum()       / n * 100
    lc_pct   = nr                                                    / n * 100
    inf_pct  = inf                                                   / n * 100

    gate("Aggressive",  "tone_primary", agg_pct, 25, 40)
    gate("Defensive",   "tone_primary", def_pct,  0, 15)
    gate("Inspiratl",   "tone_primary", ins_pct, 12, 25)
    gate("Neutral",     "tone_primary", neu_pct,  0,  8)
    gate("Opinion",     "subj_label",   opi_pct, 15, 28)
    gate("Low-conf",    "needs_review", lc_pct,   0, 42,
         note="— if >42% raise NEEDS_REVIEW_TONE_THR to 0.40")
    gate("Inf(emo)",    "emo_primary",  inf_pct,  0, 40,
         note="— if <25% EMOTION_CONF_FLOOR may need lowering to 0.35")

    all_pass = all([
        25 <= agg_pct <= 40,
        def_pct  <= 15,
        ins_pct  >= 12,
        neu_pct  <=  8,
        opi_pct  >= 15,
        lc_pct   <= 42
    ])
    verdict = "✅ ALL PASS — proceed to bulk 30k" if all_pass \
              else "❌ FAIL — review failures before bulk run"
    P(f"\n  VERDICT: {verdict}")

    # Threshold shift alert
    P(f"\n[TH] THRESHOLD SHIFT CHECK  (large vs base — take action if shifted)")
    P("-" * 45)
    P(f"  Mean tone conf:   {p_mean_tone:.3f}  (base baseline: ~0.65)")
    P(f"  Mean emo conf:    {p_mean_emo:.3f}   (base baseline: ~0.72)")
    P(f"  Mean subj conf:   {p_mean_subj:.3f}  (base baseline: ~0.68)")
    P(f"  Tone <0.38 count: {((sample_df['tone_primary_c'] < 0.38).sum())}  (base baseline: ~114/587)")
    P(f"  IF mean tone conf > 0.75: consider raising NEEDS_REVIEW_TONE_THR to 0.42")
    P(f"  IF Inf(emo) < 25%: consider lowering EMOTION_CONF_FLOOR to 0.35")

    P(f"\n{SEP}")
    P(f"  END | seed={SEED} | Saved: validate_large_seed{SEED}.xlsx")
    P(SEP)

    # Save compact txt
    txt_path = os.path.join(OUTPUT_DIR, f"compact_validate_large_seed{SEED}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"\n✅ Seed {SEED} done. Compact output: {txt_path}")
    print(f"   Open → Ctrl+A → Ctrl+C → paste to assistant for review\n")

print("\n🎉 All 3 seeds complete. Paste each compact_validate_large_seed*.txt before running 30k.")

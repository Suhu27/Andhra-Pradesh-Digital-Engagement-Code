#!/usr/bin/env python3
"""
phase3_bulk_classify.py
───────────────────────
Production bulk run — classifies ALL ~30k rows across 24 datasets.

PREREQUISITES (run phase3_validate_large.py first and confirm all QG PASS):
  ✅ Seed 42  — QG pass
  ✅ Seed 123 — QG pass (all 6 gates)
  ✅ Seed 250 — QG pass

FIXES vs validation script:
  - R9 bug fix: KCR-style condolence posts (Defensive+Fear) no longer
    misclassified as Aggressive — injury/recovery language added to
    DEATH_KEYWORDS so has_death blocks R9 from firing.
  - 'Informational' emotion label exposed as a real category in output
    (emo_low_conf flag) so downstream analysis can filter it cleanly.
  - BATCH CHECKPOINTS: classified results saved every SAVE_INTERVAL rows
    so no data is lost if the kernel crashes or times out near the end.
  - FORCE FLUSH: every save calls os.fsync() to guarantee the file is
    physically written to /kaggle/working — fixes the Kaggle panel bug
    where output files don't appear until the kernel finishes.

OUTPUT (in /kaggle/working/):
  phase3_classified_partial.csv     ← live checkpoint — updated every 2000 rows
  phase3_classified_all.csv         ← final full output (all rows inc. skipped)
  phase3_per_dataset_summary.csv    ← tone/emo/subj distributions per dataset
  phase3_run_log.txt                ← timing + rule counts + QG check
  phase3_status.txt                 ← live progress (always visible in Kaggle panel)

Runtime: ~50–90 min on Kaggle dual-T4 (batch_size=8)
"""

import os, re, time, warnings
import pandas as pd
import torch
from glob import glob
from transformers import pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = "/kaggle/input/datasets/suhruthv/fullanalysis"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
CAPTION_COL  = "caption_for_sentiment"
MIN_WORDS    = 8
BATCH_SIZE     = 8          # T4 can handle 8 with deberta-large
SAVE_INTERVAL  = 2000       # save checkpoint every N classified rows
MODEL_ID     = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
DEVICE       = 0 if torch.cuda.is_available() else -1

# Thresholds — validated on seeds 42, 123, 250
NEUTRAL_RECLASSIFY_THR  = 0.50
NEEDS_REVIEW_TONE_THR   = 0.38
EMOTION_CONF_FLOOR      = 0.40
EMOTION_SECONDARY_THR   = 0.35
TONE_SECONDARY_MIN_PRI  = 0.50
TONE_SECONDARY_THR      = 0.22
TONE_DEF_SECONDARY_THR  = 0.35

# ─────────────────────────────────────────────────────────────────────
# KEYWORD PATTERNS
# ─────────────────────────────────────────────────────────────────────
LEGAL_KEYWORDS = (
    r"\barrest\b|illegal detain|court order|case filed|\bFIR\b|\bbail\b|"
    r"\bcustody\b|ED raid|\bCBI\b|\bremand\b|\bjail\b|\bprison\b|"
    r"police action|\bsummons\b|\bwarrant\b|case against|criminal case|"
    r"\bframed\b|false case|quash petition|anticipatory bail"
)

# Condolence/injury keyword guard — prevents R7, R8, R9 from firing on genuine
# sympathy/well-wish posts.  IMPORTANT: '\bhospital\b' was removed because it
# fires on political posts like "state of govt hospitals under Jagan" — these
# are attack posts that SHOULD be classified as Aggressive, not blocked.
# '\brecover\w*' was removed for the same reason ("recover the stolen money").
# Injury kept only in explicit accident/condolence phrasing.
DEATH_KEYWORDS = (
    r"\bdeath\b|\bdied\b|\bdemise\b|passed away|passing of|condolence|"
    r"accident victim|taken away|no more|\bmourn\b|\bfuneral\b|last rites|"
    r"death anniversary|rest in peace|\bRIP\b|\buntimely\b|\bgrief\b|"
    r"bereavement|no longer with us|"
    r"injured in accident|injured in road|sustained an injur|\bunwell\b|"
    r"get well soon|swift recovery|speedy recovery|pray.*recover|prays.*recover"
)

# ─────────────────────────────────────────────────────────────────────
# HYPOTHESES  (v8c.1 — validated)
# ─────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
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

def classify_batch(clf, texts, hypotheses, names, multi_label=False):
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
    ranked = sorted(sd.items(), key=lambda x: -x[1])
    l1, c1 = ranked[0]; l2, c2 = ranked[1]
    if c1 < TONE_SECONDARY_MIN_PRI:
        return l1, round(c1, 3), "None", round(c2, 3)
    thr = TONE_DEF_SECONDARY_THR if l2 == "Defensive" else TONE_SECONDARY_THR
    return l1, round(c1, 3), (l2 if c2 >= thr else "None"), round(c2, 3)

def extract_emo(sd):
    ranked = sorted(sd.items(), key=lambda x: -x[1])
    l1, c1 = ranked[0]; l2, c2 = ranked[1]
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

    # R2: Defensive + positive emotion → Persuasive (governance claim)
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

    # R9: Defensive + Fear/Anger + no legal keyword + no death keyword → Aggressive
    # FIXED: ~has_death now also covers injury/recovery language so condolence
    # posts about opponents' injuries are not incorrectly converted to Aggressive.
    r9 = (df["tone_primary"] == "Defensive") & (
         ((df["emo_primary"] == "Fear")  & (df["emo_primary_c"] >= 0.60)) |
         ((df["emo_primary"] == "Anger") & (df["emo_primary_c"] >= 0.70))
    ) & (~has_legal) & (~has_death)
    df.loc[r9, "tone_primary"] = "Aggressive"
    cnt["R9"] = r9.sum()

    # Reliability score = mean of 3 confidences (use for downstream filtering)
    df["reliability"] = (
        df["tone_primary_c"] + df["emo_primary_c"] + df["subj_conf"]
    ) / 3

    # needs_review flag (low-confidence rows — treat as secondary analysis)
    df["needs_review"] = (
        (df["tone_primary_c"] < NEEDS_REVIEW_TONE_THR) |
        ((df["emo_primary"] != "Informational") &
         (df["emo_primary_c"] < 0.55)) |
        (df["subj_conf"] < 0.55)
    )

    return df, cnt


# ─────────────────────────────────────────────────────────────────────
# FLUSH HELPER — force file to disk so Kaggle panel shows it
# ─────────────────────────────────────────────────────────────────────
def flush_to_disk(path, df, tag=""):
    """Save a DataFrame to CSV with UTF-8 encoding and force-flush to disk.
    Prints file size AND row count so silent truncation is immediately visible.
    """
    df.to_csv(path, index=False, encoding="utf-8")
    try:
        with open(path, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass
    size_kb   = os.path.getsize(path) / 1024
    # Verify row count by re-reading the header line
    with open(path, "r", encoding="utf-8") as fh:
        saved_rows = sum(1 for _ in fh) - 1   # subtract header
    mismatch = "  \u26a0 ROW COUNT MISMATCH" if saved_rows != len(df) else ""
    print(f"  [SAVED{tag}] {os.path.basename(path)} — {size_kb:.0f} KB  "
          f"({len(df)} rows expected, {saved_rows} in file){mismatch}")


def write_status(path, done, total, elapsed_sec, ckpt_size_kb, partial_df):
    """Write a plain-text status file that Kaggle always shows in the panel
    even when CSV files are not yet visible.
    """
    pct      = done / total * 100
    eta_min  = (total - done) / (done / elapsed_sec) / 60 if done > 0 else 0
    tone_vc  = dict(partial_df["tone_primary"].value_counts()) if "tone_primary" in partial_df else {}
    lines = [
        f"=== phase3_bulk_classify — LIVE STATUS ===",
        f"Progress  : {done}/{total} rows   ({pct:.1f}%)",
        f"Elapsed   : {elapsed_sec/60:.1f} min",
        f"ETA       : {eta_min:.1f} min",
        f"Partial   : {ckpt_size_kb:.0f} KB on disk",
        f"Tone dist : {tone_vc}",
        f"Updated   : row {done}",
    ]
    with open(path, "w") as sf:
        sf.write("\n".join(lines))
    try:
        with open(path, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────
# LOAD ALL SOURCE FILES
# ─────────────────────────────────────────────────────────────────────
files = sorted(glob(os.path.join(INPUT_DIR, "*_FullAnalysis.xlsx")))
print(f"Found {len(files)} source files in {INPUT_DIR}")
if len(files) == 0:
    raise FileNotFoundError(
        f"No *_FullAnalysis.xlsx files found in {INPUT_DIR}.\n"
        f"Check your Kaggle dataset slug — it must match exactly."
    )

all_dfs = []
for fpath in files:
    account, platform, timeline = parse_filename(fpath)
    df = pd.read_excel(
        fpath, dtype=str,
        keep_default_na=False,   # CRITICAL: prevents pandas silently converting
                                 # caption strings like 'nan', 'N/A', 'null',
                                 # 'NA', '', ' ' into actual NaN floats.
                                 # Without this, captions containing those strings
                                 # would be classified as missing and skipped.
    )
    df["_account"]      = account
    df["_platform"]     = platform
    df["_timeline"]     = timeline
    df["_insufficient"] = df[CAPTION_COL].apply(is_insufficient)
    all_dfs.append(df)

corpus_df = pd.concat(all_dfs, ignore_index=True)
valid_mask = ~corpus_df["_insufficient"]
n_total    = len(corpus_df)
n_valid    = valid_mask.sum()
print(f"Total rows: {n_total}  |  Valid (classifiable): {n_valid}  |  Skipped: {n_total - n_valid}")

# ─────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} on {'GPU' if DEVICE == 0 else 'CPU'} ...")
clf = pipeline(
    "zero-shot-classification",
    model=MODEL_ID, device=DEVICE, batch_size=BATCH_SIZE
)
print("Model loaded.\n")

# ─────────────────────────────────────────────────────────────────────
# CLASSIFY VALID ROWS
# ─────────────────────────────────────────────────────────────────────
valid_df = corpus_df[valid_mask].copy().reset_index(drop=True)
captions = valid_df[CAPTION_COL].fillna("").tolist()
total    = len(captions)

all_tone = []; all_emo = []; all_subj = []
t_start  = time.time()
ckpt_csv    = os.path.join(OUTPUT_DIR, "phase3_classified_partial.csv")
status_path = os.path.join(OUTPUT_DIR, "phase3_status.txt")
print(f"Classifying {total} rows (batch_size={BATCH_SIZE}, checkpoint every {SAVE_INTERVAL}) ...")

for i in range(0, total, BATCH_SIZE):
    batch = captions[i:i + BATCH_SIZE]
    all_tone.extend(classify_batch(clf, batch, TONE_HYPOTHESES,         TONE_NAMES,         multi_label=False))
    all_emo.extend( classify_batch(clf, batch, EMOTION_HYPOTHESES,      EMOTION_NAMES,      multi_label=True))
    all_subj.extend(classify_batch(clf, batch, SUBJECTIVITY_HYPOTHESES, SUBJECTIVITY_NAMES, multi_label=False))

    done    = min(i + BATCH_SIZE, total)
    elapsed = time.time() - t_start
    eta     = (total - done) / (done / elapsed) / 60 if done > 0 else 0
    pct     = done / total * 100
    print(f"  [{done}/{total}] {pct:.1f}%  ETA: {eta:.1f} min", end="\r")

    # ── CHECKPOINT: save partial results every SAVE_INTERVAL rows ────
    # Fires when we hit a multiple of SAVE_INTERVAL OR on the final batch.
    # Overwrites phase3_classified_partial.csv each time — always contains
    # the latest classified rows so far with all rules applied.
    if done % SAVE_INTERVAL < BATCH_SIZE or done == total:
        print()  # newline so checkpoint messages don't clobber the progress line

        # Build partial classified DataFrame from everything classified so far
        tp = [extract_tone(t) for t in all_tone]
        ep = [extract_emo(e)  for e in all_emo]
        sp = [sorted(s.items(), key=lambda x: -x[1])[0] for s in all_subj]

        pdf = valid_df.iloc[:done].copy()
        pdf["tone_primary_raw"] = [r[0] for r in tp]
        pdf["tone_primary_c"]   = [r[1] for r in tp]
        pdf["tone_secondary"]   = [r[2] for r in tp]
        pdf["tone_secondary_c"] = [r[3] for r in tp]
        pdf["emo_primary_raw"]  = [r[0] for r in ep]
        pdf["emo_primary_c"]    = [r[1] for r in ep]
        pdf["emo_secondary"]    = [r[2] for r in ep]
        pdf["emo_secondary_c"]  = [r[3] for r in ep]
        pdf["emo_primary"] = pdf.apply(
            lambda r: "Informational" if r["emo_primary_c"] < EMOTION_CONF_FLOOR
                      else r["emo_primary_raw"], axis=1)
        pdf["emo_low_conf"] = pdf["emo_primary_c"] < EMOTION_CONF_FLOOR
        pdf["subj_label"]   = [r[0] for r in sp]
        pdf["subj_conf"]    = [round(r[1], 3) for r in sp]

        hl_p = pdf[CAPTION_COL].str.contains(LEGAL_KEYWORDS, case=False, na=False, regex=True)
        hd_p = pdf[CAPTION_COL].str.contains(DEATH_KEYWORDS, case=False, na=False, regex=True)
        pdf, _ = apply_rules(pdf, hl_p, hd_p)

        flush_to_disk(ckpt_csv, pdf, tag=f" [{done}/{total}]")
        ckpt_kb = os.path.getsize(ckpt_csv) / 1024
        write_status(status_path, done, total, elapsed, ckpt_kb, pdf)

elapsed_total = (time.time() - t_start) / 60
print(f"\nInference done in {elapsed_total:.1f} min")

# ─────────────────────────────────────────────────────────────────────
# BUILD RESULT COLUMNS
# ─────────────────────────────────────────────────────────────────────
t_res = [extract_tone(t) for t in all_tone]
e_res = [extract_emo(e)  for e in all_emo]
s_res = [sorted(s.items(), key=lambda x: -x[1])[0] for s in all_subj]

valid_df["tone_primary_raw"]  = [r[0] for r in t_res]
valid_df["tone_primary_c"]    = [r[1] for r in t_res]
valid_df["tone_secondary"]    = [r[2] for r in t_res]
valid_df["tone_secondary_c"]  = [r[3] for r in t_res]
valid_df["emo_primary_raw"]   = [r[0] for r in e_res]
valid_df["emo_primary_c"]     = [r[1] for r in e_res]
valid_df["emo_secondary"]     = [r[2] for r in e_res]
valid_df["emo_secondary_c"]   = [r[3] for r in e_res]

# Informational = emotion model not confident enough to assign a label
valid_df["emo_primary"] = valid_df.apply(
    lambda r: "Informational" if r["emo_primary_c"] < EMOTION_CONF_FLOOR
              else r["emo_primary_raw"], axis=1
)
valid_df["emo_low_conf"] = valid_df["emo_primary_c"] < EMOTION_CONF_FLOOR

valid_df["subj_label"] = [r[0] for r in s_res]
valid_df["subj_conf"]  = [round(r[1], 3) for r in s_res]

has_legal = valid_df[CAPTION_COL].str.contains(LEGAL_KEYWORDS, case=False, na=False, regex=True)
has_death = valid_df[CAPTION_COL].str.contains(DEATH_KEYWORDS, case=False, na=False, regex=True)

valid_df, rule_counts = apply_rules(valid_df, has_legal, has_death)

# ─────────────────────────────────────────────────────────────────────
# MERGE BACK: insufficient rows get explicit sentinel values
# (not None/NaN) so every row in the output CSV has a value in every
# column — no blank cells, fully consistent schema across all rows.
# ─────────────────────────────────────────────────────────────────────
STR_SENTINEL  = "SKIPPED"    # placeholder for skipped rows in string cols
FLT_SENTINEL  = -1.0         # placeholder for skipped rows in float cols
BOOL_SENTINEL = False        # placeholder for skipped rows in bool cols

class_cols_str  = ["tone_primary_raw", "tone_secondary", "emo_primary_raw",
                   "emo_secondary", "emo_primary", "subj_label", "tone_primary"]
class_cols_flt  = ["tone_primary_c", "tone_secondary_c", "emo_primary_c",
                   "emo_secondary_c", "subj_conf", "reliability"]
class_cols_bool = ["emo_low_conf", "needs_review"]
class_cols = class_cols_str + class_cols_flt + class_cols_bool

full_out = corpus_df.copy()
for col in class_cols_str:  full_out[col] = STR_SENTINEL
for col in class_cols_flt:  full_out[col] = FLT_SENTINEL
for col in class_cols_bool: full_out[col] = BOOL_SENTINEL

# Write classified rows back by position (reset_index was called on valid_df)
full_out.loc[valid_mask.values, class_cols] = valid_df[class_cols].values

# ─────────────────────────────────────────────────────────────────────
# DATA INTEGRITY CHECK — catch any silent corruption before saving
# ─────────────────────────────────────────────────────────────────────
print("\nRunning data integrity check ...")
errors = []
# 1. Total row count preserved
if len(full_out) != n_total:
    errors.append(f"  ROW COUNT: expected {n_total}, got {len(full_out)}")
# 2. Valid rows must have a real tone label (not sentinel)
skipped_valid   = (valid_mask & (full_out["tone_primary"] == STR_SENTINEL)).sum()
unclassified    = skipped_valid
if unclassified > 0:
    errors.append(f"  MISSING CLASSIFICATIONS: {unclassified} valid rows still have sentinel tone")
# 3. Skipped rows must retain their sentinel (not overwritten with real labels)
skipped_mask    = ~valid_mask
overwritten     = (skipped_mask & (full_out["tone_primary"] != STR_SENTINEL)).sum()
if overwritten > 0:
    errors.append(f"  SENTINEL OVERWRITTEN: {overwritten} skipped rows have non-sentinel tone")
# 4. No NaN anywhere in the output
nan_counts = full_out[class_cols].isnull().sum()
nans_found = nan_counts[nan_counts > 0]
if len(nans_found) > 0:
    errors.append(f"  NaN VALUES FOUND in: {dict(nans_found)}")
# 5. Unicode round-trip check on captions (sample 100 rows)
sample_caps = full_out[CAPTION_COL].dropna().head(100)
if sample_caps.apply(lambda x: isinstance(x, str)).all():
    pass  # all strings preserved
else:
    errors.append("  UNICODE ISSUE: some captions are not strings after merge")

if errors:
    print("  \u26a0 INTEGRITY ISSUES FOUND:")
    for e in errors:
        print(e)
else:
    print(f"  \u2705 Integrity OK: {n_total} rows, {n_valid} classified, "
          f"{n_total - n_valid} skipped (sentinel='{STR_SENTINEL}'), 0 NaN")

# ─────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────
# 1. Full classified dataset (all rows including insufficient — NaN for unclassified)
out_csv = os.path.join(OUTPUT_DIR, "phase3_classified_all.csv")
print()
flush_to_disk(out_csv, full_out, tag=" [FINAL]")

# 2. Per-dataset summary
summary_rows = []
for (acc, plat, tl), grp in valid_df.groupby(["_account", "_platform", "_timeline"]):
    n  = len(grp)
    row = {"account": acc, "platform": plat, "timeline": tl, "n_rows": n}
    for lbl in TONE_NAMES:   # Persuasive IS already in TONE_NAMES — do not add again
        row[f"tone_{lbl.lower()}"] = round((grp["tone_primary"] == lbl).sum() / n * 100, 1)
    for lbl in EMOTION_NAMES + ["Informational"]:
        row[f"emo_{lbl.lower()}"]  = round((grp["emo_primary"] == lbl).sum() / n * 100, 1)
    for lbl in SUBJECTIVITY_NAMES:
        row[f"subj_{lbl.lower()}"] = round((grp["subj_label"] == lbl).sum() / n * 100, 1)
    row["mean_reliability"]   = round(grp["reliability"].mean(), 3)
    row["pct_needs_review"]   = round(grp["needs_review"].sum() / n * 100, 1)
    row["pct_low_conf_emo"]   = round(grp["emo_low_conf"].sum()  / n * 100, 1)
    summary_rows.append(row)

summary_df  = pd.DataFrame(summary_rows).sort_values(["account", "platform", "timeline"])
summary_csv = os.path.join(OUTPUT_DIR, "phase3_per_dataset_summary.csv")
flush_to_disk(summary_csv, summary_df, tag=" [FINAL]")

# 3. Run log
log_lines = [
    f"phase3_bulk_classify.py — run log",
    f"Model: {MODEL_ID}",
    f"Device: {'GPU' if DEVICE == 0 else 'CPU'}",
    f"Total rows in corpus: {n_total}",
    f"Valid (classified): {n_valid}",
    f"Insufficient (skipped): {n_total - n_valid}",
    f"Inference time: {elapsed_total:.1f} min",
    f"Batch size: {BATCH_SIZE}",
    "",
    "RULE COUNTS:",
    *[f"  {k}: {v}" for k, v in sorted(rule_counts.items())],
    "",
    "GLOBAL DISTRIBUTION (valid rows only):",
    f"  Tone: {dict(valid_df['tone_primary'].value_counts())}",
    f"  Emotion: {dict(valid_df['emo_primary'].value_counts())}",
    f"  Subjectivity: {dict(valid_df['subj_label'].value_counts())}",
    f"  Needs review: {valid_df['needs_review'].sum()} ({valid_df['needs_review'].mean()*100:.1f}%)",
    f"  Mean reliability: {valid_df['reliability'].mean():.3f}",
    "",
    "QUALITY GATE CHECK (compare to validation targets):",
]
n_v = len(valid_df)
gates = {
    "Aggressive":    (valid_df["tone_primary"] == "Aggressive").mean() * 100,
    "Defensive":     (valid_df["tone_primary"] == "Defensive").mean()  * 100,
    "Inspirational": (valid_df["tone_primary"] == "Inspirational").mean() * 100,
    "Neutral":       (valid_df["tone_primary"] == "Neutral").mean()    * 100,
    "Opinion":       (valid_df["subj_label"]   == "Opinion").mean()    * 100,
    "Low-conf":      valid_df["needs_review"].mean()                    * 100,
}
bounds = {
    "Aggressive": (25, 40), "Defensive": (0, 15),
    "Inspirational": (12, 25), "Neutral": (0, 8),
    "Opinion": (15, 28), "Low-conf": (0, 42),
}
for label, val in gates.items():
    lo, hi = bounds[label]
    status = "PASS" if lo <= val <= hi else "WARN"
    log_lines.append(f"  {label:<14} {val:.1f}%  ({lo}–{hi}%)  {status}")

log_path = os.path.join(OUTPUT_DIR, "phase3_run_log.txt")
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
try:
    with open(log_path, "rb") as fh:
        os.fsync(fh.fileno())
except OSError:
    pass
print(f"  [SAVED] {os.path.basename(log_path)}")

print("\n\u2705 Bulk run complete.")
print(f"   Main output:  phase3_classified_all.csv")
print(f"   Summary:      phase3_per_dataset_summary.csv")
print(f"   Log:          phase3_run_log.txt")
print(f"\n   Downstream: filter on needs_review==False for high-confidence analysis,")
print(f"   or use reliability > 0.60 as your analysis threshold.")


# ─────────────────────────────────────────────────────────────────────
# METHODOLOGY REPORT — printed to console + saved as .txt
# ─────────────────────────────────────────────────────────────────────
def build_methodology_report():
    SEP  = "=" * 72
    SEP2 = "-" * 72
    M    = []
    P    = M.append

    P(SEP)
    P("  PHASE 3 CLASSIFICATION METHODOLOGY REPORT")
    P(f"  Model: {MODEL_ID}")
    P(SEP)

    # [1] DATA COVERAGE
    P("\n[1] DATA COVERAGE")
    P(SEP2)
    P(f"  Total rows across all 24 datasets : {n_total:>6}")
    P(f"  Rows classified (valid)           : {n_valid:>6}  ({n_valid/n_total*100:.1f}%)")
    P(f"  Rows SKIPPED (insufficient)       : {n_total-n_valid:>6}  ({(n_total-n_valid)/n_total*100:.1f}%)")
    P("")
    P("  WHY ROWS ARE SKIPPED:")
    P(f"  A row is skipped if caption_for_sentiment has fewer than {MIN_WORDS} non-hashtag")
    P("  words. Zero-shot NLI models cannot reliably score 6 competing hypotheses")
    P("  on inputs of 2-3 words. Confidence scores become meaningless probability")
    P("  noise spread across labels. Hashtags (#TDP, #Jagan) are excluded from")
    P("  the word count as they carry metadata not rhetorical content.")
    P("  Skipped rows remain in phase3_classified_all.csv with NaN columns.")
    P("")
    P(f"  SKIP CRITERIA (MIN_WORDS = {MIN_WORDS}):")
    P("      Empty / null caption_for_sentiment            -> skipped")
    P("      Caption is only whitespace or 'nan'           -> skipped")
    P(f"     Fewer than {MIN_WORDS} non-hashtag words in caption      -> skipped")
    P("")
    P("  SKIP BREAKDOWN BY ACCOUNT:")
    P(f"  {'Account':<12} {'Total':>7} {'Skipped':>8} {'Skip%':>7} {'Classified':>12}")
    P(f"  {'-'*52}")
    for acc in sorted(corpus_df["_account"].unique()):
        sub = corpus_df[corpus_df["_account"] == acc]
        n_s = sub["_insufficient"].sum()
        n_c = len(sub) - n_s
        P(f"  {acc.upper():<12} {len(sub):>7} {n_s:>8} {n_s/len(sub)*100:>6.1f}% {n_c:>12}")
    P("")
    P("  EXAMPLES OF SKIPPED CAPTIONS (first 10):")
    for i, (_, r) in enumerate(corpus_df[corpus_df["_insufficient"]].head(10).iterrows(), 1):
        cap = str(r[CAPTION_COL]).replace("\n", " ").strip()[:110]
        wc  = len([w for w in re.findall(r"\b\w+\b", cap) if not w.startswith("#")])
        P(f"    [{i:02d}] words={wc}  [{r['_account'].upper()[:4]}|{r['_platform'][:2]}] \"{cap}\"")

    # [2] MODEL
    P(f"\n[2] CLASSIFICATION MODEL")
    P(SEP2)
    P(f"  Model  : {MODEL_ID}")
    P(f"  Task   : Zero-shot NLI-based text classification")
    P(f"  Device : {'GPU (CUDA)' if DEVICE == 0 else 'CPU'}")
    P(f"  Batch  : {BATCH_SIZE} captions per forward pass")
    P("")
    P("  WHY THIS MODEL:")
    P("  deberta-v3-large was trained on Multi-NLI + SNLI + ANLI and fine-tuned")
    P("  by MoritzLaurer for zero-shot classification. Selected over deberta-v3-base")
    P("  after multi-seed validation (seeds 42, 123, 250):")
    P("    Mean tone confidence : 0.68  (base: 0.51)")
    P("    Low-confidence rate  : 30%   (base: 46%)")
    P("    Neutral leakage      : 0.3%  (base: 2%)")
    P("  hypothesis_template='{}' passes hypotheses as complete descriptive")
    P("  sentences so the model scores them directly without its own NLI wrapper.")

    # [3] DIMENSIONS
    P(f"\n[3] THREE CLASSIFICATION DIMENSIONS")
    P(SEP2)
    P("  Each caption is independently classified on three dimensions in one pass.")
    P("")
    P("  RHETORICAL TONE  (softmax -- one label, labels compete):")
    P("    Aggressive | Persuasive | Defensive | Inspirational")
    P("    Victim_Narrative | Neutral")
    P("    Softmax forces a winner. Prevents all scores equalising near 0.17.")
    P("")
    P("  EMOTION  (sigmoid -- multi-label, scores independent):")
    P("    Anger | Pride | Fear | Hope | Trust -- and Informational (implicit)")
    P("    Sigmoid allows Anger+Fear simultaneously. Top two become primary/secondary.")
    P(f"   If primary score < {EMOTION_CONF_FLOOR} -> labelled 'Informational' (model too uncertain).")
    P("")
    P("  SUBJECTIVITY  (softmax -- one label):")
    P("    Fact | Opinion | Mixed")

    # [4] HYPOTHESES
    P(f"\n[4] EXACT HYPOTHESES (v8c.1)")
    P(SEP2)
    P("  TONE HYPOTHESES:")
    for name, hyp in zip(TONE_NAMES, TONE_HYPOTHESES):
        P(f"    [{name}]")
        for chunk in [hyp[i:i+80] for i in range(0, len(hyp), 80)]:
            P(f"      {chunk}")
    P("\n  EMOTION HYPOTHESES:")
    for name, hyp in zip(EMOTION_NAMES, EMOTION_HYPOTHESES):
        P(f"    [{name}]")
        for chunk in [hyp[i:i+80] for i in range(0, len(hyp), 80)]:
            P(f"      {chunk}")
    P(f"    [Informational] -- no hypothesis. Assigned when max emotion score < {EMOTION_CONF_FLOOR}.")
    P("\n  SUBJECTIVITY HYPOTHESES:")
    for name, hyp in zip(SUBJECTIVITY_NAMES, SUBJECTIVITY_HYPOTHESES):
        P(f"    [{name}]")
        for chunk in [hyp[i:i+80] for i in range(0, len(hyp), 80)]:
            P(f"      {chunk}")

    # [5] THRESHOLDS
    P(f"\n[5] CONFIDENCE THRESHOLDS")
    P(SEP2)
    thresholds = [
        (f"NEUTRAL_RECLASSIFY_THR  = {NEUTRAL_RECLASSIFY_THR}",
         "Neutral below this -> re-routed by R1 to secondary tone or Persuasive."),
        (f"TONE_SECONDARY_MIN_PRI  = {TONE_SECONDARY_MIN_PRI}",
         "Primary must score >= 0.50 for a secondary tone to be reported."),
        (f"TONE_SECONDARY_THR      = {TONE_SECONDARY_THR}",
         "Secondary tone reported only if score >= 0.22 (noise floor for 6 classes)."),
        (f"TONE_DEF_SECONDARY_THR  = {TONE_DEF_SECONDARY_THR}",
         "Defensive secondary needs higher bar (0.35) -- low Defensive = noise."),
        (f"EMOTION_CONF_FLOOR      = {EMOTION_CONF_FLOOR}",
         "Emotion below 0.40 -> Informational. Validated at 15-37% of corpus."),
        (f"EMOTION_SECONDARY_THR   = {EMOTION_SECONDARY_THR}",
         "Secondary emotion reported if score >= 0.35."),
        (f"NEEDS_REVIEW_TONE_THR   = {NEEDS_REVIEW_TONE_THR}",
         "Tone below 0.38 = near-random (random = 0.17 for 6 classes). -> needs_review."),
    ]
    for name, desc in thresholds:
        P(f"  {name}")
        P(f"    -> {desc}")
        P("")

    # [6] RULES
    P(f"\n[6] POST-PROCESSING RULES R1-R9")
    P(SEP2)
    P("  Rules correct systematic model errors found in iterative validation.")
    P("  Order: R1 -> R2 -> R4 -> R6 -> R8 -> R7 -> R5 -> R9")
    P("  Each rule only fires on rows still in the target tone at that point.")
    P("")
    rules_info = [
        ("R1", rule_counts.get("R1", 0),
         "Neutral (conf < 0.50) -> secondary tone or Persuasive",
         "Model uncertainty on Neutral almost always means a political post that was hedged, "
         "not a genuine condolence. Secondary tone used if available; otherwise Persuasive "
         "is the safe fallback as it is the most common category in this corpus."),
        ("R2", rule_counts.get("R2", 0),
         "Defensive + Pride/Trust/Hope -> Persuasive",
         "Real legal denials express Fear or Anger (persecution framing). "
         "Defensive + positive emotion = governance achievement post that the model "
         "mis-scored Defensive due to first-person political language."),
        ("R4", rule_counts.get("R4", 0),
         "Neutral + Pride/Hope >= 0.40 -> Inspirational;  Trust >= 0.60 -> Persuasive",
         "Tribute posts (NTR, Ambedkar, freedom fighters), rally posts, and youth-vision "
         "posts land Neutral with high Hope/Pride. These are definitionally Inspirational. "
         "Neutral + Trust >= 0.60 = welfare-confidence post -> Persuasive."),
        ("R6", rule_counts.get("R6", 0),
         "Neutral + Anger >= 0.70 -> Aggressive",
         "Very high Anger on a Neutral post means the model detected rage/contempt but "
         "was uncertain about tone category. Threshold set high (0.70) to avoid catching "
         "mild political frustration that belongs in Neutral."),
        ("R8", rule_counts.get("R8", 0),
         "Neutral + Fear >= 0.55 -> Aggressive",
         "Fear-dominant posts ('governance collapse', 'danger to the state') are alarm-raising "
         "attacks - Aggressive. Real condolences use grief/sadness, not Fear. "
         "0.55 threshold avoids catching mild concern."),
        ("R7", rule_counts.get("R7", 0),
         "Neutral + Informational emotion + no death keywords -> Persuasive",
         "Governance factsheets and event summaries produce Informational emotion with no "
         "death vocabulary. Genuine condolences always have death-adjacent words. Without "
         "them, Neutral is wrong - these are misclassified Persuasive posts."),
        ("R5", rule_counts.get("R5", 0),
         "Defensive + Informational + no legal keywords -> Persuasive",
         "Genuine legal denial posts always cite a specific legal event (FIR, court order, "
         "arrest). Defensive + Informational + no legal keyword = a governance or stats "
         "post the model confused with a Defensive frame."),
        ("R9", rule_counts.get("R9", 0),
         "Defensive + (Fear>=0.60 OR Anger>=0.70) + no legal + no death -> Aggressive",
         "Attack posts framed as 'we are victims of opponents' score Defensive. But if "
         "emotion is Fear or Anger (not grief), the intent is to attack. Legal keywords "
         "block this to protect real legal denial posts. Death keywords block it to protect "
         "condolence posts about opponents' injuries (the KCR condolence bug fix)."),
    ]
    for rule, count, trigger, reason in rules_info:
        P(f"  [{rule}]  Rows fired: {count}")
        P(f"         Condition : {trigger}")
        P(f"         Reason    :")
        for chunk in [reason[i:i+70] for i in range(0, len(reason), 70)]:
            P(f"           {chunk}")
        P("")

    # [7] KEYWORD GATES
    P(f"\n[7] KEYWORD GATES")
    P(SEP2)
    legal_ct = has_legal.sum()
    death_ct = has_death.sum()
    both_ct  = (has_legal & has_death).sum()
    P(f"  has_legal : {legal_ct} rows ({legal_ct/n_valid*100:.1f}%)  -- blocks rules R2, R5, R9")
    P(f"  has_death : {death_ct} rows ({death_ct/n_valid*100:.1f}%)  -- blocks rules R7, R8, R9")
    P(f"  both      : {both_ct} rows ({both_ct/n_valid*100:.1f}%)  -- political-death overlap (stays model output)")
    P("")
    P("  LEGAL_KEYWORDS include: arrest, FIR, bail, custody, remand, jail,")
    P("  CBI, ED raid, court order, case filed, prison, warrant, anticipatory bail,")
    P("  framed, false case, quash petition.")
    P("  PURPOSE: ensures genuine first-person legal denial posts stay Defensive")
    P("  or Victim_Narrative and are not converted to Persuasive or Aggressive.")
    P("")
    P("  DEATH_KEYWORDS include: death, died, demise, passed away, condolence,")
    P("  funeral, mourn, RIP, grief, bereavement, last rites, injured in accident,")
    P("  sustained an injury, get well soon, swift recovery, pray.*recover.")
    P("  DELIBERATELY EXCLUDED: 'hospital' and 'recover*' -- these match political")
    P("  attack posts about government hospital failures and recovery of stolen funds.")

    # [8] OUTPUT COLUMNS
    P(f"\n[8] OUTPUT COLUMNS IN phase3_classified_all.csv")
    P(SEP2)
    col_desc = [
        ("tone_primary_raw",  "Raw model tone before any rules applied"),
        ("tone_primary_c",    "Confidence for raw primary tone (0-1)"),
        ("tone_secondary",    f"2nd tone if score >= {TONE_SECONDARY_THR}, else 'None'"),
        ("tone_secondary_c",  "Confidence for secondary tone"),
        ("emo_primary_raw",   "Raw emotion label before Informational floor"),
        ("emo_primary_c",     "Confidence for raw primary emotion (sigmoid)"),
        ("emo_secondary",     f"2nd emotion if score >= {EMOTION_SECONDARY_THR}, else 'None'"),
        ("emo_secondary_c",   "Confidence for secondary emotion"),
        ("emo_primary",       f"FINAL emotion ('Informational' if conf < {EMOTION_CONF_FLOOR})"),
        ("emo_low_conf",      f"True if emo_primary_c < {EMOTION_CONF_FLOOR}"),
        ("subj_label",        "Fact / Opinion / Mixed"),
        ("subj_conf",         "Confidence for subjectivity label"),
        ("tone_primary",      "FINAL tone after all post-processing rules applied"),
        ("reliability",       "Mean of tone_c + emo_c + subj_c (0-1). Use > 0.60 for QA."),
        ("needs_review",      f"True if tone_c<{NEEDS_REVIEW_TONE_THR} OR emo_c<0.55 OR subj_c<0.55"),
    ]
    for col, desc in col_desc:
        P(f"  {col:<22}  {desc}")

    # [9] FINAL DISTRIBUTION
    P(f"\n[9] FINAL DISTRIBUTION ({n_valid} classified rows)")
    P(SEP2)
    P("  TONE:")
    for lbl, cnt_v in valid_df["tone_primary"].value_counts().items():
        bar = "\u2588" * int(cnt_v / n_valid * 40)
        P(f"    {lbl:<22} {cnt_v:>6}  ({cnt_v/n_valid*100:>5.1f}%)  {bar}")
    P("\n  EMOTION:")
    for lbl, cnt_v in valid_df["emo_primary"].value_counts().items():
        bar = "\u2588" * int(cnt_v / n_valid * 40)
        P(f"    {lbl:<22} {cnt_v:>6}  ({cnt_v/n_valid*100:>5.1f}%)  {bar}")
    P("\n  SUBJECTIVITY:")
    for lbl, cnt_v in valid_df["subj_label"].value_counts().items():
        bar = "\u2588" * int(cnt_v / n_valid * 40)
        P(f"    {lbl:<22} {cnt_v:>6}  ({cnt_v/n_valid*100:>5.1f}%)  {bar}")
    P("")
    P(f"  needs_review    : {valid_df['needs_review'].sum()} rows "
      f"({valid_df['needs_review'].mean()*100:.1f}%) -- treat as low-confidence")
    P(f"  mean reliability: {valid_df['reliability'].mean():.3f}"
      f"   (use > 0.60 as high-confidence filter for research analysis)")

    P(f"\n{SEP}")
    P("  END -- phase3_methodology_report.txt")
    P(SEP)
    return "\n".join(M)


print("\n" + "=" * 72)
print("  PRINTING METHODOLOGY REPORT ...")
print("=" * 72)
report_text = build_methodology_report()
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "phase3_methodology_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
try:
    with open(report_path, "rb") as fh:
        os.fsync(fh.fileno())
except OSError:
    pass
print(f"\n  [SAVED] {os.path.basename(report_path)}"
      f" -- {os.path.getsize(report_path)//1024} KB")

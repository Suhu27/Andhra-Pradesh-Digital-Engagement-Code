# ==============================================================
# phase1_build_dictionary.py
# Hashtag Sentiment Analysis — Phase 1: Master Dictionary Builder
#
# Pipeline:
#   1. Scan all 24 _Sentiment.xlsx → extract hashtags + frequency + entities
#   2. Recover original casing from caption_raw (CamelCase splitting requires it)
#   3. Apply CamelCase splitting to cased forms
#   4. Translate split text → English (Google Translate, 0.7s delay, checkpoint)
#   5. Sentiment inference (cardiffnlp/twitter-roberta-base-sentiment-latest)
#   6. Save Master_Hashtag_Dictionary.csv
#   7. Print top-20 verification table
#
# Run: python phase1_build_dictionary.py
# ==============================================================

import os
import re
import time
import numpy as np
import pandas as pd
import torch
from collections import Counter
from pathlib import Path
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit only this block
# ─────────────────────────────────────────────────────────────────
SENTIMENT_FOLDER   = r"C:\Users\Suhru\Desktop\Sentiment Analysis\sentiment_output"
OUTPUT_CSV         = "Master_Hashtag_Dictionary.csv"
CHECKPOINT_FILE    = "_phase1_translation_checkpoint.csv"
MODEL_NAME         = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE         = 8      # RTX 1650 4GB — safe value
MAX_LENGTH         = 128    # Hashtag phrases are 2–6 words
TRANSLATE_DELAY    = 0.7    # Seconds between Google Translate calls (non-negotiable)
CHECKPOINT_EVERY   = 50     # Save checkpoint every N new translations

ENTITY_NAMES = ["ncbn", "tdp", "ysjagan", "ysrcp"]
LABELS       = ["Negative", "Neutral", "Positive"]   # cardiffnlp label order

# ─────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*70}")
print(f"  Phase 1 — Master Hashtag Dictionary Builder")
print(f"{'='*70}")
print(f"  Device         : {DEVICE}")
if DEVICE == "cuda":
    print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"  Output         : {OUTPUT_CSV}")
print(f"  Batch size     : {BATCH_SIZE}  |  Max length: {MAX_LENGTH}")
print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────
# UTILITY: Extract entity name from filename
# ─────────────────────────────────────────────────────────────────
def get_entity(filename: str) -> str:
    name = os.path.basename(filename).lower()
    for e in ENTITY_NAMES:
        if name.startswith(e):
            return e
    return "unknown"


# ─────────────────────────────────────────────────────────────────
# STEP 1: COLLECT HASHTAGS FROM ALL 24 FILES
# ─────────────────────────────────────────────────────────────────
print("STEP 1: Scanning all 24 Sentiment files")
print("─"*70)

files = sorted(Path(SENTIMENT_FOLDER).glob("*_Sentiment.xlsx"))
if not files:
    raise FileNotFoundError(
        f"No *_Sentiment.xlsx files found in:\n  {SENTIMENT_FOLDER}\n"
        "Check the SENTIMENT_FOLDER path."
    )
print(f"  Found {len(files)} files.\n")

# hashtag_registry: lowercase_tag → {freq: int, entities: set}
hashtag_registry = {}

# casing_counter: lowercase_tag → Counter of cased versions
# Built from caption_raw to recover original CamelCase formatting
casing_counter = {}

HASHTAG_RE = re.compile(r'#(\w+)')

for fpath in tqdm(files, desc="  Scanning"):
    entity = get_entity(str(fpath))
    try:
        df = pd.read_excel(fpath, usecols=["Extracted_Hashtags", "caption_raw"])
    except Exception as e:
        try:
            # Some files may not have caption_raw — fall back gracefully
            df = pd.read_excel(fpath, usecols=["Extracted_Hashtags"])
            df["caption_raw"] = ""
        except Exception as e2:
            print(f"\n  [WARNING] Skipping {fpath.name}: {e2}")
            continue

    # ── Build hashtag registry from Extracted_Hashtags ──────────
    for cell in df["Extracted_Hashtags"].dropna():
        raw_tags = [t.strip() for t in str(cell).split(",") if t.strip()]
        for tag in raw_tags:
            if not tag.startswith("#"):
                tag = "#" + tag
            tag_lower = tag.lower()
            if tag_lower not in hashtag_registry:
                hashtag_registry[tag_lower] = {"frequency": 0, "entities": set()}
            hashtag_registry[tag_lower]["frequency"] += 1
            hashtag_registry[tag_lower]["entities"].add(entity)

    # ── Build casing map from caption_raw ───────────────────────
    # caption_raw preserves original casing: #BhavishyathukuGuarantee
    for text in df["caption_raw"].dropna():
        for match in HASHTAG_RE.finditer(str(text)):
            word = match.group(1)          # e.g. "BhavishyathukuGuarantee"
            key  = word.lower()            # e.g. "bhavishyathukuguarantee"
            if key not in casing_counter:
                casing_counter[key] = Counter()
            casing_counter[key][word] += 1

# Build best_casing: lowercase → most common original-cased form
best_casing = {
    k: "#" + v.most_common(1)[0][0]
    for k, v in casing_counter.items()
}

total_unique = len(hashtag_registry)
total_occurrences = sum(v["frequency"] for v in hashtag_registry.values())
recovered_casing = sum(1 for k in hashtag_registry if k.lstrip('#') in casing_counter)

print(f"\n  Unique hashtags      : {total_unique:,}")
print(f"  Total occurrences    : {total_occurrences:,}")
print(f"  Casing recovered     : {recovered_casing:,} / {total_unique:,} hashtags")
print(f"  Casing not found     : {total_unique - recovered_casing:,} (will use lowercase for translation)")


# ─────────────────────────────────────────────────────────────────
# STEP 2: CAMELCASE SPLITTING
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 2: CamelCase splitting (using recovered casing from caption_raw)")
print("─"*70)


def split_camelcase(hashtag: str) -> str:
    """
    Splits a hashtag into readable words, handling:
      - CamelCase          : JaganFailed       → Jagan Failed
      - Acronym+Word       : YSRCPRowdyism     → YSRCP Rowdyism
      - Acronym+Word       : APUnsafeForWomen  → AP Unsafe For Women
      - Digit+Letter       : 2024elections     → 2024 elections
      - Letter+Digit       : AP2024            → AP 2024
    """
    text = hashtag.lstrip('#')

    # Rule 1: [A-Z]+ followed by [A-Z][a-z] → insert space before last capital block
    # Handles acronym+word: YSRCPRowdyism → YSRCP Rowdyism
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)

    # Rule 2: lowercase/digit before uppercase → insert space
    # Handles: JaganFailed → Jagan Failed
    text = re.sub(r'([a-z\d])([A-Z])', r'\1 \2', text)

    # Rule 3: digit before letter → insert space
    # Handles: 2024elections → 2024 elections
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    # Rule 4: letter before digit → insert space
    # Handles: AP2024 → AP 2024
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)

    return re.sub(r'\s+', ' ', text).strip()


# Apply splitting using recovered casing where available
records = []
for tag_lower, meta in hashtag_registry.items():
    key = tag_lower.lstrip('#')
    # Use recovered casing if available; otherwise fall back to lowercase
    cased_tag = best_casing.get(key, tag_lower)
    split_text = split_camelcase(cased_tag)

    records.append({
        "hashtag_raw"          : tag_lower,       # lowercase — consistent lookup key
        "hashtag_cased"        : cased_tag,       # original cased form (internal, for audit)
        "hashtag_split"        : split_text,
        "hashtag_translated"   : None,
        "translation_changed"  : None,
        "frequency"            : meta["frequency"],
        "appears_in_entities"  : ", ".join(sorted(meta["entities"])),
        "score_negative"       : None,
        "score_neutral"        : None,
        "score_positive"       : None,
        "sentiment_label"      : None,
        "sentiment_confidence" : None,
    })

df_dict = pd.DataFrame(records)

# Inline verification of the splitting logic
print("\n  CamelCase Split Verification (Top 15 by frequency):")
print(f"\n  {'Hashtag (lowercase)':<35} {'Cased Form':<35} {'Split Result'}")
print(f"  {'─'*35} {'─'*35} {'─'*35}")
for _, row in df_dict.nlargest(15, "frequency").iterrows():
    print(
        f"  {str(row['hashtag_raw']):<35} "
        f"{str(row['hashtag_cased']):<35} "
        f"{str(row['hashtag_split'])}"
    )


# ─────────────────────────────────────────────────────────────────
# STEP 3: TRANSLATION (checkpoint/resume support)
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 3: Translating hashtag split text → English")
print("─"*70)
est_minutes = total_unique * TRANSLATE_DELAY / 60
print(f"  Hashtags to translate : {total_unique:,}")
print(f"  Delay per call        : {TRANSLATE_DELAY}s")
print(f"  Estimated time        : {est_minutes:.1f} minutes")
print(f"  Checkpoint every      : {CHECKPOINT_EVERY} new translations\n")

translator = GoogleTranslator(source='auto', target='english')

# ── Resume from checkpoint if it exists ─────────────────────────
already_translated = {}   # tag_lower → {"translated": str, "changed": bool}
if os.path.exists(CHECKPOINT_FILE):
    df_cp = pd.read_csv(CHECKPOINT_FILE)
    done_rows = df_cp[df_cp["hashtag_translated"].notna()]
    for _, row in done_rows.iterrows():
        already_translated[row["hashtag_raw"]] = {
            "translated" : row["hashtag_translated"],
            "changed"    : bool(row["translation_changed"])
        }
    print(f"  [RESUME] Checkpoint found — {len(already_translated):,} already translated, skipping.\n")

new_translated = 0
failed_count   = 0

for i, row in tqdm(df_dict.iterrows(), total=len(df_dict), desc="  Translating"):
    raw        = row["hashtag_raw"]
    split_text = row["hashtag_split"]

    # Resume: skip already done
    if raw in already_translated:
        df_dict.at[i, "hashtag_translated"] = already_translated[raw]["translated"]
        df_dict.at[i, "translation_changed"] = already_translated[raw]["changed"]
        continue

    # Translate
    try:
        result = translator.translate(split_text)
        time.sleep(TRANSLATE_DELAY)

        if not result or not result.strip():
            # Empty result — fall back to split text
            translated = split_text
            changed    = False
        else:
            translated = result.strip()
            changed    = (translated.lower() != split_text.lower())

    except Exception:
        # Any API error — fall back to split text silently
        translated = split_text
        changed    = False
        failed_count += 1

    df_dict.at[i, "hashtag_translated"]  = translated
    df_dict.at[i, "translation_changed"] = changed
    new_translated += 1

    # Periodic checkpoint save
    if new_translated % CHECKPOINT_EVERY == 0:
        df_dict.to_csv(CHECKPOINT_FILE, index=False)

# Final checkpoint flush before model inference
df_dict.to_csv(CHECKPOINT_FILE, index=False)

print(f"\n  New translations        : {new_translated:,}")
print(f"  Resumed from checkpoint : {len(already_translated):,}")
print(f"  Failed (used fallback)  : {failed_count:,}")
changed_total = df_dict["translation_changed"].sum()
print(f"  Translation changed text: {int(changed_total):,} ({changed_total/total_unique*100:.1f}%) — Romanised Telugu")


# ─────────────────────────────────────────────────────────────────
# STEP 4: SENTIMENT INFERENCE
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 4: Sentiment inference")
print("─"*70)
print(f"  Model      : {MODEL_NAME}")
print(f"  Batch size : {BATCH_SIZE}  |  Max length: {MAX_LENGTH}")
print(f"  Texts      : {total_unique:,}\n")

print("  Loading model...")
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
model_sa   = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model_sa   = model_sa.to(DEVICE)
model_sa.eval()
print(f"  Model ready on {DEVICE}.\n")


def predict_batch(texts: list) -> list:
    safe = [t if str(t).strip() else "no content" for t in texts]
    enc  = tokenizer(
        safe,
        return_tensors = "pt",
        padding        = True,
        truncation     = True,
        max_length     = MAX_LENGTH
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model_sa(**enc)
    return softmax(out.logits.cpu().numpy(), axis=1).tolist()


texts_to_score = df_dict["hashtag_translated"].tolist()
all_scores     = []

for i in tqdm(range(0, len(texts_to_score), BATCH_SIZE), desc="  Inference"):
    batch = texts_to_score[i : i + BATCH_SIZE]
    all_scores.extend(predict_batch(batch))

df_dict["score_negative"]       = [round(s[0], 6) for s in all_scores]
df_dict["score_neutral"]        = [round(s[1], 6) for s in all_scores]
df_dict["score_positive"]       = [round(s[2], 6) for s in all_scores]
df_dict["sentiment_label"]      = [LABELS[int(np.argmax(s))] for s in all_scores]
df_dict["sentiment_confidence"] = [round(float(max(s)), 6) for s in all_scores]

print("  Inference complete.")


# ─────────────────────────────────────────────────────────────────
# STEP 5: SAVE MASTER DICTIONARY
# ─────────────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("STEP 5: Saving Master_Hashtag_Dictionary.csv")
print("─"*70)

# Drop internal cased column (it's in the audit trail via hashtag_split)
df_dict.drop(columns=["hashtag_cased"], inplace=True)

COLUMN_ORDER = [
    "hashtag_raw",
    "hashtag_split",
    "hashtag_translated",
    "score_negative",
    "score_neutral",
    "score_positive",
    "sentiment_label",
    "sentiment_confidence",
    "frequency",
    "appears_in_entities",
    "translation_changed",
]
df_dict = df_dict[COLUMN_ORDER].sort_values("frequency", ascending=False).reset_index(drop=True)
df_dict.to_csv(OUTPUT_CSV, index=False)
print(f"  Saved: {OUTPUT_CSV}  ({total_unique:,} rows)")

# Clean up checkpoint now that we're done
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print(f"  Checkpoint file removed.")


# ─────────────────────────────────────────────────────────────────
# STEP 6: VERIFICATION PRINTOUT
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("VERIFICATION — Top 20 Most Frequent Hashtags")
print(f"{'='*70}")
print(
    f"\n  {'#':<4} "
    f"{'hashtag_raw':<32} "
    f"{'hashtag_split':<32} "
    f"{'hashtag_translated':<32} "
    f"{'Label':<10} "
    f"{'Conf':>6}  "
    f"{'Freq':>6}  "
    f"Chg?"
)
print(
    f"  {'─'*4} {'─'*32} {'─'*32} {'─'*32} "
    f"{'─'*10} {'─'*6}  {'─'*6}  {'─'*4}"
)

top20 = df_dict.head(20)
for i, row in top20.iterrows():
    split       = str(row["hashtag_split"])[:31]
    translated  = str(row["hashtag_translated"])[:31]
    changed_str = "YES" if row["translation_changed"] else "no"
    print(
        f"  {i+1:<4} "
        f"{str(row['hashtag_raw']):<32} "
        f"{split:<32} "
        f"{translated:<32} "
        f"{str(row['sentiment_label']):<10} "
        f"{row['sentiment_confidence']:>6.3f}  "
        f"{int(row['frequency']):>6}  "
        f"{changed_str}"
    )

# ── Distribution Summary ─────────────────────────────────────────
print(f"\n{'─'*70}")
print("DISTRIBUTION — All unique hashtags")
print(f"{'─'*70}")
for label, count in df_dict["sentiment_label"].value_counts().items():
    pct = count / total_unique * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<12}: {count:>5} ({pct:5.1f}%)  {bar}")

print(f"\n  Total unique hashtags  : {total_unique:,}")
print(f"  Avg confidence         : {df_dict['sentiment_confidence'].mean():.3f}")
print(f"  High conf  (> 0.80)    : {(df_dict['sentiment_confidence'] > 0.80).sum():,}")
print(f"  Medium conf (0.5–0.8)  : {((df_dict['sentiment_confidence'] >= 0.5) & (df_dict['sentiment_confidence'] <= 0.8)).sum():,}")
print(f"  Low conf   (< 0.50)    : {(df_dict['sentiment_confidence'] < 0.50).sum():,}  ← watch these carefully")

# ── Romanised Telugu Spot-Check ──────────────────────────────────
print(f"\n{'─'*70}")
print("ROMANISED TELUGU SPOT-CHECK — Top 15 where translation_changed=True")
print(f"{'─'*70}")
telugu_tags = df_dict[df_dict["translation_changed"] == True].head(15)
for _, row in telugu_tags.iterrows():
    print(
        f"  {str(row['hashtag_raw']):<35} "
        f"→ '{str(row['hashtag_translated'])}'  "
        f"[{row['sentiment_label']}, conf={row['sentiment_confidence']:.3f}]"
    )

# ── Per-Entity Hashtag Usage ─────────────────────────────────────
print(f"\n{'─'*70}")
print("PER-ENTITY HASHTAG COUNT (unique hashtags used by each party)")
print(f"{'─'*70}")
for entity in ENTITY_NAMES:
    count = df_dict["appears_in_entities"].str.contains(entity).sum()
    print(f"  {entity:<12}: {count:>5} unique hashtags")

shared = (df_dict["appears_in_entities"].str.contains(",")).sum()
print(f"\n  Shared across parties  : {shared:>5} hashtags (used by 2+ entities)")

print(f"\n{'='*70}")
print(f"[DONE] Phase 1 complete.")
print(f"       Review {OUTPUT_CSV} and the verification table above.")
print(f"       Paste the top-20 table and Romanised Telugu spot-check")
print(f"       back for political accuracy verification.")
print(f"       DO NOT run phase2_map_to_posts.py until dictionary is confirmed.")
print(f"{'='*70}\n")
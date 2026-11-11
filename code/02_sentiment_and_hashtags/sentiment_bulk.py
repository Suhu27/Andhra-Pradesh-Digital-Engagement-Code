# ============================================================
# sentiment_bulk.py
# Bulk Sentiment Analysis — All 24 Datasets
# Run: python sentiment_bulk.py
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import emoji
from pathlib import Path
import time

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATASETS_FOLDER = r"C:\Users\Suhru\Desktop\Sentiment Analysis"
OUTPUT_FOLDER   = r"C:\Users\Suhru\Desktop\Sentiment Analysis\sentiment_output"
MODEL_NAME      = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE      = 16
MAX_LENGTH      = 512

DATASET_FILES = [
    "ncbn_Facebook_JanMay2024_HashtagClassified.xlsx",
    "ncbn_Facebook_JunAug2023_HashtagClassified.xlsx",
    "ncbn_Facebook_SepDec2023_HashtagClassified.xlsx",
    "ncbn_Instagram_JanMay2024_HashtagClassified.xlsx",
    "ncbn_Instagram_JunAug2023_HashtagClassified.xlsx",
    "ncbn_Instagram_SepDec2023_HashtagClassified.xlsx",
    "tdp_Facebook_JanMay2024_HashtagClassified.xlsx",
    "tdp_Facebook_JunAug2023_HashtagClassified.xlsx",
    "tdp_Facebook_SepDec2023_HashtagClassified.xlsx",
    "tdp_Instagram_JanMay2024_HashtagClassified.xlsx",
    "tdp_Instagram_JunAug2023_HashtagClassified.xlsx",
    "tdp_Instagram_SepDec2023_HashtagClassified.xlsx",
    "ysjagan_Facebook_JanMay2024_HashtagClassified.xlsx",
    "ysjagan_Facebook_JunAug2023_HashtagClassified.xlsx",
    "ysjagan_Facebook_SepDec2023_HashtagClassified.xlsx",
    "ysjagan_Instagram_JanMay2024_HashtagClassified.xlsx",
    "ysjagan_Instagram_JunAug2023_HashtagClassified.xlsx",
    "ysjagan_Instagram_SepDec2023_HashtagClassified.xlsx",
    "ysrcp_Facebook_JanMay2024_HashtagClassified.xlsx",
    "ysrcp_Facebook_JunAug2023_HashtagClassified.xlsx",
    "ysrcp_Facebook_SepDec2023_HashtagClassified.xlsx",
    "ysrcp_Instagram_JanMay2024_HashtagClassified.xlsx",
    "ysrcp_Instagram_JunAug2023_HashtagClassified.xlsx",
    "ysrcp_Instagram_SepDec2023_HashtagClassified.xlsx",
]

LABELS = ["Negative", "Neutral", "Positive"]

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def clean_caption(text: str) -> str:
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', 'user', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'\s+', ' ', text).strip()
    if re.sub(r'[^\w]', '', text) == '':
        return ""
    return text


def predict_batch(texts, tokenizer, model):
    safe_texts = [t if t.strip() else "no content available" for t in texts]
    encoded = tokenizer(
        safe_texts, return_tensors='pt',
        padding=True, truncation=True, max_length=MAX_LENGTH
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        output = model(**encoded)
    return softmax(output.logits.cpu().numpy(), axis=1).tolist()


def process_file(filepath, tokenizer, model, output_folder):
    filename = os.path.basename(filepath)
    print(f"\n  Processing: {filename}")

    df = pd.read_excel(filepath)
    print(f"  Rows: {len(df)}")

    df['caption_for_sentiment'] = df['caption_en'].apply(clean_caption)

    texts = df['caption_for_sentiment'].tolist()
    all_scores = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"  [{filename[:30]}]", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        all_scores.extend(predict_batch(batch, tokenizer, model))

    df['score_negative']       = [s[0] for s in all_scores]
    df['score_neutral']        = [s[1] for s in all_scores]
    df['score_positive']       = [s[2] for s in all_scores]
    df['sentiment_label']      = [LABELS[int(np.argmax(s))] for s in all_scores]
    df['sentiment_confidence'] = [float(max(s)) for s in all_scores]

    # Save output with _Sentiment suffix
    out_name = filename.replace("_HashtagClassified.xlsx", "_Sentiment.xlsx")
    out_path = os.path.join(output_folder, out_name)
    df.to_excel(out_path, index=False)

    dist = df['sentiment_label'].value_counts().to_dict()
    print(f"  Saved → {out_name} | {dist}")
    return dist


# ─────────────────────────────────────────────
# LOAD MODEL ONCE
# ─────────────────────────────────────────────
print(f"\n[INFO] Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("[INFO] Model ready.\n")

# ─────────────────────────────────────────────
# PROCESS ALL FILES
# ─────────────────────────────────────────────
summary_rows = []
start_time = time.time()

for fname in DATASET_FILES:
    fpath = os.path.join(DATASETS_FOLDER, fname)
    if not os.path.exists(fpath):
        print(f"  [WARNING] File not found: {fname} — skipping.")
        continue
    dist = process_file(fpath, tokenizer, model, OUTPUT_FOLDER)
    parts = fname.replace("_HashtagClassified.xlsx", "").split("_")
    entity   = parts[0]
    platform = parts[1]
    period   = "_".join(parts[2:])
    summary_rows.append({
        "entity": entity, "platform": platform, "period": period,
        "file": fname, **dist
    })

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
elapsed = time.time() - start_time
print(f"\n{'='*70}")
print(f"ALL DONE in {elapsed/60:.1f} minutes")
print(f"{'='*70}")

summary_df = pd.DataFrame(summary_rows).fillna(0)
summary_path = os.path.join(OUTPUT_FOLDER, "_SUMMARY_AllDatasets.xlsx")
summary_df.to_excel(summary_path, index=False)
print(f"\nSummary saved → {summary_path}")
print(summary_df.to_string())
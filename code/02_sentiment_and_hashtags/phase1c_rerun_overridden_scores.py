# ============================================================
# phase1c_rerun_overridden_scores.py
# Re-runs model inference on ALL manually overridden rows.
#
# Design:
#   - Keeps manual sentiment_label (never overrides it)
#   - Updates score_negative, score_neutral, score_positive with
#     real softmax values from the model
#   - Sets sentiment_confidence = model's score for the ASSIGNED label
#     (not the max score). This is intentional:
#     - High confidence  → model agrees with the manual label
#     - Low confidence   → model disagrees (e.g. #ncbn ~0.08)
#     - Downstream conf > 0.5 filters correctly exclude disagreements
#
# Input : Master_Hashtag_Dictionary_Patch2.csv
# Output: Master_Hashtag_Dictionary_Final.csv
# ============================================================

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
from tqdm import tqdm

INPUT_FILE  = "Master_Hashtag_Dictionary_Patch2.csv"
OUTPUT_FILE = "Master_Hashtag_Dictionary_Final.csv"
MODEL_NAME  = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE  = 32
MAX_LENGTH  = 64   # hashtag translations are short — 64 tokens is ample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_TO_IDX = {"Negative": 0, "Neutral": 1, "Positive": 2}
IDX_TO_LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

print(f"[INFO] Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[INFO] GPU   : {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
print(f"\n[STEP 1] Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)

def parse_bool(val):
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ('true', '1', 'yes')

df['manual_override']    = df['manual_override'].fillna(False).apply(parse_bool)
df['translation_changed'] = df['translation_changed'].fillna(False).apply(parse_bool)

override_mask = df['manual_override'] == True
n_override    = override_mask.sum()

print(f"         Total rows             : {len(df):,}")
print(f"         Manually overridden    : {n_override}")

if n_override == 0:
    print("\n[ERROR] No manual_override=True rows found.")
    print("        Did phase1b_patch2.py run successfully?")
    exit(1)

override_df = df[override_mask].copy()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f"\n[STEP 2] Loading model: {MODEL_NAME}")
print("         (Uses cached weights — no download if already run once)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print("         Model ready.")

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def run_inference(texts: list) -> list:
    safe = [t if pd.notna(t) and str(t).strip() else "no content" for t in texts]
    enc  = tokenizer(safe, return_tensors='pt', padding=True,
                     truncation=True, max_length=MAX_LENGTH)
    enc  = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    return softmax(out.logits.cpu().numpy(), axis=1).tolist()

texts   = override_df['hashtag_translated'].tolist()
indices = override_df.index.tolist()

print(f"\n[STEP 3] Re-running inference on {n_override} overridden rows...")
all_scores = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Inferring"):
    batch = texts[i : i + BATCH_SIZE]
    all_scores.extend(run_inference(batch))

# ─────────────────────────────────────────────
# UPDATE SCORES
# Rule:
#   score_negative/neutral/positive ← real model softmax
#   sentiment_label                 ← unchanged (manual override)
#   sentiment_confidence            ← model's score for the assigned label
#     (intentionally NOT max-score — reflects model agreement)
# ─────────────────────────────────────────────
print(f"\n[STEP 4] Updating scores (keeping manual labels)...")
print(f"\n  {'Hashtag':<38} {'Label':<10} {'Neg':>6} {'Neu':>6} {'Pos':>6}  "
      f"{'Conf(label)':>11}  {'Model agrees?'}")
print("  " + "─"*95)

model_agrees_count = 0
model_disagrees    = []

for row_idx, scores in zip(indices, all_scores):
    neg, neu, pos = scores[0], scores[1], scores[2]
    label         = df.loc[row_idx, 'sentiment_label']   # keep manual label

    score_map = {"Negative": neg, "Neutral": neu, "Positive": pos}
    conf      = score_map.get(label, max(scores))         # conf = label's score

    # What the model would naturally predict
    model_pred = IDX_TO_LABEL[int(np.argmax(scores))]
    agrees     = (model_pred == label)
    if agrees:
        model_agrees_count += 1
    else:
        model_disagrees.append((df.loc[row_idx, 'hashtag_raw'], label, model_pred, conf))

    df.loc[row_idx, 'score_negative']       = round(neg, 6)
    df.loc[row_idx, 'score_neutral']        = round(neu, 6)
    df.loc[row_idx, 'score_positive']       = round(pos, 6)
    df.loc[row_idx, 'sentiment_confidence'] = round(conf, 6)

    agree_flag = "✓" if agrees else f"✗ (model→{model_pred})"
    h = df.loc[row_idx, 'hashtag_raw']
    print(f"  {h:<38} {label:<10} {neg:>6.3f} {neu:>6.3f} {pos:>6.3f}  {conf:>11.3f}  {agree_flag}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
dist      = df['sentiment_label'].value_counts()
freq_dist = df.groupby('sentiment_label')['frequency'].sum()
total_freq = df['frequency'].sum()

print(f"\n{'='*70}")
print(f"PHASE 1 COMPLETE — {OUTPUT_FILE}")
print(f"{'='*70}")

print(f"\n  Re-inference results ({n_override} rows):")
print(f"    Model agrees with manual label   : {model_agrees_count} / {n_override}")
print(f"    Model disagrees (label still kept): {len(model_disagrees)}")

if model_disagrees:
    print(f"\n  Disagreement detail:")
    for h, manual, model_pred, conf in model_disagrees:
        print(f"    {h:<38} manual={manual:<10} model={model_pred:<10} "
              f"conf={conf:.3f}  ← low conf signals distrust correctly")

print(f"\n  Final dictionary label distribution:")
print(f"\n  {'Label':<12} {'Unique #':<10} {'% Unique':<12} "
      f"{'Total Occurrences':<20} {'% Corpus'}")
print("  " + "─"*62)
for lbl in ["Negative", "Neutral", "Positive"]:
    u  = dist.get(lbl, 0)
    fo = freq_dist.get(lbl, 0)
    print(f"  {lbl:<12} {u:<10,} {u/len(df)*100:<12.1f} {fo:<20,} {fo/total_freq*100:.1f}%")

print(f"\n  NOTE: ~75-78% Neutral is expected and correct.")
print(f"        Name tags, location tags, event tags (#IndependenceDay, #kuppam,")
print(f"        #Tirupati) are genuinely sentiment-neutral. The 22-25% Neg+Pos")
print(f"        represent the politically charged subset — those are the")
print(f"        analytically interesting rows for Phase 2 mapping.")
print(f"\n  NEXT: Run  phase2_map_hashtag_sentiment.py")
print(f"        Input: {OUTPUT_FILE}  +  24 *_Sentiment.xlsx files")

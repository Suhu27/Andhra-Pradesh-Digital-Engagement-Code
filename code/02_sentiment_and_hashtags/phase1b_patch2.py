# ============================================================
# phase1b_patch2.py
# Applies Patch 2: 26 label corrections + 7 translation fixes
# Corrections verified per political audit.
#
# Input : Master_Hashtag_Dictionary_Patched.csv
# Output: Master_Hashtag_Dictionary_Patch2.csv
#
# Run BEFORE phase1c_rerun_overridden_scores.py
# Translation fixes are applied first so phase1c gets correct text
# ============================================================

import pandas as pd
import sys

INPUT_FILE  = "Master_Hashtag_Dictionary_Patched.csv"
OUTPUT_FILE = "Master_Hashtag_Dictionary_Patch2.csv"

# ─────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────
print(f"[STEP 1] Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"         Rows: {len(df)}")

# Normalise manual_override to actual booleans
def parse_bool(val):
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ('true', '1', 'yes')

df['manual_override']   = df['manual_override'].fillna(False).apply(parse_bool)
df['translation_changed'] = df['translation_changed'].fillna(False).apply(parse_bool)

existing_overrides = df['manual_override'].sum()
print(f"         Existing manual overrides: {existing_overrides}")

# ─────────────────────────────────────────────────────────────────
# STEP 3 FIRST — Translation fixes
# Applied before label corrections so phase1c infers on correct text
# Excluded: #abbaikilledbabai — label already Negative (correct);
#           translation fix only, manual_override stays False unless
#           it was already True
# ─────────────────────────────────────────────────────────────────
TRANSLATION_FIXES = {
    "#cmysjagan":              "Chief Minister YS Jagan",
    "#abbaikilledbabai":       "Boy killed uncle",
    "#ysraasara":              "YSR Aasara welfare scheme",
    "#ysrjaganannacolonies":   "Jagananna housing colonies",
    "#nadunedu":               "Nadu Nedu school renovation program",
    "#scamsternaidu":          "Scamster Naidu",
    "#tdpfordevelopment":      "TDP for Development",
}

print(f"\n[STEP 2] Applying translation fixes ({len(TRANSLATION_FIXES)} hashtags)...")
trans_applied = 0
trans_not_found = []
for hashtag, new_translation in TRANSLATION_FIXES.items():
    mask = df["hashtag_raw"] == hashtag
    if mask.sum() == 0:
        trans_not_found.append(hashtag)
        print(f"  [WARN]  {hashtag} — not found in dictionary")
        continue
    old = df.loc[mask, "hashtag_translated"].values[0]
    df.loc[mask, "hashtag_translated"] = new_translation
    df.loc[mask, "translation_changed"] = True
    trans_applied += 1
    print(f"  [TRANS] {hashtag:<35} '{old}'")
    print(f"          {'→':<35} '{new_translation}'")

print(f"\n  Applied: {trans_applied}  |  Not found: {len(trans_not_found)}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — Label corrections (26 hashtags)
#
# EXCLUDED from this list (already correct or Neutral-acceptable):
#   #raakadaliraa → Neutral is correct; mobilisation slogan reads
#                   emotionally flat to RoBERTa — not a misclassification
#   #abbaikilledbabai → already Negative; translation fix above is enough
# ─────────────────────────────────────────────────────────────────
LABEL_CORRECTIONS = {
    # YSRCP positive pride / identity tags → Positive
    "#cmysjagan":               "Positive",   # "Chief Minister YS Jagan"
    "#ysjagancares":            "Positive",
    "#prajagalam":              "Positive",
    "#yuvagalam":               "Positive",
    "#samajikasadhikarayatra":  "Positive",
    "#ysrjaganannacolonies":    "Positive",   # translation also fixed above
    "#ysraasara":               "Positive",   # translation also fixed above

    # TDP/NCBN positive support tags → Positive
    "#tdpjsptogether":          "Positive",
    "#iamwithbabu":             "Positive",
    "#babunumallirappidham":    "Positive",
    "#ntrliveson":              "Positive",
    "#voteforcycle":            "Positive",   # confirmed: ncbn, tdp entities only
    "#siddham":                 "Positive",
    "#babusuper6":              "Positive",

    # YSRCP attack tags on TDP/NCBN → Negative
    "#mosagadubabu":            "Negative",   # "Fraudster Babu" — direct attack
    "#pichijagan":              "Negative",   # "Mad/Crazy Jagan" — TDP taunt
    "#khaidino7691":            "Negative",   # CBN prison number — taunt from YSRCP
    "#scamsterchandrababu":     "Negative",   # translation also fixed above
    "#looterlokesh":            "Negative",
    "#bluffmasterchandrababu":  "Negative",
    "#dalitdhrohijagan":        "Negative",   # "Dalit betrayer Jagan" — TDP attack
    "#pensionsdongajagan":      "Negative",   # "Pensions thief Jagan"
    "#jarugujagan":             "Negative",   # "Chase Away Jagan"
    "#tdpjspbjpcollapse":       "Negative",   # YSRCP attack on alliance

    # TDP attack tags on YSRCP → Negative
    "#rayalaseemadrohicbn":     "Negative",   # YSRCP: "CBN betrayed Rayalaseema"
    "#richestcmjaganvspoor":    "Negative",
}

print(f"\n[STEP 3] Applying label corrections ({len(LABEL_CORRECTIONS)} hashtags)...")
labels_applied   = 0
labels_changed   = 0
labels_not_found = []
label_change_log = []

for hashtag, new_label in LABEL_CORRECTIONS.items():
    mask = df["hashtag_raw"] == hashtag
    if mask.sum() == 0:
        labels_not_found.append(hashtag)
        print(f"  [WARN]  {hashtag} — not found in dictionary")
        continue
    old_label = df.loc[mask, "sentiment_label"].values[0]
    old_conf  = df.loc[mask, "sentiment_confidence"].values[0]
    df.loc[mask, "sentiment_label"]  = new_label
    df.loc[mask, "manual_override"]  = True
    labels_applied += 1
    if old_label != new_label:
        labels_changed += 1
        direction = f"{old_label:<10} → {new_label}"
        print(f"  [LABEL] {hashtag:<35} {direction}  (conf was {old_conf:.3f})")
        label_change_log.append((hashtag, old_label, new_label, old_conf))
    else:
        print(f"  [OK]    {hashtag:<35} already {new_label} — override flagged only")

print(f"\n  Applied: {labels_applied}  |  Labels actually changed: {labels_changed}  "
      f"|  Not found: {len(labels_not_found)}")

if labels_not_found:
    print(f"\n  ⚠ Not found: {labels_not_found}")
    print("    Check spelling — hashtag_raw column uses lowercase with no spaces.")

# ─────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_FILE, index=False)

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
total_overrides = df["manual_override"].sum()
dist  = df["sentiment_label"].value_counts()
freq_dist = df.groupby("sentiment_label")["frequency"].sum()
total_freq = df["frequency"].sum()

print(f"\n{'='*70}")
print(f"PATCH 2 COMPLETE — {OUTPUT_FILE}")
print(f"{'='*70}")
print(f"\n  Total rows               : {len(df):,}")
print(f"  Total manual overrides   : {total_overrides}  (was {existing_overrides})")
print(f"  Net new overrides added  : {total_overrides - existing_overrides}")
print(f"\n  {'Label':<12} {'Unique #':<10} {'% Unique':<12} "
      f"{'Total Occurrences':<20} {'% Corpus'}")
print("  " + "─"*62)
for lbl in ["Negative", "Neutral", "Positive"]:
    u  = dist.get(lbl, 0)
    fo = freq_dist.get(lbl, 0)
    print(f"  {lbl:<12} {u:<10,} {u/len(df)*100:<12.1f} {fo:<20,} {fo/total_freq*100:.1f}%")

print(f"\n  NEXT: Run  phase1c_rerun_overridden_scores.py")
print(f"        Input will be: {OUTPUT_FILE}")

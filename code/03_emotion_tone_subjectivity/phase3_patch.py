# Extracted unchanged from emotionclass.py's final patch cell.
# phase3_patch.py — post-hoc fixes, no re-classification needed
import pandas as pd, re

df = pd.read_csv("phase3_classified_all.csv")
classified = df[df["tone_primary"] != "SKIPPED"].copy()

# ── FIX 1: Grief/Solidarity flag (replaces need for Sadness re-run) ────────
# Rows where tone signals solidarity/condolence AND death keywords present.
# This is a derived analytical flag, not re-classification.
DEATH_KW = (
    r"\bdeath\b|\bdied\b|\bdemise\b|passed away|condolence|"
    r"funeral|\bmourn\b|\bRIP\b|\bgrief\b|last rites|"
    r"injured in accident|sustained an injur|get well soon|"
    r"swift recovery|speedy recovery|pray.*recover"
)
has_death = df["caption_for_sentiment"].str.contains(
    DEATH_KW, case=False, na=False, regex=True
)
df["emo_grief_solidarity_flag"] = (
    has_death & df["tone_primary"].isin(["Neutral", "Victim_Narrative"])
)
# These 13 Neutral + subset of Victim_Narrative rows are your grief content.
# Use this flag in research instead of a Sadness label.

# ── FIX 2: Defensive semantic context note ─────────────────────────────────
# YSRCP Defensive rows are government defending its *treatment* of arrested
# opponent — not self-defence. Flag for researcher awareness.
LEGAL_KW = (
    r"\barrest\b|illegal detain|court order|\bFIR\b|\bbail\b|"
    r"\bcustody\b|\bjail\b|\bprison\b|\bremand\b|case filed"
)
has_legal = df["caption_for_sentiment"].str.contains(
    LEGAL_KW, case=False, na=False, regex=True
)
df["defensive_inverted_flag"] = (
    (df["tone_primary"] == "Defensive") &
    (df["_account"] == "ysrcp") &
    has_legal
)
# Use this to split Defensive into two sub-types in your analysis if needed.

# ── FIX 3: YSJAGAN low-n warning column ────────────────────────────────────
# Marks YSJAGAN rows so researchers know not to run significance tests on them
df["ysjagan_low_n_flag"] = df["_account"] == "ysjagan"

# ── FIX 4: Inspirational quality gate ─────────────────────────────────────
# Flags Inspirational rows with sarcasm/question signals
# (catches the "user do you have money" type contamination)
SARCASM_KW = r"\bdo you\b|\bwho\b\?|really\?|\bwhy\b\?|can you|how can"
df["inspirational_sarcasm_risk"] = (
    (df["tone_primary"] == "Inspirational") &
    df["caption_for_sentiment"].str.contains(SARCASM_KW, case=False, na=False, regex=True)
)

# ── SAVE ───────────────────────────────────────────────────────────────────
df.to_csv("phase3_classified_patched.csv", index=False, encoding="utf-8")
print("Patch complete.")
print(f"  Grief/solidarity flagged  : {df['emo_grief_solidarity_flag'].sum()}")
print(f"  Defensive inverted flagged: {df['defensive_inverted_flag'].sum()}")
print(f"  YSJAGAN low-n rows        : {df['ysjagan_low_n_flag'].sum()}")
print(f"  Inspirational sarcasm risk: {df['inspirational_sarcasm_risk'].sum()}")

import pandas as pd
import glob
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm

# ---------------- CONFIG ----------------
SLEEP_EVERY = 50       # sleep after every N translations
SLEEP_TIME = 1        # seconds to sleep
RETRY_SLEEP = 5         # seconds to sleep on failure
MAX_RETRIES = 3

translator = GoogleTranslator(source="auto", target="en")

csv_files = glob.glob("*.csv")

if not csv_files:
    raise ValueError("No CSV files found.")

def translate_text(text):
    for attempt in range(MAX_RETRIES):
        try:
            return translator.translate(text)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP)
            else:
                print("❌ Translation failed after retries:", text[:80])
                return ""

for file in csv_files:
    print(f"\n📄 Processing file: {file}")

    df = pd.read_csv(file, encoding="utf-8")

    if "caption_clean" not in df.columns:
        print("⚠️  Skipped (no caption_clean column)")
        continue

    if "caption_en" not in df.columns:
        df["caption_en"] = pd.NA

    # Rows needing translation
    mask = (
        df["caption_clean"].notna()
        & df["caption_clean"].astype(str).str.strip().ne("")
        & df["caption_en"].isna()
    )

    indices = df[mask].index.tolist()
    print(f"   🔤 Captions to translate: {len(indices)}")

    counter = 0

    for idx in tqdm(indices):
        text = str(df.at[idx, "caption_clean"]).strip()

        translated = translate_text(text)
        df.at[idx, "caption_en"] = translated

        counter += 1

        # Rate-limit protection
        if counter % SLEEP_EVERY == 0:
            time.sleep(SLEEP_TIME)

    # Save after each file (resume-safe)
    df.to_csv(
        file,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f"
    )

    print(f"   ✅ Finished translating {file}")

print("\n🎯 ALL FILES TRANSLATED SUCCESSFULLY.")

import pandas as pd
import glob
import emoji

# Get all CSV files in the folder
csv_files = glob.glob("*.csv")

if not csv_files:
    raise ValueError("No CSV files found.")

def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

for file in csv_files:
    print(f"\n📄 Processing: {file}")

    # Unicode-safe read
    df = pd.read_csv(file, encoding="utf-8")

    if "caption_raw" not in df.columns:
        print("⚠️  Skipped (no caption_raw column)")
        continue

    # Initialize caption_clean with NaN
    df["caption_clean"] = pd.NA

    # Identify valid captions
    valid_mask = df["caption_raw"].notna() & df["caption_raw"].astype(str).str.strip().ne("")

    valid_count = valid_mask.sum()
    print(f"   Valid captions found: {valid_count}")

    # Clean emojis only
    df.loc[valid_mask, "caption_clean"] = (
        df.loc[valid_mask, "caption_raw"]
        .astype(str)
        .apply(remove_emojis)
        .str.strip()
    )

    # Save back (Excel + Unicode safe, no scientific notation)
    df.to_csv(
        file,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10f"
    )

    print("   ✅ Emojis removed → caption_clean created")

print("\n🎯 STEP 1 complete: caption cleaning done.")

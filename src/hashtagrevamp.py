import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
CODEBOOK = BASE_DIR / "Hastags Thematic.xlsx"

CAPTION_PRIMARY   = "caption_raw"
CAPTION_FALLBACK  = "caption_en"
NEW_COL_EXTRACTED = "Extracted_Hashtags"
NEW_COL_THEME     = "Hashtag_Theme"
THEME_JOIN        = " | "
DROP_KEYWORDS     = ["hashtag_theme", "hashtag_themes", "extracted_hashtag", "extracted_hashtags"]
EXCLUDE_THEMES    = {"Digital / miscellaneous"}   # themes to suppress from output

HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)


def build_hashtag_map() -> dict:
    df = pd.read_excel(CODEBOOK, dtype=str, engine="openpyxl")
    mapping = {}
    for theme in df.columns:
        for cell in df[theme].dropna():
            tag = cell.strip().casefold()
            if tag.startswith("#"):
                mapping[tag] = theme
    return mapping


def get_caption_col(df: pd.DataFrame) -> pd.Series:
    for col in (CAPTION_PRIMARY, CAPTION_FALLBACK):
        if col in df.columns:
            return df[col].fillna("").astype(str)
    for col in df.columns:
        if "caption" in col.casefold():
            return df[col].fillna("").astype(str)
    return pd.Series([""] * len(df), dtype=str)


def row_to_extracted_and_theme(tags: list, tag2theme: dict) -> tuple:
    themes = []
    for t in tags:
        th = tag2theme.get(t)
        if th and th not in themes and th not in EXCLUDE_THEMES:
            themes.append(th)
    return (", ".join(tags), THEME_JOIN.join(themes))


def process_file(filepath: Path, tag2theme: dict):
    df = pd.read_excel(filepath, engine="openpyxl", keep_default_na=False)

    # Drop stale classification columns
    drop_cols = [c for c in df.columns if any(kw in c.casefold() for kw in DROP_KEYWORDS)]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Vectorised hashtag extraction
    source    = get_caption_col(df)
    tags_list = source.str.findall(HASHTAG_RE).apply(
        lambda lst: [t.casefold() for t in lst] if lst else []
    )

    extracted, themes = zip(*[row_to_extracted_and_theme(tags, tag2theme) for tags in tags_list])
    df[NEW_COL_EXTRACTED] = extracted
    df[NEW_COL_THEME]     = themes

    df.to_excel(filepath, index=False, engine="openpyxl")

    classified = sum(1 for t in themes if t)
    return len(df), classified


def main():
    tag2theme = build_hashtag_map()
    print(f"Loaded {len(tag2theme)} hashtag→theme mappings from '{CODEBOOK.name}'")
    print(f"Excluding themes: {EXCLUDE_THEMES}\n")

    files = sorted(
        f for f in BASE_DIR.glob("*.xlsx")
        if f.name != CODEBOOK.name and not f.name.startswith("~$")
    )
    total = len(files)
    print(f"Found {total} file(s) to process:\n")

    for i, f in enumerate(files, 1):
        try:
            rows, classified = process_file(f, tag2theme)
            print(f"  [{i}/{total}]  ✓  {f.name}  →  {rows} rows, {classified} classified")
        except PermissionError:
            print(f"  [{i}/{total}]  ✗  {f.name}  →  File is open in Excel. Close it and retry.")
        except Exception as e:
            print(f"  [{i}/{total}]  ✗  {f.name}  →  ERROR: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()

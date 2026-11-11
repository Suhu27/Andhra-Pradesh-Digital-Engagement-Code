# data_loader.py
# ──────────────────────────────────────────────────────────────────────────────
# Single source of truth for all data loading, cleaning, and validation.
# Every phase script imports from here — guarantees identical data across all
# analyses. Reads fresh from Excel every call — no intermediate files.
# ──────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "Analysis_Output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── CONSTANTS (imported by all phase scripts) ──────────────────────────────────
PARTY_COLOURS = {
    "ncbn"    : "#2271B3",   # Steel Blue
    "tdp"     : "#F5A800",   # Gold
    "ysjagan" : "#C0392B",   # Crimson
    "ysrcp"   : "#1AA068",   # Royal Green
}

PARTY_ORDER    = ["ncbn", "tdp", "ysjagan", "ysrcp"]
PERIOD_ORDER   = ["JunAug2023", "SepDec2023", "JanMay2024"]
PLATFORM_ORDER = ["Facebook", "Instagram"]

ER_MONTHLY  = "ER_Monthly"
ER_TIMELINE = "ER_Timeline"

# ── FIX 1: regex=False prevents | being treated as regex OR operator ───────────
THEME_SEP = " | "

PERIOD_DATE_RANGES = {
    "JunAug2023" : ("2023-06-01", "2023-08-31"),
    "SepDec2023" : ("2023-09-01", "2023-12-31"),
    "JanMay2024" : ("2024-01-01", "2024-05-31"),
}


# ── INTERNAL HELPERS ───────────────────────────────────────────────────────────

def _parse_filename(filepath: Path) -> dict:
    parts = filepath.stem.split("_")
    return {
        "account"  : parts[0],
        "platform" : parts[1],
        "period"   : parts[2],
    }


def _load_raw() -> pd.DataFrame:
    files = sorted(
        f for f in BASE_DIR.glob("*_HashtagClassified.xlsx")
        if not f.name.startswith("~$")
    )
    if not files:
        raise FileNotFoundError(f"No *_HashtagClassified.xlsx files found in {BASE_DIR}")

    print(f"[data_loader] Found {len(files)} source files\n")
    frames = []
    for f in files:
        meta = _parse_filename(f)
        df   = pd.read_excel(f, engine="openpyxl", keep_default_na=False)
        df["account"]  = meta["account"]
        df["platform"] = meta["platform"]
        df["period"]   = meta["period"]
        frames.append(df)
        print(f"  ✓  {f.name:<55}  {len(df):>5} rows")

    raw = pd.concat(frames, ignore_index=True)
    print(f"\n[data_loader] Total rows loaded : {len(raw):,}\n")
    return raw


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    # ── FIX 2: strip timezone BEFORE converting to Period to avoid warning ─────
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["date"] = df["date"].dt.tz_convert("Asia/Kolkata")
    date_naive   = df["date"].dt.tz_localize(None)
    df["month"]      = date_naive.dt.to_period("M").astype(str)
    df["month_date"] = date_naive.dt.to_period("M").dt.to_timestamp()

    # Engagement rates → percentage
    df[ER_MONTHLY]  = pd.to_numeric(df["Engagement_Rate_Monthly"],      errors="coerce") * 100
    df[ER_TIMELINE] = pd.to_numeric(df["Engagement_Rate_Timeline_Avg"], errors="coerce") * 100

    # Engagement counts
    for col in ["likes", "comments_count", "shares", "views"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Hashtag_Theme: empty string → NaN
    df["Hashtag_Theme"] = df["Hashtag_Theme"].replace("", np.nan)

    # Ordered categoricals
    df["account"]  = pd.Categorical(df["account"],  categories=PARTY_ORDER,    ordered=True)
    df["platform"] = pd.Categorical(df["platform"], categories=PLATFORM_ORDER, ordered=True)
    df["period"]   = pd.Categorical(df["period"],   categories=PERIOD_ORDER,   ordered=True)

    return df


def _validate_dates(df: pd.DataFrame):
    print("[data_loader] Date range validation:")
    for period, (start, end) in PERIOD_DATE_RANGES.items():
        subset = df[df["period"] == period]
        if subset.empty:
            print(f"  ⚠️  {period}: no data found")
            continue
        ts_start = pd.Timestamp(start, tz="Asia/Kolkata")
        ts_end   = pd.Timestamp(end,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
        out      = subset[(subset["date"] < ts_start) | (subset["date"] >= ts_end)]
        a_start  = subset["date"].min().strftime("%Y-%m-%d")
        a_end    = subset["date"].max().strftime("%Y-%m-%d")
        if not out.empty:
            print(f"  ⚠️  {period}: {len(out)} post(s) outside expected range "
                  f"| actual: {a_start} → {a_end}")
        else:
            print(f"  ✓  {period}: all dates valid | actual: {a_start} → {a_end}")
    print()


def _spot_check_engagement(df: pd.DataFrame):
    raw_eng = df["likes"] + df["shares"] + df["comments_count"]
    r1 = df[(raw_eng > 0) & (df[ER_MONTHLY].isna() | (df[ER_MONTHLY] == 0))]
    r2 = df[(raw_eng == 0) & (df[ER_MONTHLY] > 0)]
    print("[data_loader] Engagement rate spot-check:")
    print(f"  Engagement > 0 but ER_Monthly is 0 or NaN : {len(r1):,}")
    print(f"  Engagement = 0 but ER_Monthly > 0         : {len(r2):,}")
    print(f"  NaN in ER_Monthly                         : {df[ER_MONTHLY].isna().sum():,}")
    print(f"  NaN in ER_Timeline                        : {df[ER_TIMELINE].isna().sum():,}")
    print()


def _flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df["is_outlier"] = False
    print("[data_loader] Outlier flagging (IQR per account on ER_Monthly):")
    for account in PARTY_ORDER:
        mask = df["account"] == account
        q1   = df.loc[mask, ER_MONTHLY].quantile(0.25)
        q3   = df.loc[mask, ER_MONTHLY].quantile(0.75)
        iqr  = q3 - q1
        lo   = q1 - 1.5 * iqr
        hi   = q3 + 1.5 * iqr
        flag = mask & ((df[ER_MONTHLY] < lo) | (df[ER_MONTHLY] > hi))
        df.loc[flag, "is_outlier"] = True
        n   = flag.sum()
        tot = mask.sum()
        print(f"  {account:<12} : {n:4d} / {tot:4d} posts flagged "
              f"({n/tot*100:.1f}%)  |  IQR bounds: [{lo:.4f}%, {hi:.4f}%]")
    print()
    return df


def _build_themed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes multi-theme posts so each theme is treated independently.
    CRITICAL: regex=False ensures ' | ' is treated as a literal string,
    not as a regex OR operator (which would split on spaces).
    theme_count records how many themes the original post had.
    Engagement is fully attributed to each theme independently.
    """
    themed = df[df["Hashtag_Theme"].notna()].copy()

    # ── FIX 1 APPLIED HERE: regex=False is the critical fix ───────────────────
    themed["Hashtag_Theme"] = themed["Hashtag_Theme"].str.split(THEME_SEP, regex=False)
    themed["theme_count"]   = themed["Hashtag_Theme"].str.len()
    themed = themed.explode("Hashtag_Theme").reset_index(drop=True)
    themed["Hashtag_Theme"] = themed["Hashtag_Theme"].str.strip()
    themed = themed[themed["Hashtag_Theme"].ne("")].copy()
    return themed


def _print_validation_summary(master: pd.DataFrame, themed: pd.DataFrame):
    sep = "─" * 62

    print(f"\n{sep}")
    print("MASTER DATAFRAME SUMMARY")
    print(sep)
    print(f"  Total posts              : {len(master):,}")
    print(f"  Date range (IST)         : "
          f"{master['date'].min().strftime('%Y-%m-%d')} → "
          f"{master['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Posts WITH theme         : {master['Hashtag_Theme'].notna().sum():,}")
    print(f"  Posts WITHOUT theme      : {master['Hashtag_Theme'].isna().sum():,}")
    print(f"  Outlier posts flagged    : {master['is_outlier'].sum():,}")
    print(f"  ER_Monthly  range        : "
          f"{master[ER_MONTHLY].min():.4f}% → {master[ER_MONTHLY].max():.4f}%")
    print(f"  ER_Monthly  median       : {master[ER_MONTHLY].median():.4f}%")
    print(f"  ER_Timeline range        : "
          f"{master[ER_TIMELINE].min():.4f}% → {master[ER_TIMELINE].max():.4f}%")
    print(f"  ER_Timeline median       : {master[ER_TIMELINE].median():.4f}%")

    print(f"\n  Posts per Account × Platform:")
    for (acc, plat), n in master.groupby(["account", "platform"], observed=True).size().items():
        print(f"    {acc:<12} {plat:<12} : {n:,}")

    print(f"\n  Posts per Account × Period:")
    for (acc, per), n in master.groupby(["account", "period"], observed=True).size().items():
        print(f"    {acc:<12} {per:<15} : {n:,}")

    print(f"\n{sep}")
    print("THEMED DATAFRAME SUMMARY (exploded per theme)")
    print(sep)
    print(f"  Total theme-rows         : {len(themed):,}")
    print(f"  Unique themes            : {len(themed['Hashtag_Theme'].unique())}")

    print(f"\n  Theme counts (all accounts combined):")
    for theme, count in themed["Hashtag_Theme"].value_counts().items():
        print(f"    {theme:<45} : {count:,}")

    print(f"\n  Multi-theme post breakdown (original posts):")
    mc = (
        master[master["Hashtag_Theme"].notna()]["Hashtag_Theme"]
        .str.split(THEME_SEP, regex=False)
        .str.len()
        .value_counts()
        .sort_index()
    )
    for n_themes, count in mc.items():
        print(f"    Posts with {n_themes:>2} theme(s) : {count:,}")

    print(sep + "\n")


# ── PUBLIC API ─────────────────────────────────────────────────────────────────

def load_master(verbose: bool = True) -> pd.DataFrame:
    """All posts. Use for volume, engagement, timeline, platform analyses."""
    raw    = _load_raw()
    master = _clean(raw)
    if verbose:
        _validate_dates(master)
        _spot_check_engagement(master)
    master = _flag_outliers(master)
    if verbose:
        print("[data_loader] master_df ready\n")
    return master


def load_themed(verbose: bool = True) -> pd.DataFrame:
    """Exploded themed posts only. Use for all thematic analyses."""
    master = load_master(verbose=verbose)
    themed = _build_themed(master)
    if verbose:
        _print_validation_summary(master, themed)
    return themed


def load_both(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (master_df, themed_df). Use when a phase needs both — reads Excel once."""
    raw    = _load_raw()
    master = _clean(raw)
    if verbose:
        _validate_dates(master)
        _spot_check_engagement(master)
    master = _flag_outliers(master)
    themed = _build_themed(master)
    if verbose:
        _print_validation_summary(master, themed)
    return master, themed


# ── STANDALONE VALIDATION ──────────────────────────────────────────────────────
if __name__ == "__main__":
    master, themed = load_both(verbose=True)
    print(f"✅  master_df : {len(master):,} rows")
    print(f"✅  themed_df : {len(themed):,} rows")
    print(f"\ndata_loader.py is ready. Run any phase script next.")

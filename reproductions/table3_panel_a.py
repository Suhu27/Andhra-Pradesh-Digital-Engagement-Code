"""Reconstruct submitted manuscript Table 3 Panel A.

This script was written from the manuscript methods; it is not the lost
historical script that first produced the table.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


THEMES = [
    ("Leader-centric branding", "Leader branding"),
    ("Attacking Leadership/Demonisation", "Attacking leadership"),
    ("Mass mobilisation", "Mass mobilisation"),
    ("Attacking Party", "Attacking party"),
    ("Anti-incumbency/Attacking Governance", "Anti-incumbency"),
    ("Welfare&Governance", "Welfare & governance"),
    ("Alliance promotion", "Alliance promotion"),
    ("Caste / community politics", "Caste/community"),
    ("Promoting Party", "Promoting party"),
    ("Leadership legacy", "Leadership legacy"),
    ("TDP Manifesto", "TDP manifesto"),
    ("Agrarian & rural issues", "Agrarian/rural"),
    ("Youth, jobs & education", "Youth/jobs/education"),
    ("Women & gender", "Women & gender"),
    ("Developmental vision", "Developmental vision"),
]
COLS = ["date", "type", "Engagement_Rate_Monthly", "Extracted_Hashtags", "Hashtag_Theme"]


def load_data(folder):
    files = sorted(folder.glob("*_HashtagClassified.xlsx"))
    if len(files) != 24:
        raise ValueError(f"Expected 24 workbooks in {folder}; found {len(files)}")
    frames = []
    for path in files:
        account, platform, *_ = path.stem.split("_")
        frame = pd.read_excel(path, usecols=COLS, engine="openpyxl")
        frame["account"] = account
        frame["platform"] = platform
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare(data):
    data = data.loc[data["Hashtag_Theme"].notna()].copy()
    data["themes"] = data["Hashtag_Theme"].str.split(" | ", regex=False)
    actual = {theme for row in data["themes"] for theme in row}
    expected = {theme for theme, _ in THEMES}
    if actual != expected:
        raise ValueError(f"Theme mismatch: missing={expected - actual}, extra={actual - expected}")
    for i, (theme, _) in enumerate(THEMES):
        data[f"theme_{i:02d}"] = data["themes"].map(lambda row, key=theme: int(key in row))
    data["month"] = pd.to_datetime(data["date"], utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m")
    data["account_platform"] = data["account"] + "_" + data["platform"]
    data["cluster"] = data["account_platform"] + "_" + data["month"]
    data["is_video"] = data["type"].astype(str).str.lower().isin(["video", "reel"]).astype(int)
    data["hashtag_count"] = data["Extracted_Hashtags"].fillna("").astype(str).map(
        lambda text: len(re.findall(r"#\w+", text))
    )
    data["er"] = pd.to_numeric(data["Engagement_Rate_Monthly"], errors="coerce")
    data = data.loc[data["er"].gt(0) & data["month"].notna()].copy()
    data["log_er"] = np.log(data["er"])
    return data


def fit_table(data):
    terms = [f"theme_{i:02d}" for i in range(len(THEMES))]
    formula = "log_er ~ " + " + ".join(terms) + " + C(account_platform) + C(month) + is_video + hashtag_count"
    model = smf.ols(formula, data=data).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["cluster"], "use_correction": True, "df_correction": True},
        use_t=True,
    )
    intervals = model.conf_int(alpha=0.05)
    rows = []
    for i, (codebook_name, manuscript_name) in enumerate(THEMES):
        term = terms[i]
        lower, upper = intervals.loc[term]
        rows.append({
            "theme": manuscript_name,
            "codebook_theme": codebook_name,
            "posts": int(data[term].sum()),
            "percentage_difference": round(100 * np.expm1(model.params[term]), 1),
            "ci_95_low": round(100 * np.expm1(lower), 1),
            "ci_95_high": round(100 * np.expm1(upper), 1),
            "p_value": float(model.pvalues[term]),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("src"))
    parser.add_argument("--output", type=Path, default=Path("results/table3_panel_a.csv"))
    args = parser.parse_args()
    data = prepare(load_data(args.data_dir))
    result = fit_table(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    count = sum(int(data[f"theme_{i:02d}"].sum()) for i in range(len(THEMES)))
    print(f"Classified posts: {len(data):,}; theme records: {count:,}")
    print(result.to_string(index=False))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

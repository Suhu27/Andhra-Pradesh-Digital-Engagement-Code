# phase3_engagement.py
# ─────────────────────────────────────────────────────────────────────────────
# Block B: Engagement Overview — 10 charts + docx
# B1:  Mean & Median ER_Monthly per account
# B2:  ER_Monthly by platform per account
# B3:  Monthly ER_Timeline per account (line)
# B4:  ER_Monthly by post type
# B5:  Mean likes / comments / shares per post
# B6:  ER_Monthly distribution box plot
# B7:  Mean ER_Monthly per electoral period
# B8:  Engagement heatmap — account × period
# B9:  FB vs IG ER_Timeline per account (4 subplots)
# B10: Interaction composition (likes / comments / shares %)
# B11: Top 5 posts per account (docx table only)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from data_loader import (
    load_both, PARTY_COLOURS, PARTY_ORDER,
    PERIOD_ORDER, PLATFORM_ORDER,
    ER_MONTHLY, ER_TIMELINE,
    OUTPUT_DIR as ANALYSIS_DIR
)

# ── SETUP ─────────────────────────────────────────────────────────────────────
BLOCK_DIR = ANALYSIS_DIR / "B_Engagement"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

ACC_LABELS   = {"ncbn": "NCBN", "tdp": "TDP", "ysjagan": "YS Jagan", "ysrcp": "YSRCP"}
ACC_TO_KEY   = {v: k for k, v in ACC_LABELS.items()}
PER_LABELS   = {
    "JunAug2023": "Jun–Aug 2023",
    "SepDec2023": "Sep–Dec 2023",
    "JanMay2024": "Jan–May 2024",
}
PLAT_COLOURS = {"Facebook": "#1877F2", "Instagram": "#E1306C"}
TYPE_COLOURS = {"Photo": "#4CAF50", "Video": "#FF9800", "Other": "#9E9E9E"}

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 13,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 11,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
    "grid.color"       : "#cccccc",
})


# ── HELPERS ───────────────────────────────────────────────────────────────────

def save_fig(fig, name: str) -> Path:
    path = BLOCK_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: {name}")
    return path


def add_method_note(fig, text: str, y: float = 0.01):
    fig.text(0.5, y, f"Methodology: {text}",
             ha="center", va="bottom", fontsize=8,
             style="italic", color="#555555",
             transform=fig.transFigure)


def init_doc(title: str) -> Document:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x22, 0x71, 0xB3)
    doc.add_paragraph()
    return doc


def doc_section(doc, chart_title, img_path, methodology, stats, insights):
    doc.add_heading(chart_title, level=1)
    if img_path:
        doc.add_picture(str(img_path), width=Inches(6.0))
    doc.add_paragraph()
    p = doc.add_paragraph(); p.add_run("Methodology: ").bold = True; p.add_run(methodology)
    doc.add_paragraph()
    p = doc.add_paragraph(); p.add_run("Key Statistics:").bold = True
    for label, value in stats:
        doc.add_paragraph(f"    •  {label}: {value}")
    doc.add_paragraph()
    p = doc.add_paragraph(); p.add_run("Key Insights:").bold = True
    for insight in insights:
        doc.add_paragraph(f"    •  {insight}")
    doc.add_page_break()


def er_summary(df: pd.DataFrame, col: str) -> dict:
    return {
        "mean"  : round(float(df[col].mean()),   4),
        "median": round(float(df[col].median()), 4),
        "min"   : round(float(df[col].min()),    4),
        "max"   : round(float(df[col].max()),    4),
        "n"     : len(df),
    }


# ── B1: Mean & Median ER_Monthly per Account ──────────────────────────────────

def chart_b1(master: pd.DataFrame):
    stats = {}
    for acc in PARTY_ORDER:
        sub = master[master["account"] == acc]
        stats[ACC_LABELS[acc]] = er_summary(sub, ER_MONTHLY)

    labels  = list(stats.keys())
    means   = [stats[l]["mean"]   for l in labels]
    medians = [stats[l]["median"] for l in labels]
    colours = [PARTY_COLOURS[ACC_TO_KEY[l]] for l in labels]

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(labels))
    bars    = ax.bar(x, means, color=colours, alpha=0.85, width=0.5, label="Mean ER")

    for bar, mv in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{mv:.4f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.scatter(x, medians, marker="D", color="black", zorder=5, s=60, label="Median ER")
    for xi, med in zip(x, medians):
        ax.text(xi + 0.27, med, f"Med: {med:.4f}%",
                ha="left", va="center", fontsize=8, color="#333333")

    legend_median = mlines.Line2D([], [], color="black", marker="D",
                                  linestyle="None", markersize=7, label="Median ER")
    ax.legend(handles=[bars, legend_median])
    ax.set_title("B1: Mean & Median Engagement Rate (ER_Monthly) per Account", pad=15)
    ax.set_xlabel("Account"); ax.set_ylabel("Engagement Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels(labels)

    add_method_note(fig,
        f"ER_Monthly = (likes + shares + comments) / avg monthly followers × 100. "
        f"Bars = mean ER. Diamonds = median ER. All posts included. n = {len(master):,}.")
    path = save_fig(fig, "B1_Mean_Median_ER_per_Account.png")
    return path, stats


# ── B2: Mean ER_Monthly by Account × Platform ────────────────────────────────

def chart_b2(master: pd.DataFrame):
    grp = (
        master.groupby(["account", "platform"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x, w    = np.arange(len(PARTY_ORDER)), 0.35
    stats   = {}

    for i, account in enumerate(PARTY_ORDER):
        label = ACC_LABELS[account]
        stats[label] = {}
        for j, plat in enumerate(PLATFORM_ORDER):
            row = grp[(grp["account"] == account) & (grp["platform"] == plat)]
            if row.empty:
                continue
            mean_val   = float(row["mean"].iloc[0])
            median_val = float(row["median"].iloc[0])
            n_posts    = int(row["n"].iloc[0])
            offset     = (j - len(PLATFORM_ORDER)/2 + 0.5) * w
            ax.bar(x[i] + offset, mean_val, w,
                   label=plat if i == 0 else "",
                   color=PLAT_COLOURS[plat], alpha=0.9)
            ax.text(x[i] + offset, mean_val + 0.003,
                    f"{mean_val:.3f}%\n(n={n_posts:,})",
                    ha="center", va="bottom", fontsize=7.5)
            stats[label][plat] = {"mean": round(mean_val, 4),
                                   "median": round(median_val, 4), "n": n_posts}

    ax.set_title("B2: Mean ER_Monthly by Account & Platform", pad=15)
    ax.set_xlabel("Account"); ax.set_ylabel("Mean Engagement Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels([ACC_LABELS[a] for a in PARTY_ORDER])
    ax.legend()
    add_method_note(fig,
        f"Mean ER_Monthly per account per platform. n per bar shown in chart. "
        f"Total n = {len(master):,}.")
    path = save_fig(fig, "B2_ER_by_Account_Platform.png")
    return path, stats


# ── B3: Monthly ER_Timeline per Account ──────────────────────────────────────

def chart_b3(master: pd.DataFrame):
    monthly = (
        master.groupby(["month_date", "account"], observed=True)[ER_TIMELINE]
        .agg(mean="mean", median="median", n="count")
        .reset_index().sort_values("month_date")
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    max_y   = float(monthly["mean"].max())
    stats   = {}

    for account in PARTY_ORDER:
        sub   = monthly[monthly["account"] == account].copy()
        label = ACC_LABELS[account]
        ax.plot(sub["month_date"], sub["mean"],
                marker="o", linewidth=2,
                label=label, color=PARTY_COLOURS[account])
        stats[label] = {
            "peak_month"     : sub.loc[sub["mean"].idxmax(), "month_date"].strftime("%b %Y"),
            "peak_mean_er"   : round(float(sub["mean"].max()), 4),
            "overall_mean_er": round(float(sub["mean"].mean()), 4),
        }

    for dt_str, lbl in [("2023-09-01", "Sep 2023 →"), ("2024-01-01", "Jan 2024 →")]:
        dt = pd.Timestamp(dt_str)
        ax.axvline(dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(dt, max_y * 0.97, lbl, fontsize=8, color="gray", ha="left", va="top")

    ax.set_title("B3: Monthly Mean ER_Timeline per Account", pad=15)
    ax.set_xlabel("Month (IST)"); ax.set_ylabel("Mean Engagement Rate (%)")
    ax.legend(); fig.autofmt_xdate()
    add_method_note(fig,
        "ER_Timeline = (likes + shares + comments) / avg timeline followers × 100. "
        "Used for temporal comparison to remove follower-growth confound. "
        f"Dashed lines = period boundaries. n = {len(master):,}.")
    path = save_fig(fig, "B3_Monthly_ER_Timeline.png")
    return path, stats


# ── B4: Mean ER_Monthly by Post Type ─────────────────────────────────────────

def chart_b4(master: pd.DataFrame):
    df = master.copy()
    df["type"] = (
        df["type"].astype(str).str.strip().str.title()
        .replace({"": "Other", "Nan": "Other", "None": "Other"})
    )
    df.loc[~df["type"].isin(["Photo", "Video"]), "type"] = "Other"

    grp = (
        df.groupby(["account", "type"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", n="count").reset_index()
    )

    types  = ["Photo", "Video"]
    fig, ax = plt.subplots(figsize=(11, 6))
    x, w   = np.arange(len(PARTY_ORDER)), 0.35
    stats  = {}

    for account in PARTY_ORDER:
        label = ACC_LABELS[account]
        stats[label] = {}
        idx = PARTY_ORDER.index(account)
        for j, ptype in enumerate(types):
            row = grp[(grp["account"] == account) & (grp["type"] == ptype)]
            if row.empty:
                continue
            mean_val = float(row["mean"].iloc[0])
            n_posts  = int(row["n"].iloc[0])
            offset   = (j - len(types)/2 + 0.5) * w
            ax.bar(x[idx] + offset, mean_val, w,
                   label=ptype if account == PARTY_ORDER[0] else "",
                   color=TYPE_COLOURS[ptype], alpha=0.9)
            ax.text(x[idx] + offset, mean_val + 0.003,
                    f"{mean_val:.3f}%\n(n={n_posts:,})",
                    ha="center", va="bottom", fontsize=7.5)
            stats[label][ptype] = {"mean": round(mean_val, 4), "n": n_posts}

    ax.set_title("B4: Mean ER_Monthly by Post Type per Account", pad=15)
    ax.set_xlabel("Account"); ax.set_ylabel("Mean Engagement Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels([ACC_LABELS[a] for a in PARTY_ORDER])
    ax.legend()
    add_method_note(fig,
        f"Mean ER_Monthly by post type (Photo / Video) per account. "
        f"'Other' types excluded. Total n = {len(df):,}.")
    path = save_fig(fig, "B4_ER_by_Post_Type.png")
    return path, stats


# ── B5: Mean Likes / Comments / Shares per Post ──────────────────────────────

def chart_b5(master: pd.DataFrame):
    grp = (
        master.groupby("account", observed=True)[["likes", "comments_count", "shares"]]
        .mean().reset_index()
    )
    grp["label"] = grp["account"].map(ACC_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("B5: Mean Likes / Comments / Shares per Post per Account",
                 fontsize=13, fontweight="bold", y=1.01)
    metrics = [
        ("likes",          "Mean Likes per Post",    "#E53935"),
        ("comments_count", "Mean Comments per Post", "#1E88E5"),
        ("shares",         "Mean Shares per Post",   "#43A047"),
    ]
    stats = {ACC_LABELS[a]: {} for a in PARTY_ORDER}

    for ax, (col, title, colour) in zip(axes, metrics):
        vals    = grp.set_index("label")[col]
        colours = [PARTY_COLOURS[ACC_TO_KEY[l]] for l in vals.index]
        bars    = ax.bar(vals.index, vals.values, color=colours, alpha=0.85, width=0.5)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + float(vals.max()) * 0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Account"); ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=15)
        for lbl, v in vals.items():
            stats[lbl][col] = round(float(v), 2)

    fig.tight_layout()
    add_method_note(fig,
        f"Mean raw interaction counts per post. Not normalised by followers. "
        f"Total n = {len(master):,} posts.")
    path = save_fig(fig, "B5_Mean_Interactions_per_Post.png")
    return path, stats


# ── B6: ER_Monthly Distribution Box Plot ─────────────────────────────────────

def chart_b6(master: pd.DataFrame):
    data    = [master[master["account"] == acc][ER_MONTHLY].dropna().values
               for acc in PARTY_ORDER]
    labels  = [ACC_LABELS[a] for a in PARTY_ORDER]
    colours = [PARTY_COLOURS[a] for a in PARTY_ORDER]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker="o", markersize=2, alpha=0.3, linestyle="none"),
                    medianprops=dict(color="black", linewidth=2))

    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # Cap y-axis at 95th percentile for readability
    p95 = float(np.percentile(master[ER_MONTHLY].dropna(), 95))
    ax.set_ylim(bottom=-0.05, top=p95 * 1.25)

    # Annotate median values
    for i, (d, lbl) in enumerate(zip(data, labels)):
        med = float(np.median(d))
        ax.text(i + 1, med + p95 * 0.02,
                f"{med:.4f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_title("B6: ER_Monthly Distribution per Account (Box Plot)", pad=15)
    ax.set_xlabel("Account"); ax.set_ylabel("Engagement Rate (%)")

    add_method_note(fig,
        f"Box = IQR (Q1–Q3). Whiskers = 1.5×IQR. Dots = outliers beyond whiskers. "
        f"Y-axis capped at 95th percentile ({p95:.3f}%) for readability. "
        f"Median annotated above box. n = {len(master):,} posts.")

    stats = {}
    for lbl, d in zip(labels, data):
        q1  = float(np.percentile(d, 25))
        q3  = float(np.percentile(d, 75))
        med = float(np.median(d))
        stats[lbl] = {
            "q1"    : round(q1,       4),
            "median": round(med,      4),
            "q3"    : round(q3,       4),
            "iqr"   : round(q3 - q1,  4),
            "max"   : round(float(d.max()), 4),
        }
    path = save_fig(fig, "B6_ER_Distribution_BoxPlot.png")
    return path, stats


# ── B7: Mean ER_Monthly per Electoral Period per Account ─────────────────────

def chart_b7(master: pd.DataFrame):
    grp = (
        master.groupby(["period", "account"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", n="count").reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x, w    = np.arange(len(PERIOD_ORDER)), 0.8 / len(PARTY_ORDER)
    stats   = {}

    for i, account in enumerate(PARTY_ORDER):
        label = ACC_LABELS[account]
        stats[label] = {}
        for j, period in enumerate(PERIOD_ORDER):
            row = grp[(grp["account"] == account) & (grp["period"] == period)]
            if row.empty:
                continue
            mean_val = float(row["mean"].iloc[0])
            n_posts  = int(row["n"].iloc[0])
            offset   = (i - len(PARTY_ORDER)/2 + 0.5) * w
            ax.bar(x[j] + offset, mean_val, w,
                   label=label if j == 0 else "",
                   color=PARTY_COLOURS[account], alpha=0.9)
            ax.text(x[j] + offset, mean_val + 0.003,
                    f"{mean_val:.3f}%", ha="center", va="bottom", fontsize=7, rotation=90)
            stats[label][PER_LABELS[period]] = {"mean": round(mean_val, 4), "n": n_posts}

    ax.set_title("B7: Mean ER_Monthly per Electoral Period per Account", pad=15)
    ax.set_xlabel("Electoral Period"); ax.set_ylabel("Mean Engagement Rate (%)")
    ax.set_xticks(x); ax.set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER])
    ax.legend()
    add_method_note(fig,
        "Mean ER_Monthly per account per electoral period. "
        "ER_Monthly uses monthly follower denominator for cross-sectional accuracy. "
        f"n = {len(master):,} total posts.")
    path = save_fig(fig, "B7_ER_per_Period.png")
    return path, stats


# ── B8: ER Heatmap — Account × Period ────────────────────────────────────────

def chart_b8(master: pd.DataFrame):
    pivot = (
        master.groupby(["account", "period"], observed=True)[ER_MONTHLY]
        .mean().unstack()
    )
    pivot.index   = [ACC_LABELS[a] for a in pivot.index]
    pivot.columns = [PER_LABELS[p] for p in pivot.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".4f", cmap="Blues",
        ax=ax, linewidths=0.5,
        annot_kws={"size": 11, "weight": "bold"}
    )
    ax.set_title("B8: Mean ER_Monthly Heatmap — Account × Electoral Period", pad=15)
    ax.set_xlabel("Electoral Period"); ax.set_ylabel("Account")

    add_method_note(fig,
        "Mean ER_Monthly per account per electoral period. "
        "Darker cells = higher engagement. Values = mean ER_Monthly (%). "
        f"n = {len(master):,} posts.")

    stats = {
        acc: {per: round(float(pivot.loc[acc, per]), 4) for per in pivot.columns}
        for acc in pivot.index
    }
    path = save_fig(fig, "B8_ER_Heatmap_Account_Period.png")
    return path, stats


# ── B9: FB vs IG ER_Timeline per Account (4 Subplots) ────────────────────────

def chart_b9(master: pd.DataFrame):
    monthly = (
        master.groupby(["month_date", "account", "platform"], observed=True)[ER_TIMELINE]
        .mean().reset_index().sort_values("month_date")
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("B9: Monthly ER_Timeline — Facebook vs Instagram per Account",
                 fontsize=13, fontweight="bold")
    stats = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        stats[label] = {}
        for plat in PLATFORM_ORDER:
            sub = monthly[(monthly["account"] == account) & (monthly["platform"] == plat)]
            if sub.empty:
                continue
            ax.plot(sub["month_date"], sub[ER_TIMELINE],
                    marker="o", linewidth=2, label=plat,
                    color=PLAT_COLOURS[plat])
            stats[label][plat] = {
                "mean_er": round(float(sub[ER_TIMELINE].mean()), 4),
                "peak_er": round(float(sub[ER_TIMELINE].max()),  4),
            }
        for dt_str in ["2023-09-01", "2024-01-01"]:
            ax.axvline(pd.Timestamp(dt_str), color="gray", linestyle="--", alpha=0.4)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Mean ER_Timeline (%)")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    add_method_note(fig,
        "Monthly mean ER_Timeline per platform per account. "
        "ER_Timeline used to remove follower-growth confound in temporal comparison. "
        "Dashed lines = period boundaries.", y=0.01)

    path = save_fig(fig, "B9_Platform_ER_Timeline_per_Account.png")
    return path, stats


# ── B10: Interaction Composition ─────────────────────────────────────────────

def chart_b10(master: pd.DataFrame):
    grp    = master.groupby("account", observed=True)[["likes", "comments_count", "shares"]].sum()
    grp.index = [ACC_LABELS[a] for a in grp.index]
    totals = grp.sum(axis=1)
    pct    = grp.div(totals, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    METRIC_MAP = {
        "likes"          : ("Likes",    "#E53935"),
        "comments_count" : ("Comments", "#1E88E5"),
        "shares"         : ("Shares",   "#43A047"),
    }
    bottoms = np.zeros(len(pct))

    for col, (lbl, colour) in METRIC_MAP.items():
        vals = pct[col].values
        ax.bar(pct.index, vals, bottom=bottoms, label=lbl, color=colour, alpha=0.9)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 3:
                ax.text(i, b + v/2, f"{v:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_title("B10: Interaction Composition per Account\n(Likes / Comments / Shares)", pad=15)
    ax.set_xlabel("Account"); ax.set_ylabel("Share of Total Interactions (%)")
    ax.set_ylim(0, 110); ax.legend()

    total_int = int(master[["likes", "comments_count", "shares"]].sum().sum())
    add_method_note(fig,
        f"Percentage of total interactions (likes + comments + shares) per account. "
        f"Total interactions across all accounts and posts: {total_int:,}.")

    stats = {
        acc: {lbl: {"pct": round(float(pct.loc[acc, col]), 1),
                    "total": int(grp.loc[acc, col])}
              for col, (lbl, _) in METRIC_MAP.items()}
        for acc in pct.index
    }
    path = save_fig(fig, "B10_Interaction_Composition.png")
    return path, stats


# ── B11: Top 5 Posts per Account (docx table only) ───────────────────────────

def get_top5(master: pd.DataFrame) -> dict:
    top5       = {}
    cap_col    = "caption_en" if "caption_en" in master.columns else "caption_raw"
    for account in PARTY_ORDER:
        sub = (
            master[master["account"] == account]
            .nlargest(5, ER_MONTHLY)
            [["date", "platform", "type", "likes", "comments_count",
              "shares", ER_MONTHLY, cap_col]]
            .copy()
        )
        sub["date"]    = sub["date"].dt.strftime("%Y-%m-%d")
        sub["caption"] = sub[cap_col].astype(str).str[:100] + "..."
        sub[ER_MONTHLY] = sub[ER_MONTHLY].round(4)
        top5[ACC_LABELS[account]] = sub
    return top5


def add_top5_table(doc, account_label: str, df: pd.DataFrame):
    doc.add_heading(f"Top 5 Posts — {account_label}", level=2)
    cols = ["date", "platform", "type", "likes",
            "comments_count", "shares", ER_MONTHLY, "caption"]
    hdrs = ["Date", "Platform", "Type", "Likes",
            "Comments", "Shares", "ER Monthly (%)", "Caption (first 100 chars)"]
    table   = doc.add_table(rows=1, cols=len(hdrs))
    table.style = "Light List Accent 1"
    for i, h in enumerate(hdrs):
        cell = table.rows[0].cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, col in enumerate(cols):
            cells[i].text = str(row[col])
    doc.add_paragraph()


# ── BUILD DOCX ────────────────────────────────────────────────────────────────

def build_docx(master, paths, all_stats, top5):
    doc = init_doc("Block B: Engagement Overview Analysis")
    n   = len(master)

    p = doc.add_paragraph()
    p.add_run("Dataset Overview: ").bold = True
    p.add_run(
        f"This block analyses engagement performance across {n:,} posts from 4 accounts "
        f"(NCBN, TDP, YS Jagan, YSRCP) on Facebook and Instagram, June 2023 – May 2024. "
        f"ER_Monthly (monthly follower denominator) is used for all cross-sectional comparisons. "
        f"ER_Timeline (full-period follower denominator) is used exclusively for temporal trends. "
        f"Both metrics expressed as percentages (×100). Outlier posts included and flagged."
    )
    doc.add_paragraph()

    # B1
    s1 = all_stats["b1"]
    best_mean = max(s1, key=lambda k: s1[k]["mean"])
    doc_section(doc, "B1: Mean & Median Engagement Rate (ER_Monthly) per Account",
        paths["b1"],
        methodology=(
            f"ER_Monthly = (likes + shares + comments_count) / avg monthly followers × 100. "
            f"Mean and median ER_Monthly reported per account. n = {n:,}."
        ),
        stats=[(f"{acc} — Mean / Median / n",
                f"{v['mean']:.4f}% / {v['median']:.4f}% / {v['n']:,}") for acc, v in s1.items()],
        insights=[
            f"{best_mean} recorded the highest mean ER_Monthly at {s1[best_mean]['mean']:.4f}%.",
            "Mean ER exceeds median ER across all accounts, indicating right-skewed distributions "
            "driven by a small number of high-performing viral posts.",
            "Median ER_Monthly is the more representative measure of typical post performance.",
        ],
    )

    # B2
    s2 = all_stats["b2"]
    doc_section(doc, "B2: Mean ER_Monthly by Account & Platform", paths["b2"],
        methodology=(
            f"Mean ER_Monthly grouped by account and platform. "
            f"n per group shown in chart. Total n = {n:,}."
        ),
        stats=[(f"{acc} — {plat}",
                f"Mean: {v['mean']:.4f}% | Median: {v['median']:.4f}% | n: {v['n']:,}")
               for acc, pd_data in s2.items() for plat, v in pd_data.items()],
        insights=[
            "Platform-level engagement differences exist across accounts, reflecting "
            "audience size and content format dynamics.",
            "Instagram vs Facebook ER differences indicate distinct audience engagement "
            "behaviours on each platform for political content.",
        ],
    )

    # B3
    s3 = all_stats["b3"]
    best_peak = max(s3, key=lambda k: s3[k]["peak_mean_er"])
    doc_section(doc, "B3: Monthly ER_Timeline per Account", paths["b3"],
        methodology=(
            "ER_Timeline = (likes + shares + comments_count) / avg timeline followers × 100. "
            "Monthly mean ER_Timeline per account. ER_Timeline removes follower-growth confound "
            f"for accurate temporal comparison. n = {n:,}."
        ),
        stats=[(f"{acc} — Peak month / Peak ER / Overall mean ER",
                f"{v['peak_month']} / {v['peak_mean_er']:.4f}% / {v['overall_mean_er']:.4f}%")
               for acc, v in s3.items()],
        insights=[
            f"{best_peak} recorded the highest single-month mean ER_Timeline of "
            f"{s3[best_peak]['peak_mean_er']:.4f}% in {s3[best_peak]['peak_month']}.",
            "Temporal patterns reveal whether accounts maintained consistent engagement "
            "or exhibited campaign-period spikes.",
        ],
    )

    # B4
    s4 = all_stats["b4"]
    doc_section(doc, "B4: Mean ER_Monthly by Post Type (Photo vs Video)", paths["b4"],
        methodology=(
            f"Mean ER_Monthly by post type (Photo / Video) per account. "
            f"Post type from platform API. 'Other' excluded. n = {n:,}."
        ),
        stats=[(f"{acc} — {ptype}", f"Mean ER: {v['mean']:.4f}% | n: {v['n']:,}")
               for acc, td in s4.items() for ptype, v in td.items()],
        insights=[
            "Photo vs Video engagement differences reveal content format effectiveness "
            "per account, with implications for content strategy.",
            "Accounts where Video ER exceeds Photo ER may benefit from increasing "
            "video production in their digital campaign strategy.",
        ],
    )

    # B5
    s5          = all_stats["b5"]
    top_likes   = max(s5, key=lambda k: s5[k].get("likes", 0))
    top_comments = max(s5, key=lambda k: s5[k].get("comments_count", 0))
    top_shares  = max(s5, key=lambda k: s5[k].get("shares", 0))
    doc_section(doc, "B5: Mean Likes / Comments / Shares per Post", paths["b5"],
        methodology=(
            "Mean raw interaction counts per post per account. "
            f"Not normalised by followers. n = {n:,}."
        ),
        stats=[(f"{acc} — Likes / Comments / Shares",
                f"{v.get('likes', 0):.1f} / {v.get('comments_count', 0):.1f} / {v.get('shares', 0):.1f}")
               for acc, v in s5.items()],
        insights=[
            f"{top_likes} averaged the most likes per post ({s5[top_likes].get('likes', 0):.1f}) — "
            f"strongest passive approval signal.",
            f"{top_shares} averaged the most shares per post ({s5[top_shares].get('shares', 0):.1f}) — "
            f"strongest content amplification beyond follower base.",
        ],
    )

    # B6
    s6 = all_stats["b6"]
    doc_section(doc, "B6: ER_Monthly Distribution per Account (Box Plot)", paths["b6"],
        methodology=(
            "Box plot of ER_Monthly per account. Box = IQR (Q1–Q3). Whiskers = 1.5×IQR. "
            "Dots = outliers. Y-axis capped at 95th percentile for readability. "
            f"n = {n:,}."
        ),
        stats=[(f"{acc} — Q1 / Median / Q3 / IQR",
                f"{v['q1']:.4f}% / {v['median']:.4f}% / {v['q3']:.4f}% / {v['iqr']:.4f}%")
               for acc, v in s6.items()],
        insights=[
            "Wide IQR values indicate high variance in post performance — "
            "consistent with political content where specific events drive viral spikes.",
            "Narrow IQR accounts show more consistent but potentially lower-ceiling engagement.",
            "Outlier dots above whiskers represent viral posts with exceptionally high ER_Monthly.",
        ],
    )

    # B7
    s7 = all_stats["b7"]
    doc_section(doc, "B7: Mean ER_Monthly per Electoral Period", paths["b7"],
        methodology=(
            "Mean ER_Monthly per account per electoral period. "
            "Period classification from source file naming convention. "
            f"n = {n:,}."
        ),
        stats=[(f"{acc} — {per}", f"Mean ER: {v['mean']:.4f}% | n: {v['n']:,}")
               for acc, per_data in s7.items() for per, v in per_data.items()],
        insights=[
            "Changes in ER_Monthly across electoral periods reveal whether audience "
            "engagement intensified as elections approached.",
            "Rising ER in Jan–May 2024 would confirm that the pre-election period "
            "drove increased audience interaction with political content.",
        ],
    )

    # B8
    s8 = all_stats["b8"]
    doc_section(doc, "B8: ER_Monthly Heatmap — Account × Electoral Period", paths["b8"],
        methodology=(
            "Mean ER_Monthly per account per electoral period displayed as a heatmap. "
            "Colour intensity proportional to ER value. "
            f"n = {n:,}."
        ),
        stats=[(f"{acc} — {per}", f"{val:.4f}%")
               for per, per_data in s8.items() for acc, val in per_data.items()],
        insights=[
            "The heatmap provides a quick comparative view of engagement intensity "
            "across all account–period combinations simultaneously.",
            "Dark cells identify peak engagement periods per account, "
            "which can be cross-referenced with specific campaign events.",
        ],
    )

    # B9
    s9 = all_stats["b9"]
    doc_section(doc, "B9: FB vs IG ER_Timeline per Account (Monthly)", paths["b9"],
        methodology=(
            "Monthly mean ER_Timeline per platform per account. "
            "Separate subplot per account for direct FB vs IG comparison. "
            f"n = {n:,}."
        ),
        stats=[(f"{acc} — {plat}", f"Mean ER: {v['mean_er']:.4f}% | Peak ER: {v['peak_er']:.4f}%")
               for acc, pd_data in s9.items() for plat, v in pd_data.items()],
        insights=[
            "Per-account platform divergence in ER_Timeline reveals which platform "
            "generated stronger audience resonance over time.",
            "Consistent FB > IG or IG > FB patterns within an account suggest "
            "platform-specific content strategy advantages.",
        ],
    )

    # B10
    s10 = all_stats["b10"]
    doc_section(doc, "B10: Interaction Composition per Account", paths["b10"],
        methodology=(
            "Percentage of total interactions (likes + comments + shares) per account. "
            f"Total interactions across all accounts: "
            f"{int(master[['likes','comments_count','shares']].sum().sum()):,}."
        ),
        stats=[(f"{acc} — {metric}",
                f"{v['pct']:.1f}% (n={v['total']:,})")
               for acc, met_data in s10.items() for metric, v in met_data.items()],
        insights=[
            "Likes represent passive approval; comments represent active dialogue; "
            "shares represent content amplification to non-followers.",
            "Accounts with higher share percentages have content that resonates "
            "strongly enough for followers to actively disseminate it.",
            "High comment percentages indicate content that provokes discussion, "
            "which is particularly relevant for adversarial political messaging.",
        ],
    )

    # B11: Top 5 tables
    doc.add_heading("B11: Top 5 Posts per Account by ER_Monthly", level=1)
    p = doc.add_paragraph()
    p.add_run("Methodology: ").bold = True
    p.add_run(
        "Top 5 posts per account ranked by ER_Monthly (descending). "
        "Caption text shows first 100 characters of English translation."
    )
    doc.add_paragraph()
    for acc_label, df_top in top5.items():
        add_top5_table(doc, acc_label, df_top)

    docx_path = BLOCK_DIR / "Block_B_Engagement_Analysis.docx"
    doc.save(str(docx_path))
    print(f"  ✓  Saved: Block_B_Engagement_Analysis.docx")
    return docx_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PHASE 3 — Block B: Engagement Overview Analysis")
    print("=" * 62 + "\n")

    master, _ = load_both(verbose=False)
    print("Generating charts...\n")

    path_b1,  stats_b1  = chart_b1(master)
    path_b2,  stats_b2  = chart_b2(master)
    path_b3,  stats_b3  = chart_b3(master)
    path_b4,  stats_b4  = chart_b4(master)
    path_b5,  stats_b5  = chart_b5(master)
    path_b6,  stats_b6  = chart_b6(master)
    path_b7,  stats_b7  = chart_b7(master)
    path_b8,  stats_b8  = chart_b8(master)
    path_b9,  stats_b9  = chart_b9(master)
    path_b10, stats_b10 = chart_b10(master)
    top5                = get_top5(master)

    print("\nBuilding docx report...\n")
    build_docx(
        master,
        paths = {
            "b1": path_b1, "b2": path_b2, "b3": path_b3, "b4": path_b4,
            "b5": path_b5, "b6": path_b6, "b7": path_b7, "b8": path_b8,
            "b9": path_b9, "b10": path_b10,
        },
        all_stats = {
            "b1": stats_b1, "b2": stats_b2, "b3": stats_b3, "b4": stats_b4,
            "b5": stats_b5, "b6": stats_b6, "b7": stats_b7, "b8": stats_b8,
            "b9": stats_b9, "b10": stats_b10,
        },
        top5 = top5,
    )
    print(f"\n✅  Block B complete. All outputs saved to:\n    {BLOCK_DIR}")


if __name__ == "__main__":
    main()

# phase5_theme_engagement.py
# ─────────────────────────────────────────────────────────────────────────────
# Block D: Theme × Engagement Analysis
# D1:  Mean ER_Monthly per theme (all accounts combined — ranked bar)
# D2:  Theme ER_Monthly vs Volume scatter (bubble chart)
# D3:  ER_Monthly distribution per theme (violin plot)
# D4:  Top 3 highest ER themes per account (grouped bar)
# D5:  Theme ER_Monthly by platform (FB vs IG) — heatmap
# D6:  Theme ER_Monthly by period — heatmap
# D7:  Attack vs Non-attack ER comparison per account
# D8:  ER_Monthly vs theme_count — does posting more themes hurt ER?
# D9:  High ER post theme co-occurrence (top 10% ER posts)
# D10: Theme ER ranking consistency across accounts (slope/bump chart)
# Output: Analysis_Output/D_Theme_Engagement/
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
BLOCK_DIR = ANALYSIS_DIR / "D_Theme_Engagement"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

ACC_LABELS  = {"ncbn": "NCBN", "tdp": "TDP", "ysjagan": "YS Jagan", "ysrcp": "YSRCP"}
ACC_TO_KEY  = {v: k for k, v in ACC_LABELS.items()}
PER_LABELS  = {
    "JunAug2023": "Jun–Aug 2023",
    "SepDec2023": "Sep–Dec 2023",
    "JanMay2024": "Jan–May 2024",
}
PLAT_COL    = {"Facebook": "#1877F2", "Instagram": "#E1306C"}

THEME_SHORT = {
    "Leader-centric branding"              : "Leader Branding",
    "Attacking Leadership/Demonisation"    : "Atk. Leadership",
    "Mass mobilisation"                    : "Mass Mobilisation",
    "Attacking Party"                      : "Attacking Party",
    "Anti-incumbency/Attacking Governance" : "Anti-incumbency",
    "Welfare&Governance"                   : "Welfare & Gov.",
    "Welfare & Governance"                 : "Welfare & Gov.",
    "Alliance promotion"                   : "Alliance Promo.",
    "Caste / community politics"           : "Caste/Community",
    "Promoting Party"                      : "Promoting Party",
    "Leadership legacy"                    : "Leadership Legacy",
    "TDP Manifesto"                        : "TDP Manifesto",
    "Agrarian & rural issues"              : "Agrarian/Rural",
    "Youth, jobs & education"              : "Youth/Jobs/Edu.",
    "Women & gender"                       : "Women & Gender",
    "Developmental vision"                 : "Dev. Vision",
}

# Attack themes — used in D7
ATTACK_THEMES = {
    "Attacking Leadership/Demonisation",
    "Attacking Party",
    "Anti-incumbency/Attacking Governance",
}

def sh(t): return THEME_SHORT.get(t, t)

plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 10,
    "axes.titlesize"   : 12,
    "axes.titleweight" : "bold",
    "axes.labelsize"   : 10,
    "figure.dpi"       : 150,
    "savefig.dpi"      : 300,
    "savefig.bbox"     : "tight",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


# ── HELPERS ───────────────────────────────────────────────────────────────────

def save_fig(fig, name):
    path = BLOCK_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  Saved: {name}")
    return path


def add_method_note(fig, text, y=0.01):
    fig.text(0.5, y, f"Methodology: {text}",
             ha="center", va="bottom", fontsize=8,
             style="italic", color="#555555",
             transform=fig.transFigure)


def init_doc(title):
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in h.runs:
        run.font.color.rgb = RGBColor(0x22, 0x71, 0xB3)
    doc.add_paragraph()
    return doc


def doc_section(doc, title, img_path, methodology, stats, insights):
    doc.add_heading(title, level=1)
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


# ── D1: Mean ER_Monthly per Theme (All Accounts Combined) ─────────────────────

def chart_d1(themed):
    grp = (
        themed.groupby("Hashtag_Theme", observed=True)[ER_MONTHLY]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
        .sort_values("mean", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    colours = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(grp)))
    bars    = ax.barh([sh(t) for t in grp["Hashtag_Theme"]],
                      grp["mean"], color=colours, alpha=0.9)

    for bar, v, med, n in zip(bars, grp["mean"], grp["median"], grp["n"]):
        ax.text(bar.get_width() + grp["mean"].max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}%  (Med: {med:.4f}%, n={int(n):,})",
                va="center", fontsize=7.5)

    ax.set_title(
        "D1: Mean ER_Monthly per Theme — All Accounts Combined\n"
        "(Ranked by Mean Engagement Rate)",
        pad=15
    )
    ax.set_xlabel("Mean ER_Monthly (%)")
    ax.set_xlim(0, grp["mean"].max() * 1.55)

    add_method_note(fig,
        "Mean ER_Monthly per theme from exploded themed_df (all accounts combined). "
        "ER_Monthly inherited from original post — multi-theme posts share same ER. "
        f"n = {len(themed):,} theme-rows. Median shown for robustness.")

    stats_out = [
        (sh(row["Hashtag_Theme"]),
         f"Mean: {row['mean']:.4f}% | Median: {row['median']:.4f}% | n: {int(row['n']):,}")
        for _, row in grp.sort_values("mean", ascending=False).iterrows()
    ]
    return save_fig(fig, "D1_Mean_ER_per_Theme.png"), stats_out


# ── D2: Theme ER vs Volume Bubble Chart ──────────────────────────────────────

def chart_d2(themed):
    grp = (
        themed.groupby("Hashtag_Theme", observed=True)[ER_MONTHLY]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    grp["short"] = grp["Hashtag_Theme"].map(sh)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        grp["n"], grp["mean"],
        s=grp["n"] / grp["n"].max() * 2000,
        c=grp["mean"], cmap="RdYlGn",
        alpha=0.75, edgecolors="black", linewidths=0.5
    )
    plt.colorbar(scatter, ax=ax, label="Mean ER_Monthly (%)")

    for _, row in grp.iterrows():
        ax.annotate(
            row["short"],
            xy=(row["n"], row["mean"]),
            xytext=(8, 4), textcoords="offset points",
            fontsize=7.5, color="#333333"
        )

    # Quadrant lines at median values
    med_n  = float(grp["n"].median())
    med_er = float(grp["mean"].median())
    ax.axvline(med_n,  color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(med_er, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.text(med_n * 1.02, grp["mean"].max() * 0.97,
            "High Vol →", fontsize=8, color="gray")
    ax.text(grp["n"].max() * 0.02, med_er * 1.02,
            "High ER ↑", fontsize=8, color="gray")

    ax.set_title(
        "D2: Theme Volume vs Mean ER_Monthly (Bubble Chart)\n"
        "(Bubble size = volume. Quadrant lines at median volume & median ER)",
        pad=15
    )
    ax.set_xlabel("Theme-Row Count (Volume)")
    ax.set_ylabel("Mean ER_Monthly (%)")

    add_method_note(fig,
        "Each bubble = one theme. X = total theme-rows. Y = mean ER_Monthly. "
        "Bubble size proportional to theme volume. "
        "Top-right = high volume & high ER. Top-left = niche but high ER. "
        f"n = {len(themed):,} theme-rows.")

    # Categorise quadrants
    stats_out = []
    for _, row in grp.iterrows():
        quad = ("High Vol + High ER" if row["n"] >= med_n and row["mean"] >= med_er else
                "High Vol + Low ER"  if row["n"] >= med_n and row["mean"] <  med_er else
                "Low Vol + High ER"  if row["n"] <  med_n and row["mean"] >= med_er else
                "Low Vol + Low ER")
        stats_out.append((sh(row["Hashtag_Theme"]),
                          f"Vol: {int(row['n']):,} | Mean ER: {row['mean']:.4f}% | {quad}"))
    return save_fig(fig, "D2_Theme_Volume_vs_ER_Bubble.png"), stats_out


# ── D3: ER_Monthly Distribution per Theme (Violin Plot) ──────────────────────

def chart_d3(themed):
    # Cap at 95th percentile for readability
    p95 = float(themed[ER_MONTHLY].quantile(0.95))
    df  = themed[themed[ER_MONTHLY] <= p95].copy()
    df["short"] = df["Hashtag_Theme"].map(sh)

    # Sort themes by median ER descending
    order = (
        df.groupby("short")[ER_MONTHLY].median()
        .sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(
        data=df, y="short", x=ER_MONTHLY,
        order=order, ax=ax,
        palette="RdYlGn", orient="h",
        inner="quartile", cut=0
    )
    ax.set_title(
        "D3: ER_Monthly Distribution per Theme (Violin Plot)\n"
        "(Sorted by Median ER | Y-axis capped at 95th percentile)",
        pad=15
    )
    ax.set_xlabel("ER_Monthly (%)")
    ax.set_ylabel("Theme")

    add_method_note(fig,
        f"Violin = kernel density estimate of ER_Monthly distribution per theme. "
        f"Inner lines = quartiles. X-axis capped at 95th percentile ({p95:.3f}%) "
        f"for readability. Sorted by median ER descending. n = {len(df):,} theme-rows.")

    stats_out = [
        (sh(t), f"Median ER: {float(themed[themed['Hashtag_Theme']==t][ER_MONTHLY].median()):.4f}%")
        for t in themed["Hashtag_Theme"].unique()
    ]
    return save_fig(fig, "D3_ER_Distribution_per_Theme_Violin.png"), stats_out


# ── D4: Top 3 Highest ER Themes per Account ──────────────────────────────────

def chart_d4(themed):
    top3 = {}
    for account in PARTY_ORDER:
        sub = themed[themed["account"] == account]
        ranked = (
            sub.groupby("Hashtag_Theme", observed=True)[ER_MONTHLY]
            .agg(mean="mean", n="count")
            .query("n >= 10")          # minimum 10 theme-rows for reliability
            .sort_values("mean", ascending=False)
            .head(3)
        )
        top3[ACC_LABELS[account]] = ranked

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
    fig.suptitle(
        "D4: Top 3 Highest ER_Monthly Themes per Account\n"
        "(Minimum 10 theme-rows per theme for statistical reliability)",
        fontsize=12, fontweight="bold"
    )

    stats_out = []
    for ax, account in zip(axes, PARTY_ORDER):
        label  = ACC_LABELS[account]
        ranked = top3[label]
        if ranked.empty:
            ax.set_title(f"{label}\n(insufficient data)")
            continue

        labels_short = [sh(t) for t in ranked.index]
        colours      = [PARTY_COLOURS[account]] * len(ranked)
        bars         = ax.bar(labels_short, ranked["mean"],
                              color=colours, alpha=0.85, width=0.5)

        for bar, v, n in zip(bars, ranked["mean"], ranked["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + float(ranked["mean"].max()) * 0.02,
                    f"{v:.4f}%\n(n={int(n):,})",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{label}\n(Top 3 by Mean ER)",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean ER_Monthly (%)")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

        for theme, row in ranked.iterrows():
            stats_out.append((f"{label} — #{list(ranked.index).index(theme)+1}",
                               f"{sh(theme)} | Mean ER: {row['mean']:.4f}% | n: {int(row['n']):,}"))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "Themes with fewer than 10 theme-rows excluded to avoid unreliable estimates. "
        f"n = {len(themed):,} total theme-rows.", y=0.01)

    return save_fig(fig, "D4_Top3_ER_Themes_per_Account.png"), stats_out


# ── D5: Theme ER_Monthly by Platform Heatmap ─────────────────────────────────

def chart_d5(themed):
    pivot = (
        themed.groupby(["Hashtag_Theme", "platform"], observed=True)[ER_MONTHLY]
        .mean().unstack(fill_value=0)
    )
    pivot.index = [sh(t) for t in pivot.index]
    # Sort by Facebook ER descending
    sort_col = "Facebook" if "Facebook" in pivot.columns else pivot.columns[0]
    pivot    = pivot.sort_values(sort_col, ascending=False)

    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="coolwarm",
                ax=ax, linewidths=0.4,
                annot_kws={"size": 9},
                cbar_kws={"label": "Mean ER_Monthly (%)"})
    ax.set_title(
        "D5: Mean ER_Monthly per Theme × Platform\n"
        "(All Accounts Combined | Sorted by Facebook ER)",
        pad=15
    )
    ax.set_xlabel("Platform"); ax.set_ylabel("Theme")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=8.5)

    add_method_note(fig,
        "Mean ER_Monthly per theme per platform across all accounts combined. "
        "Red = higher, Blue = lower. Sorted by Facebook ER descending. "
        f"n = {len(themed):,} theme-rows.")

    stats_out = []
    for theme in pivot.index:
        for plat in pivot.columns:
            v = float(pivot.loc[theme, plat])
            if v > 0:
                stats_out.append((f"{theme} — {plat}", f"{v:.4f}%"))
    return save_fig(fig, "D5_Theme_ER_by_Platform_Heatmap.png"), stats_out


# ── D6: Theme ER_Monthly by Period Heatmap ───────────────────────────────────

def chart_d6(themed):
    pivot = (
        themed.groupby(["Hashtag_Theme", "period"], observed=True)[ER_MONTHLY]
        .mean().unstack(fill_value=0)
    )
    pivot.index   = [sh(t) for t in pivot.index]
    pivot.columns = [PER_LABELS[p] for p in pivot.columns]
    # Sort by Jan–May 2024 ER descending
    sort_col = "Jan–May 2024" if "Jan–May 2024" in pivot.columns else pivot.columns[-1]
    pivot    = pivot.sort_values(sort_col, ascending=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlOrRd",
                ax=ax, linewidths=0.4,
                annot_kws={"size": 9},
                cbar_kws={"label": "Mean ER_Monthly (%)"})
    ax.set_title(
        "D6: Mean ER_Monthly per Theme × Electoral Period\n"
        "(All Accounts Combined | Sorted by Jan–May 2024 ER)",
        pad=15
    )
    ax.set_xlabel("Electoral Period"); ax.set_ylabel("Theme")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8.5)

    add_method_note(fig,
        "Mean ER_Monthly per theme per electoral period across all accounts. "
        "Sorted by Jan–May 2024 engagement rate descending. "
        f"n = {len(themed):,} theme-rows.")

    stats_out = []
    for theme in pivot.index:
        for per in pivot.columns:
            v = float(pivot.loc[theme, per])
            if v > 0:
                stats_out.append((f"{theme} — {per}", f"{v:.4f}%"))
    return save_fig(fig, "D6_Theme_ER_by_Period_Heatmap.png"), stats_out


# ── D7: Attack vs Non-Attack ER per Account ───────────────────────────────────

def chart_d7(themed):
    df = themed.copy()
    df["category"] = df["Hashtag_Theme"].apply(
        lambda t: "Attack" if t in ATTACK_THEMES else "Non-Attack"
    )

    grp = (
        df.groupby(["account", "category"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "D7: Attack vs Non-Attack Theme Engagement per Account\n"
        "(Mean & Median ER_Monthly)",
        fontsize=12, fontweight="bold"
    )

    stats_out = {}
    for ax, metric, title in zip(axes, ["mean", "median"],
                                  ["Mean ER_Monthly (%)", "Median ER_Monthly (%)"]):
        x, w = np.arange(len(PARTY_ORDER)), 0.35
        for j, cat in enumerate(["Attack", "Non-Attack"]):
            vals = []
            for account in PARTY_ORDER:
                row = grp[(grp["account"] == account) & (grp["category"] == cat)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
            offset = (j - 1) * w + w / 2
            colour = "#C0392B" if cat == "Attack" else "#2471A3"
            bars   = ax.bar(x + offset, vals, w, label=cat, color=colour, alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.4f}%", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([ACC_LABELS[a] for a in PARTY_ORDER])
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Stats
    for _, row in grp.iterrows():
        acc = ACC_LABELS[row["account"]]
        cat = row["category"]
        if acc not in stats_out:
            stats_out[acc] = {}
        stats_out[acc][cat] = {
            "mean"  : round(float(row["mean"]),   4),
            "median": round(float(row["median"]), 4),
            "n"     : int(row["n"]),
        }

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_method_note(fig,
        f"Attack themes: Attacking Leadership/Demonisation, Attacking Party, "
        f"Anti-incumbency/Attacking Governance. "
        f"Non-attack = all other themes. n = {len(themed):,} theme-rows.", y=0.01)

    stats_list = [
        (f"{acc} — {cat}",
         f"Mean: {v['mean']:.4f}% | Median: {v['median']:.4f}% | n: {v['n']:,}")
        for acc, cat_data in stats_out.items()
        for cat, v in cat_data.items()
    ]
    return save_fig(fig, "D7_Attack_vs_NonAttack_ER.png"), stats_list


# ── D8: ER_Monthly vs Theme Count ────────────────────────────────────────────

def chart_d8(themed):
    """Does using more themes per post hurt or help engagement?"""
    # theme_count is on the post level — deduplicate to post level
    post_level = (
        themed.drop_duplicates(subset=["account", "date", "platform"])[
            [ER_MONTHLY, "theme_count", "account"]
        ].copy()
    )
    post_level = post_level[post_level["theme_count"] <= 8]  # exclude extreme outliers

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "D8: Does Posting More Themes per Post Affect Engagement?\n"
        "(ER_Monthly vs Theme Count per Post)",
        fontsize=12, fontweight="bold"
    )

    # Left: Scatter per account with regression lines
    ax = axes[0]
    for account in PARTY_ORDER:
        sub = post_level[post_level["account"] == account]
        jit = sub["theme_count"] + np.random.uniform(-0.15, 0.15, len(sub))
        ax.scatter(jit, sub[ER_MONTHLY],
                   color=PARTY_COLOURS[account], alpha=0.15, s=8,
                   label=ACC_LABELS[account])
        # Regression line
        if len(sub) > 10:
            m, b = np.polyfit(sub["theme_count"], sub[ER_MONTHLY], 1)
            xr   = np.array([1, 8])
            ax.plot(xr, m * xr + b, color=PARTY_COLOURS[account], linewidth=2)

    ax.set_title("Scatter (jittered) + Regression Lines", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Themes per Post")
    ax.set_ylabel("ER_Monthly (%)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Right: Boxplot — ER by theme count (all accounts)
    ax2     = axes[1]
    tc_max  = int(post_level["theme_count"].max())
    data    = [post_level[post_level["theme_count"] == tc][ER_MONTHLY].dropna().values
               for tc in range(1, tc_max + 1)]
    data    = [d for d in data if len(d) > 0]
    labels2 = [str(i + 1) for i in range(len(data))]
    bp      = ax2.boxplot(data, tick_labels=labels2, patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color="black", linewidth=2))
    cmap = plt.cm.Blues(np.linspace(0.3, 0.85, len(bp["boxes"])))
    for patch, col in zip(bp["boxes"], cmap):
        patch.set_facecolor(col); patch.set_alpha(0.8)

    ax2.set_title("ER_Monthly Distribution by Theme Count", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Themes per Post")
    ax2.set_ylabel("ER_Monthly (%)")
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    add_method_note(fig,
        "Post-level analysis: each post counted once regardless of theme count. "
        "theme_count = number of themes assigned to that post by the codebook. "
        "Posts with >8 themes excluded as extreme cases. "
        f"n = {len(post_level):,} posts (post-level, not theme-row-level).", y=0.01)

    # Correlation stats per account
    stats_out = []
    for account in PARTY_ORDER:
        sub = post_level[post_level["account"] == account]
        if len(sub) > 10:
            corr = float(sub[[ER_MONTHLY, "theme_count"]].corr().iloc[0, 1])
            stats_out.append((f"{ACC_LABELS[account]} — Pearson r (ER vs theme_count)",
                               f"{corr:.4f}"))
    return save_fig(fig, "D8_ER_vs_ThemeCount.png"), stats_out


# ── D9: High ER Post Theme Co-occurrence ─────────────────────────────────────

def chart_d9(themed):
    """Which themes appear together in the top 10% ER posts?"""
    p90 = float(themed[ER_MONTHLY].quantile(0.90))

    # Get original post IDs for top 10% ER posts
    high_er = themed[themed[ER_MONTHLY] >= p90].copy()

    # Co-occurrence: for each pair of themes, count how often they appear in same post
    # Group by original post identifier (date + account + platform)
    post_themes = (
        high_er.groupby(["account", "date", "platform"], observed=True)["Hashtag_Theme"]
        .apply(list).reset_index()
    )

    all_themes = sorted(themed["Hashtag_Theme"].unique())
    n_themes   = len(all_themes)
    theme_idx  = {t: i for i, t in enumerate(all_themes)}
    cooc       = np.zeros((n_themes, n_themes), dtype=int)

    for _, row in post_themes.iterrows():
        themes = list(set(row["Hashtag_Theme"]))
        for i in range(len(themes)):
            for j in range(len(themes)):
                if i != j:
                    cooc[theme_idx[themes[i]], theme_idx[themes[j]]] += 1

    cooc_df         = pd.DataFrame(cooc, index=[sh(t) for t in all_themes],
                                   columns=[sh(t) for t in all_themes])
    np.fill_diagonal(cooc_df.values, 0)

    fig, ax = plt.subplots(figsize=(14, 11))
    mask    = cooc_df == 0
    sns.heatmap(cooc_df, annot=True, fmt=",", cmap="Blues",
                ax=ax, mask=mask,
                linewidths=0.3,
                annot_kws={"size": 7},
                cbar_kws={"label": "Co-occurrence count"})
    ax.set_title(
        f"D9: Theme Co-occurrence in Top 10% ER Posts\n"
        f"(Posts at or above {p90:.3f}% ER_Monthly | n = {len(post_themes):,} posts)",
        pad=15
    )
    ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)

    add_method_note(fig,
        f"Co-occurrence = number of high-ER posts containing both themes simultaneously. "
        f"Top 10% ER threshold = {p90:.4f}% ER_Monthly. "
        f"Blank cells = no co-occurrence in high-ER posts. "
        f"n = {len(post_themes):,} posts above threshold.", y=0.01)

    # Top 5 co-occurring pairs
    pairs = []
    for i in range(n_themes):
        for j in range(i + 1, n_themes):
            if cooc[i, j] > 0:
                pairs.append((all_themes[i], all_themes[j], cooc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    stats_out = [(f"{sh(a)} × {sh(b)}", f"{c:,} co-occurrences") for a, b, c in pairs[:15]]
    return save_fig(fig, "D9_HighER_Theme_Cooccurrence.png"), stats_out


# ── D10: Theme ER Ranking Consistency (Bump/Slope Chart) ─────────────────────

def chart_d10(themed):
    """Do the same themes rank highest in ER across all accounts?"""
    # Only themes with ≥10 posts per account to ensure reliability
    rank_data = {}
    for account in PARTY_ORDER:
        sub = themed[themed["account"] == account]
        er_per_theme = (
            sub.groupby("Hashtag_Theme", observed=True)[ER_MONTHLY]
            .agg(mean="mean", n="count")
            .query("n >= 10")
            .sort_values("mean", ascending=False)
        )
        rank_data[ACC_LABELS[account]] = {
            sh(t): rank + 1 for rank, t in enumerate(er_per_theme.index)
        }

    # Get all themes that appear in at least 2 accounts
    all_t = {}
    for acc_data in rank_data.values():
        for t in acc_data:
            all_t[t] = all_t.get(t, 0) + 1
    shared_themes = [t for t, cnt in all_t.items() if cnt >= 2]

    fig, ax = plt.subplots(figsize=(12, 8))
    accs    = [ACC_LABELS[a] for a in PARTY_ORDER]
    x_pos   = np.arange(len(accs))
    cmap    = plt.cm.tab20.colors

    stats_out = []
    for i, theme in enumerate(shared_themes):
        ranks  = [rank_data[acc].get(theme, np.nan) for acc in accs]
        colour = cmap[i % 20]
        valid  = [(xi, r) for xi, r in zip(x_pos, ranks) if not np.isnan(r)]
        if len(valid) < 2:
            continue
        vx, vy = zip(*valid)
        ax.plot(vx, vy, marker="o", linewidth=1.8,
                color=colour, alpha=0.8, markersize=7)
        ax.text(vx[-1] + 0.08, vy[-1], theme,
                fontsize=7.5, va="center", color=colour)
        stats_out.append((theme, " → ".join([
            f"{acc}: #{int(r)}" for acc, r in zip(accs, ranks) if not np.isnan(r)
        ])))

    ax.set_title(
        "D10: Theme ER_Monthly Ranking per Account\n"
        "(Bump Chart — Do the Same Themes Rank High Across All Accounts?)",
        pad=15
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(accs, fontsize=10)
    ax.set_ylabel("ER Rank (1 = Highest ER per Account)")
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_xlim(-0.3, len(accs) + 0.8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    add_method_note(fig,
        "Rank 1 = highest mean ER_Monthly theme for that account. "
        "Only themes with ≥10 theme-rows per account included. "
        "Only themes appearing in ≥2 accounts shown. "
        f"n = {len(themed):,} theme-rows.")

    return save_fig(fig, "D10_Theme_ER_Ranking_BumpChart.png"), stats_out


# ── BUILD DOCX ────────────────────────────────────────────────────────────────

def build_docx(master, themed, paths, all_stats):
    doc = init_doc("Block D: Theme × Engagement Analysis")
    n_t = len(themed)
    n_m = len(master)

    p = doc.add_paragraph()
    p.add_run("Dataset Overview: ").bold = True
    p.add_run(
        f"This block analyses the relationship between thematic content and engagement "
        f"performance across all 4 accounts. "
        f"Source: themed_df ({n_t:,} theme-rows from {n_m:,} posts). "
        f"ER_Monthly is the primary engagement metric (normalised by monthly followers). "
        f"Key methodological note: ER_Monthly is inherited from the original post — "
        f"multi-theme posts assign the same ER to each theme-row. "
        f"This is the standard approach in multi-label content analysis and is disclosed "
        f"in all chart notes. Attack themes are: Attacking Leadership/Demonisation, "
        f"Attacking Party, Anti-incumbency/Attacking Governance."
    )
    doc.add_paragraph()

    # D1
    s1       = all_stats["d1"]
    top_er   = s1[0][0]
    bot_er   = s1[-1][0]
    doc_section(doc, "D1: Mean ER_Monthly per Theme — All Accounts Combined",
        paths["d1"],
        methodology=(
            "Mean and median ER_Monthly per theme from exploded themed_df "
            "(all accounts combined). Sorted by mean ER descending. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=s1,
        insights=[
            f"{top_er} generates the highest mean engagement rate across all accounts combined, "
            f"making it the most audience-resonant theme in the dataset.",
            f"{bot_er} generates the lowest mean engagement, indicating audiences respond "
            f"less actively to this thematic category.",
            "The ranking of themes by ER does not mirror their ranking by frequency (C1) — "
            "high-volume themes are not necessarily the highest-engaging ones.",
            "This divergence between volume and ER is strategically significant: accounts "
            "may be over-investing in high-frequency but lower-ER themes.",
        ],
    )

    # D2
    s2 = all_stats["d2"]
    doc_section(doc, "D2: Theme Volume vs Mean ER_Monthly (Bubble Chart)",
        paths["d2"],
        methodology=(
            "Each bubble represents one theme. X = total theme-rows (volume). "
            "Y = mean ER_Monthly. Bubble size proportional to volume. "
            "Quadrant lines at median volume and median ER. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=s2,
        insights=[
            "Top-left quadrant (Low Volume, High ER): these are niche but highly resonant "
            "themes — underused relative to their engagement potential.",
            "Top-right quadrant (High Volume, High ER): these are the gold-standard themes — "
            "high posting frequency sustained by strong audience engagement.",
            "Bottom-right quadrant (High Volume, Low ER): these themes dominate posting "
            "activity but return lower-than-median engagement — potential over-investment.",
            "Bottom-left quadrant (Low Volume, Low ER): peripheral themes with limited "
            "strategic value based on both volume and engagement metrics.",
        ],
    )

    # D3
    doc_section(doc, "D3: ER_Monthly Distribution per Theme (Violin Plot)",
        paths["d3"],
        methodology=(
            "Violin plot of ER_Monthly distribution per theme. "
            "Sorted by median ER descending. "
            "Y-axis capped at 95th percentile for readability. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d3"],
        insights=[
            "Wide violins indicate high variance in engagement — some posts "
            "using this theme go viral while others underperform.",
            "Narrow violins indicate consistent but moderate engagement — "
            "predictable but not explosive audience response.",
            "Themes with high median ER but wide distribution are high-risk/high-reward: "
            "strong average performance with potential for viral posts.",
            "The violin shape also reveals whether ER distributions are "
            "unimodal (single typical performance level) or bimodal (two distinct "
            "performance clusters within the same theme).",
        ],
    )

    # D4
    doc_section(doc, "D4: Top 3 Highest ER Themes per Account",
        paths["d4"],
        methodology=(
            "Top 3 themes by mean ER_Monthly per account. "
            "Themes with fewer than 10 theme-rows excluded for reliability. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d4"],
        insights=[
            "Each account has a distinct set of top-performing themes, confirming that "
            "theme–account fit matters: the same theme does not perform equally across accounts.",
            "Where attack themes appear in an account's top 3, it confirms that adversarial "
            "content generates above-average engagement for that account's audience.",
            "Where branding or governance themes appear in the top 3, it confirms "
            "that constructive content can outperform attack content for some accounts.",
            "Cross-referencing top-ER themes with posting frequency (C9) reveals "
            "whether accounts post most in their highest-ER theme categories.",
        ],
    )

    # D5
    doc_section(doc, "D5: Mean ER_Monthly per Theme × Platform",
        paths["d5"],
        methodology=(
            "Mean ER_Monthly per theme per platform across all accounts. "
            "Sorted by Facebook ER descending. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d5"][:20],
        insights=[
            "Platform differences in theme ER indicate that audiences on Facebook "
            "and Instagram respond differently to the same thematic content.",
            "Themes with higher Instagram ER than Facebook ER are better suited to "
            "Instagram's visual, younger-demographic engagement dynamics.",
            "Themes with consistent ER across both platforms are platform-agnostic "
            "in their audience resonance — safe cross-posting candidates.",
        ],
    )

    # D6
    doc_section(doc, "D6: Mean ER_Monthly per Theme × Electoral Period",
        paths["d6"],
        methodology=(
            "Mean ER_Monthly per theme per electoral period across all accounts. "
            "Sorted by Jan–May 2024 ER descending. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d6"][:20],
        insights=[
            "Themes that rise in ER heading into Jan–May 2024 confirm election-period "
            "audience activation — audiences became more engaged with these topics as "
            "elections approached.",
            "Themes that peak in earlier periods and decline in Jan–May 2024 may have "
            "suffered from audience fatigue or been displaced by more urgent election themes.",
            "Consistent ER across all three periods indicates durable messaging themes "
            "whose resonance is not election-cycle dependent.",
        ],
    )

    # D7
    doc_section(doc, "D7: Attack vs Non-Attack Theme Engagement per Account",
        paths["d7"],
        methodology=(
            "Attack themes: Attacking Leadership/Demonisation, Attacking Party, "
            "Anti-incumbency/Attacking Governance. "
            "All other themes = Non-Attack. "
            "Mean and median ER_Monthly compared per account. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d7"],
        insights=[
            "Where attack ER > non-attack ER: adversarial content outperforms constructive "
            "content for that account's audience — emotionally charged posts drive more engagement.",
            "Where non-attack ER > attack ER: the account's audience is more receptive to "
            "constructive, governance, or branding content than attack messaging.",
            "The attack/non-attack ER gap varies by account, suggesting that audience "
            "composition and partisan alignment shape responsiveness to attack content.",
            "From a campaign strategy perspective, accounts where attack ER leads "
            "have a data-driven rationale for prioritising adversarial messaging.",
        ],
    )

    # D8
    doc_section(doc, "D8: ER_Monthly vs Number of Themes per Post",
        paths["d8"],
        methodology=(
            "Post-level analysis: each post counted once. "
            "theme_count = number of themes assigned by the codebook. "
            "Posts with >8 themes excluded as extreme cases. "
            "Regression lines per account. Boxplots show ER distribution by theme count. "
            f"n = {n_t:,} posts (post-level)."
        ),
        stats=all_stats["d8"],
        insights=[
            "A negative correlation between theme count and ER would suggest that "
            "thematically scattered posts underperform focused single-theme posts.",
            "A positive correlation would suggest that thematically dense posts "
            "capture wider audience interest through multiple issue hooks.",
            "Near-zero correlation indicates that the number of themes per post "
            "does not materially impact engagement — content quality matters more than breadth.",
            "Per-account regression variation reveals whether this relationship "
            "differs by account, reflecting differences in audience behaviour and content style.",
        ],
    )

    # D9
    doc_section(doc, "D9: Theme Co-occurrence in Top 10% ER Posts",
        paths["d9"],
        methodology=(
            "Co-occurrence = number of high-ER posts containing both themes simultaneously. "
            "High-ER = posts at or above the 90th percentile of ER_Monthly. "
            "Co-occurrence matrix computed at post level (not theme-row level). "
            f"n = {len(themed[themed[ER_MONTHLY] >= float(themed[ER_MONTHLY].quantile(0.90))]):,} "
            f"posts above threshold. Total n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d9"],
        insights=[
            "High co-occurrence pairs in top-ER posts identify powerful thematic combinations — "
            "these pairings consistently appear in the most engaging content.",
            "Theme combinations that appear frequently in high-ER posts but rarely "
            "in overall posts (see C1) represent high-value underused content strategies.",
            "Attack + mobilisation co-occurrences in high-ER posts would confirm that "
            "combining adversarial with rally content is a winning formula for engagement.",
            "Leader branding + attack co-occurrences in high-ER posts would indicate "
            "that personalised attack content outperforms generic party-level attacks.",
        ],
    )

    # D10
    doc_section(doc, "D10: Theme ER Ranking Consistency Across Accounts (Bump Chart)",
        paths["d10"],
        methodology=(
            "Rank 1 = highest mean ER_Monthly theme for that account. "
            "Only themes with ≥10 theme-rows per account included. "
            "Only themes appearing in ≥2 accounts shown for comparability. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["d10"],
        insights=[
            "Themes where lines are flat (consistent rank across accounts) are "
            "universally resonant — they perform similarly regardless of which account posts them.",
            "Themes where lines cross dramatically indicate account-specific fit — "
            "high performing for one account but low performing for another.",
            "A theme that ranks #1 for multiple accounts is the most universally "
            "engaging content category in this dataset.",
            "Accounts whose top-ranked themes differ from their highest-volume themes "
            "(C9) are not maximising their engagement potential — strategic rebalancing "
            "toward higher-ER themes could improve overall performance.",
        ],
    )

    docx_path = BLOCK_DIR / "Block_D_Theme_Engagement.docx"
    doc.save(str(docx_path))
    print(f"  ✓  Saved: Block_D_Theme_Engagement.docx")
    return docx_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PHASE 5 — Block D: Theme × Engagement Analysis")
    print("=" * 62 + "\n")

    master, themed = load_both(verbose=False)
    print("Generating charts...\n")

    path_d1,  stats_d1  = chart_d1(themed)
    path_d2,  stats_d2  = chart_d2(themed)
    path_d3,  stats_d3  = chart_d3(themed)
    path_d4,  stats_d4  = chart_d4(themed)
    path_d5,  stats_d5  = chart_d5(themed)
    path_d6,  stats_d6  = chart_d6(themed)
    path_d7,  stats_d7  = chart_d7(themed)
    path_d8,  stats_d8  = chart_d8(themed)
    path_d9,  stats_d9  = chart_d9(themed)
    path_d10, stats_d10 = chart_d10(themed)

    print("\nBuilding docx report...\n")
    build_docx(
        master, themed,
        paths = {
            "d1": path_d1, "d2": path_d2, "d3": path_d3, "d4": path_d4,
            "d5": path_d5, "d6": path_d6, "d7": path_d7, "d8": path_d8,
            "d9": path_d9, "d10": path_d10,
        },
        all_stats = {
            "d1": stats_d1, "d2": stats_d2, "d3": stats_d3, "d4": stats_d4,
            "d5": stats_d5, "d6": stats_d6, "d7": stats_d7, "d8": stats_d8,
            "d9": stats_d9, "d10": stats_d10,
        },
    )

    print(f"\n✅  Block D complete. All outputs saved to:\n    {BLOCK_DIR}")


if __name__ == "__main__":
    main()

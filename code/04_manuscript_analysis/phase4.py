# phase4_thematic_dist.py
# ─────────────────────────────────────────────────────────────────────────────
# Block C: Thematic Distribution Analysis
# C1: Overall theme frequency (all accounts combined)
# C2: Account × Theme heatmap (absolute counts)
# C3: Thematic composition per account (100% stacked horizontal)
# C4: Cross-party theme ownership per theme
# C5: Top 8 themes per account (grouped bar)
# C6: Thematic shifts across electoral periods (4 subplots)
# C7: Platform theme comparison — FB vs IG (4 subplots)
# C8: Unclassified posts per account
# Output: Analysis_Output/C_Thematic_Distribution/
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    OUTPUT_DIR as ANALYSIS_DIR
)

# ── SETUP ─────────────────────────────────────────────────────────────────────
BLOCK_DIR = ANALYSIS_DIR / "C_Thematic_Distribution"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

ACC_LABELS = {"ncbn": "NCBN", "tdp": "TDP", "ysjagan": "YS Jagan", "ysrcp": "YSRCP"}
ACC_TO_KEY = {v: k for k, v in ACC_LABELS.items()}
PER_LABELS = {
    "JunAug2023": "Jun–Aug 2023",
    "SepDec2023": "Sep–Dec 2023",
    "JanMay2024": "Jan–May 2024",
}

# Short labels for chart axes (15 themes → compact names)
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


def sh(theme: str) -> str:
    """Return short label for a theme."""
    return THEME_SHORT.get(theme, theme)


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


# ── C1: Overall Theme Frequency ───────────────────────────────────────────────

def chart_c1(themed: pd.DataFrame, theme_colours: dict):
    counts  = themed["Hashtag_Theme"].value_counts().sort_values()  # ascending for barh

    fig, ax = plt.subplots(figsize=(10, 8))
    colours = [theme_colours.get(t, "#999999") for t in counts.index]
    bars    = ax.barh([sh(t) for t in counts.index], counts.values,
                      color=colours, alpha=0.85)

    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_width() + counts.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=9, fontweight="bold")

    ax.set_title("C1: Overall Theme Frequency Across All Accounts", pad=15)
    ax.set_xlabel("Number of Theme-Rows")
    ax.set_xlim(0, counts.max() * 1.15)

    n = len(themed)
    add_method_note(fig,
        f"Theme-row counts from exploded themed_df across all accounts. "
        f"One post with N themes contributes N theme-rows. n = {n:,} theme-rows.")

    stats_out = [(sh(t), f"{c:,}") for t, c in counts.sort_values(ascending=False).items()]
    return save_fig(fig, "C1_Overall_Theme_Frequency.png"), stats_out


# ── C2: Account × Theme Heatmap ──────────────────────────────────────────────

def chart_c2(themed: pd.DataFrame):
    pivot = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)
        .size().unstack(fill_value=0)
    )
    pivot.index   = [ACC_LABELS[a] for a in pivot.index]
    pivot.columns = [sh(t) for t in pivot.columns]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, annot=True, fmt=",", cmap="YlOrRd",
                ax=ax, linewidths=0.3, annot_kws={"size": 8})
    ax.set_title("C2: Theme-Row Count Heatmap — Account × Theme", pad=15)
    ax.set_xlabel("Theme"); ax.set_ylabel("Account")
    ax.tick_params(axis="x", rotation=40, labelsize=8)

    n = len(themed)
    add_method_note(fig,
        f"Raw theme-row counts per account per theme. "
        f"Darker = higher frequency. n = {n:,} theme-rows.")

    stats_out = [
        (f"{acc} — {theme}", f"{int(pivot.loc[acc, theme]):,}")
        for acc in pivot.index for theme in pivot.columns
        if int(pivot.loc[acc, theme]) > 0
    ]
    return save_fig(fig, "C2_Account_Theme_Heatmap.png"), stats_out


# ── C3: Thematic Composition per Account (100% Stacked Horizontal) ───────────

def chart_c3(themed: pd.DataFrame, theme_colours: dict, all_themes: list):
    pivot = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)
        .size().unstack(fill_value=0)
    )
    pct           = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.index     = [ACC_LABELS[a] for a in pct.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    lefts   = np.zeros(len(pct))

    for theme in all_themes:
        if theme not in pct.columns:
            continue
        vals = pct[theme].values
        ax.barh(pct.index, vals, left=lefts,
                label=sh(theme), color=theme_colours.get(theme, "#999999"), alpha=0.9)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 7:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
        lefts += vals

    ax.set_title("C3: Thematic Composition per Account (% of Theme-Rows)", pad=15)
    ax.set_xlabel("% of Theme-Rows"); ax.set_xlim(0, 113)
    ax.legend(loc="upper right", fontsize=7, ncol=2,
              bbox_to_anchor=(1.38, 1.0))

    n = len(themed)
    add_method_note(fig,
        f"% = theme-rows(account, theme) / total theme-rows(account) × 100. n = {n:,}.")

    top_theme_per_acc = {acc: pct.loc[acc].idxmax() for acc in pct.index}
    stats_out = [
        (f"{acc} — Dominant theme",
         f"{sh(t)} ({pct.loc[acc, t]:.1f}%)")
        for acc, t in top_theme_per_acc.items()
    ]
    return save_fig(fig, "C3_Thematic_Composition_per_Account.png"), stats_out


# ── C4: Cross-Party Theme Ownership ──────────────────────────────────────────

def chart_c4(themed: pd.DataFrame):
    pivot = (
        themed.groupby(["Hashtag_Theme", "account"], observed=True)
        .size().unstack(fill_value=0)
    )
    pct         = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.columns = [ACC_LABELS[a] for a in pct.columns]
    # Sort themes ascending by total count for horizontal readability
    pct         = pct.loc[pct.sum(axis=1).sort_values(ascending=True).index]
    pct.index   = [sh(t) for t in pct.index]

    fig, ax = plt.subplots(figsize=(11, 9))
    lefts   = np.zeros(len(pct))

    for account in PARTY_ORDER:
        label = ACC_LABELS[account]
        if label not in pct.columns:
            continue
        vals = pct[label].values
        ax.barh(pct.index, vals, left=lefts,
                label=label, color=PARTY_COLOURS[account], alpha=0.9)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 12:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        lefts += vals

    ax.set_title("C4: Cross-Party Theme Ownership\n"
                 "(% of Each Theme's Posts by Account)", pad=15)
    ax.set_xlabel("% of Theme's Posts"); ax.set_xlim(0, 113)
    ax.legend()

    n = len(themed)
    add_method_note(fig,
        f"% = theme-rows(account, theme) / total theme-rows(theme) × 100. "
        f"Shows which account dominates each theme. n = {n:,} theme-rows.")

    stats_out = [
        (f"{theme} — Dominant account",
         f"{pct.loc[theme].idxmax()} ({pct.loc[theme].max():.1f}%)")
        for theme in pct.index
    ]
    return save_fig(fig, "C4_Cross_Party_Theme_Ownership.png"), stats_out


# ── C5: Top 8 Themes per Account (Grouped Bar) ───────────────────────────────

def chart_c5(themed: pd.DataFrame):
    top8 = themed["Hashtag_Theme"].value_counts().nlargest(8).index.tolist()
    grp  = (
        themed[themed["Hashtag_Theme"].isin(top8)]
        .groupby(["account", "Hashtag_Theme"], observed=True)
        .size().reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    x, w    = np.arange(len(top8)), 0.8 / len(PARTY_ORDER)
    stats_out = {}

    for i, account in enumerate(PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = grp[grp["account"] == account].set_index("Hashtag_Theme")
        vals  = [int(sub.loc[t, "count"]) if t in sub.index else 0 for t in top8]
        stats_out[label] = {sh(t): v for t, v in zip(top8, vals)}
        offset = (i - len(PARTY_ORDER) / 2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=label, color=PARTY_COLOURS[account], alpha=0.9)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 5,
                        f"{v:,}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)

    ax.set_title("C5: Top 8 Themes — Theme-Row Count per Account", pad=15)
    ax.set_xlabel("Theme"); ax.set_ylabel("Number of Theme-Rows")
    ax.set_xticks(x)
    ax.set_xticklabels([sh(t) for t in top8], rotation=20, ha="right")
    ax.legend()

    n = len(themed)
    add_method_note(fig,
        f"Top 8 themes by total cross-account frequency. n = {n:,} theme-rows.")

    stats_list = [(f"{acc} — {theme}", f"{count:,}")
                  for acc, td in stats_out.items()
                  for theme, count in td.items()]
    return save_fig(fig, "C5_Top8_Themes_per_Account.png"), stats_list


# ── C6: Thematic Shifts Across Electoral Periods (4 Subplots) ────────────────

def chart_c6(themed: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("C6: Thematic Composition Shifts Across Electoral Periods per Account",
                 fontsize=13, fontweight="bold")
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = themed[themed["account"] == account]

        pivot = sub.groupby(["period", "Hashtag_Theme"], observed=True).size().unstack(fill_value=0)
        for p in PERIOD_ORDER:
            if p not in pivot.index:
                pivot.loc[p] = 0
        pivot = pivot.reindex(PERIOD_ORDER)

        # Normalise: % of theme-rows per period
        pct           = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pct.index     = [PER_LABELS[p] for p in pct.index]
        pct.columns   = [sh(t) for t in pct.columns]
        pct           = pct.T   # themes as rows, periods as columns

        sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                    ax=ax, linewidths=0.3,
                    annot_kws={"size": 7},
                    cbar_kws={"label": "% of Period Theme-Rows"})
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Electoral Period"); ax.set_ylabel("Theme")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=8)
        stats_out[label] = pct.to_dict()

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_method_note(fig,
        "Theme % per period = theme-rows(account, period, theme) / "
        "total theme-rows(account, period) × 100. "
        "Shows how each account's thematic messaging shifted across periods.", y=0.01)

    # Dominant theme per account per period
    stats_list = []
    for acc, per_data in stats_out.items():
        for period, theme_data in per_data.items():
            top_t = max(theme_data, key=theme_data.get)
            stats_list.append((f"{acc} — {period} — Dominant theme",
                                f"{top_t} ({theme_data[top_t]:.1f}%)"))
    return save_fig(fig, "C6_Theme_Temporal_Shifts.png"), stats_list


# ── C7: Platform Theme Comparison — FB vs IG (4 Subplots) ────────────────────

def chart_c7(themed: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("C7: Thematic Emphasis — Facebook vs Instagram per Account",
                 fontsize=13, fontweight="bold")
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = themed[themed["account"] == account]

        pivot = sub.groupby(["platform", "Hashtag_Theme"], observed=True).size().unstack(fill_value=0)
        pct           = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pct.columns   = [sh(t) for t in pct.columns]
        pct           = pct.T   # themes as rows, platforms as columns

        sns.heatmap(pct, annot=True, fmt=".1f", cmap="Greens",
                    ax=ax, linewidths=0.3,
                    annot_kws={"size": 8},
                    cbar_kws={"label": "% of Platform Theme-Rows"})
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Platform"); ax.set_ylabel("Theme")
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=9)
        stats_out[label] = pct.to_dict()

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_method_note(fig,
        "Theme % per platform = theme-rows(account, platform, theme) / "
        "total theme-rows(account, platform) × 100. "
        "Shows if accounts tailor thematic messaging per platform.", y=0.01)

    stats_list = []
    for acc, plat_data in stats_out.items():
        for plat, theme_data in plat_data.items():
            top_t = max(theme_data, key=theme_data.get)
            stats_list.append((f"{acc} — {plat} — Dominant theme",
                                f"{top_t} ({theme_data[top_t]:.1f}%)"))
    return save_fig(fig, "C7_Platform_Theme_Comparison.png"), stats_list


# ── C8: Unclassified Posts per Account ───────────────────────────────────────

def chart_c8(master: pd.DataFrame):
    grp = (
        master.groupby("account", observed=True)
        .apply(lambda x: pd.Series({
            "unclassified": int(x["Hashtag_Theme"].isna().sum()),
            "total"       : len(x),
        }))
        .reset_index()
    )
    grp["pct"]    = grp["unclassified"] / grp["total"] * 100
    grp["label"]  = grp["account"].map(ACC_LABELS)
    colours       = [PARTY_COLOURS[a] for a in grp["account"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("C8: Posts Without Hashtag Theme Classification per Account",
                 fontsize=13, fontweight="bold", y=1.01)

    # Left: absolute count
    bars = axes[0].bar(grp["label"], grp["unclassified"], color=colours, alpha=0.85, width=0.5)
    for bar, v in zip(bars, grp["unclassified"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{int(v):,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Count", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Account"); axes[0].set_ylabel("Number of Posts")

    # Right: percentage
    bars2 = axes[1].bar(grp["label"], grp["pct"], color=colours, alpha=0.85, width=0.5)
    for bar, v in zip(bars2, grp["pct"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Percentage of Total Posts", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Account"); axes[1].set_ylabel("% of Total Posts")

    fig.tight_layout()
    n = len(master)
    add_method_note(fig,
        f"Unclassified = posts where Hashtag_Theme is blank "
        f"(no hashtag matched the thematic codebook). Total n = {n:,} posts.")

    stats_out = [
        (f"{row['label']} — Unclassified count / %",
         f"{int(row['unclassified']):,} / {row['pct']:.1f}%")
        for _, row in grp.iterrows()
    ]
    return save_fig(fig, "C8_Unclassified_Posts.png"), stats_out


# ── BUILD DOCX ────────────────────────────────────────────────────────────────

def build_docx(master, themed, paths, all_stats):
    doc             = init_doc("Block C: Thematic Distribution Analysis")
    n_master        = len(master)
    n_themed        = len(themed)
    n_with_theme    = int(master["Hashtag_Theme"].notna().sum())
    n_without_theme = int(master["Hashtag_Theme"].isna().sum())

    p = doc.add_paragraph()
    p.add_run("Dataset Overview: ").bold = True
    p.add_run(
        f"This block analyses the thematic distribution of political posts across 4 accounts "
        f"(NCBN, TDP, YS Jagan, YSRCP), June 2023 – May 2024. "
        f"Total posts: {n_master:,}. Posts with at least one theme: {n_with_theme:,}. "
        f"Posts without a theme (excluded from thematic analysis): {n_without_theme:,}. "
        f"After multi-theme explosion, {n_with_theme:,} posts generate {n_themed:,} theme-rows. "
        f"A post with N themes contributes N theme-rows — each theme is attributed independently. "
        f"This methodology follows standard multi-label content analysis practice and is "
        f"disclosed in all chart methodology notes."
    )
    doc.add_paragraph()

    # ── C1 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C1: Overall Theme Frequency Across All Accounts",
        paths["c1"],
        methodology=(
            f"Total theme-row counts per theme across all 4 accounts combined. "
            f"Sorted by frequency. n = {n_themed:,} theme-rows. "
            f"Multi-theme posts contribute one count per theme."
        ),
        stats=all_stats["c1"],
        insights=[
            "Leader-centric branding and Attacking Leadership/Demonisation are the most "
            "frequent themes, confirming the personalised and adversarial character of "
            "Andhra Pradesh political social media during this period.",
            "Attack and mobilisation themes collectively dominate over governance, welfare, "
            "and policy themes — consistent with campaign-phase content strategies that "
            "prioritise voter activation over policy communication.",
            "Developmental vision, Women & gender, and Agrarian/rural are the least "
            "represented themes, suggesting these are niche rather than mainstream "
            "hashtag-driven messaging areas.",
            "The concentration of volume in the top 4 themes indicates that despite 15 "
            "categories, political messaging is highly focused on a few core narratives.",
        ],
    )

    # ── C2 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C2: Account × Theme Heatmap (Absolute Counts)",
        paths["c2"],
        methodology=(
            f"Raw theme-row counts per account per theme. "
            f"Darker cells = higher frequency. n = {n_themed:,} theme-rows."
        ),
        stats=all_stats["c2"][:24],  # cap for docx length
        insights=[
            "The heatmap reveals both absolute volume differences between accounts "
            "and the relative thematic concentration within each account.",
            "Accounts with high counts in attack-category themes signal adversarial "
            "communication strategies; those concentrated in branding themes signal "
            "image-building strategies.",
            "Sparse rows (accounts with low overall volume) vs dense rows (high-volume "
            "accounts) reflect the structural disparity in posting frequency across accounts.",
        ],
    )

    # ── C3 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C3: Thematic Composition per Account (% Stacked)",
        paths["c3"],
        methodology=(
            f"% = theme-rows(account, theme) / total theme-rows(account) × 100. "
            f"Normalised within each account — shows messaging identity independent of volume. "
            f"n = {n_themed:,} theme-rows."
        ),
        stats=all_stats["c3"],
        insights=[
            "The composition chart removes volume bias — it shows what each account "
            "is about, not just how much they post.",
            "Accounts with high attack-theme shares project an adversarial messaging "
            "posture; high branding shares indicate leader image-building focus.",
            "Accounts with broader thematic diversity signal multi-issue communication "
            "strategies compared to those concentrated in 2–3 themes.",
            "Direct comparison of composition across accounts reveals fundamentally "
            "different content strategies despite shared political context.",
        ],
    )

    # ── C4 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C4: Cross-Party Theme Ownership",
        paths["c4"],
        methodology=(
            f"% = theme-rows(account, theme) / total theme-rows(theme) × 100. "
            f"For each theme: which account contributed what share. n = {n_themed:,}."
        ),
        stats=all_stats["c4"],
        insights=[
            "Theme ownership reveals which parties set the agenda for each political topic — "
            "a key indicator of issue dominance in political communication.",
            "Themes dominated by a single account (>60% ownership) represent exclusive "
            "issue territory; themes with spread ownership are competitive thematic battlegrounds.",
            "Party-specific themes (TDP Manifesto, Alliance promotion) are naturally "
            "concentrated in the originating party's accounts.",
            "Attack-theme ownership distribution reveals which parties are targeted "
            "most frequently and which are doing the attacking.",
        ],
    )

    # ── C5 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C5: Top 8 Themes per Account (Absolute Count)",
        paths["c5"],
        methodology=(
            f"Top 8 themes by total cross-account frequency. "
            f"Count per account shown per theme. n = {n_themed:,} theme-rows."
        ),
        stats=all_stats["c5"][:20],
        insights=[
            "The grouped bar chart enables direct absolute comparison of the highest-frequency "
            "themes across accounts in a single view.",
            "Volume disparities in shared themes reflect overall posting volume differences — "
            "cross-reference with C3 (composition %) for normalised comparisons.",
            "Themes where one account dramatically outpaces others indicate theme ownership "
            "by volume, not just by proportion.",
        ],
    )

    # ── C6 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C6: Thematic Composition Shifts Across Electoral Periods",
        paths["c6"],
        methodology=(
            "Theme % per period = theme-rows(account, period, theme) / "
            "total theme-rows(account, period) × 100. "
            "Shows how each account's thematic emphasis shifted from Jun–Aug 2023 "
            "→ Sep–Dec 2023 → Jan–May 2024 (pre-election campaign period)."
        ),
        stats=all_stats["c6"],
        insights=[
            "Increases in attack and mobilisation theme % heading into Jan–May 2024 confirm "
            "a strategic shift from general communication to campaign-phase messaging.",
            "Decreases in governance or welfare theme % over time may indicate a strategic "
            "de-emphasis of policy content as election day approached.",
            "The temporal evolution of thematic composition is a key indicator of "
            "campaign strategy adaptation in response to the electoral calendar.",
            "Accounts that maintain consistent thematic composition across all periods "
            "demonstrate a sustained strategic communication framework, while volatile "
            "compositions reflect reactive or event-driven content strategies.",
        ],
    )

    # ── C7 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C7: Thematic Emphasis — Facebook vs Instagram",
        paths["c7"],
        methodology=(
            "Theme % per platform = theme-rows(account, platform, theme) / "
            "total theme-rows(account, platform) × 100. "
            "Reveals platform-specific thematic strategy per account."
        ),
        stats=all_stats["c7"],
        insights=[
            "Platform-specific thematic differences indicate deliberate audience targeting — "
            "tailoring message themes to demographic profiles of each platform.",
            "Instagram's younger, more visual demographic may explain higher proportions of "
            "leader branding and mobilisation content on that platform.",
            "Facebook's more politically engaged audience may receive higher proportions of "
            "attack and governance content reflecting more text-intensive engagement norms.",
            "Accounts with near-identical FB/IG thematic distributions suggest a "
            "platform-agnostic content strategy — same messages cross-posted.",
        ],
    )

    # ── C8 ────────────────────────────────────────────────────────────────────
    doc_section(doc, "C8: Unclassified Posts per Account",
        paths["c8"],
        methodology=(
            f"Unclassified posts = posts where Hashtag_Theme is blank. "
            f"Reasons: (1) post has no hashtags; (2) hashtags used do not appear in "
            f"the thematic codebook; (3) hashtags are informal or one-off. "
            f"Total posts: {n_master:,}. With theme: {n_with_theme:,}. "
            f"Without theme: {n_without_theme:,}."
        ),
        stats=all_stats["c8"],
        insights=[
            "High unclassified rates indicate accounts that communicate without "
            "systematic hashtag strategies, relying on captions rather than hashtags for reach.",
            "Low unclassified rates indicate disciplined, hashtag-driven content strategies "
            "consistent with coordinated digital campaigns.",
            "Unclassified posts are excluded from all thematic analyses in Blocks C and D. "
            "This exclusion rate is transparently reported here for research integrity.",
        ],
    )

    docx_path = BLOCK_DIR / "Block_C_Thematic_Distribution.docx"
    doc.save(str(docx_path))
    print(f"  ✓  Saved: Block_C_Thematic_Distribution.docx")
    return docx_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PHASE 4 — Block C: Thematic Distribution Analysis")
    print("=" * 62 + "\n")

    master, themed = load_both(verbose=False)

    # Build theme colour map (consistent across all C-block charts)
    all_themes    = themed["Hashtag_Theme"].value_counts().index.tolist()
    theme_colours = {
        t: plt.cm.tab20.colors[i % 20]
        for i, t in enumerate(all_themes)
    }

    print("Generating charts...\n")

    path_c1, stats_c1 = chart_c1(themed, theme_colours)
    path_c2, stats_c2 = chart_c2(themed)
    path_c3, stats_c3 = chart_c3(themed, theme_colours, all_themes)
    path_c4, stats_c4 = chart_c4(themed)
    path_c5, stats_c5 = chart_c5(themed)
    path_c6, stats_c6 = chart_c6(themed)
    path_c7, stats_c7 = chart_c7(themed)
    path_c8, stats_c8 = chart_c8(master)

    print("\nBuilding docx report...\n")
    build_docx(
        master, themed,
        paths = {
            "c1": path_c1, "c2": path_c2, "c3": path_c3, "c4": path_c4,
            "c5": path_c5, "c6": path_c6, "c7": path_c7, "c8": path_c8,
        },
        all_stats = {
            "c1": stats_c1, "c2": stats_c2, "c3": stats_c3, "c4": stats_c4,
            "c5": stats_c5, "c6": stats_c6, "c7": stats_c7, "c8": stats_c8,
        },
    )

    print(f"\n✅  Block C complete. All outputs saved to:\n    {BLOCK_DIR}")


if __name__ == "__main__":
    main()

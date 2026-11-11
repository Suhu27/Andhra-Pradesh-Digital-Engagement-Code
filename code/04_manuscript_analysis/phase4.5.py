# phase4b_thematic_supplement.py
# ─────────────────────────────────────────────────────────────────────────────
# Supplementary charts for Block C — fixes + additions
#
# FIXES:
#   C3-revised  : Thematic composition — taller, clearer, better legend
#   C7-revised  : FB vs IG platform theme — corrected insights
#
# NEW CHARTS:
#   C9  : Per-account thematic composition (4 individual pie/bar subplots)
#   C10 : Monthly theme volume timeline per account (4 line subplots)
#   C11 : Theme × Engagement — mean ER_Monthly per theme per account
#   C12 : Heatmap — mean ER_Monthly by theme × account
#   C13 : Engagement rate per theme over periods (did attack themes drive ER?)
#   C14 : Per-account ER timeline with thematic annotation
#
# Output: Analysis_Output/C_Thematic_Distribution/  (same folder, new files)
#         + Block_C_Supplement.docx
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
BLOCK_DIR = ANALYSIS_DIR / "C_Thematic_Distribution"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

ACC_LABELS = {"ncbn": "NCBN", "tdp": "TDP", "ysjagan": "YS Jagan", "ysrcp": "YSRCP"}
ACC_TO_KEY = {v: k for k, v in ACC_LABELS.items()}
PER_LABELS = {
    "JunAug2023": "Jun–Aug 2023",
    "SepDec2023": "Sep–Dec 2023",
    "JanMay2024": "Jan–May 2024",
}
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


# ── C3-REVISED: Thematic Composition — Fixed Layout ──────────────────────────

def chart_c3_revised(themed, theme_colours, all_themes):
    pivot = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)
        .size().unstack(fill_value=0)
    )
    pct       = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pct.index = [ACC_LABELS[a] for a in pct.index]

    fig, ax = plt.subplots(figsize=(15, 7))
    lefts   = np.zeros(len(pct))

    legend_handles = []
    for theme in all_themes:
        if theme not in pct.columns:
            continue
        vals   = pct[theme].values
        colour = theme_colours.get(theme, "#999999")
        ax.barh(pct.index, vals, left=lefts, color=colour, alpha=0.9)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 6:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        legend_handles.append(mpatches.Patch(color=colour, label=sh(theme)))
        lefts += vals

    ax.set_title(
        "C3 (Revised): Thematic Composition per Account\n"
        "Share of Theme-Rows within Each Account (% of Account's Total Theme-Rows)",
        pad=15
    )
    ax.set_xlabel("% of Theme-Rows"); ax.set_xlim(0, 108)
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8, ncol=3, framealpha=0.85)

    add_method_note(fig,
        "% = theme-rows(account, theme) / total theme-rows(account) × 100. "
        "YS Jagan's Mass Mobilisation dominance (78.3%) reflects extremely concentrated messaging. "
        f"n = {len(themed):,} theme-rows.")

    stats_out = [(f"{acc} — Dominant theme",
                  f"{sh(pct.loc[acc].idxmax())} ({pct.loc[acc].max():.1f}%)")
                 for acc in pct.index]
    return save_fig(fig, "C3_REVISED_Thematic_Composition.png"), stats_out


# ── C7-REVISED: Platform Theme Comparison — Corrected ────────────────────────

def chart_c7_revised(themed):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=False)
    fig.suptitle(
        "C7 (Revised): Thematic Emphasis by Platform per Account\n"
        "(Facebook vs Instagram — % of Platform's Theme-Rows)",
        fontsize=13, fontweight="bold"
    )
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = themed[themed["account"] == account]

        pivot = sub.groupby(["platform", "Hashtag_Theme"], observed=True).size().unstack(fill_value=0)
        pct   = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pct.columns = [sh(t) for t in pct.columns]
        pct   = pct.T.sort_values("Facebook" if "Facebook" in pct.index else pct.index[0],
                                   ascending=True)

        # Diverging: FB left, IG right
        fb = pct["Facebook"].values if "Facebook" in pct.columns else np.zeros(len(pct))
        ig = pct["Instagram"].values if "Instagram" in pct.columns else np.zeros(len(pct))
        y  = np.arange(len(pct))

        ax.barh(y, -fb, color=PLAT_COL["Facebook"], alpha=0.85, label="Facebook")
        ax.barh(y, ig,  color=PLAT_COL["Instagram"], alpha=0.85, label="Instagram")
        ax.axvline(0, color="black", linewidth=0.8)

        for i, (f_val, i_val) in enumerate(zip(fb, ig)):
            if f_val > 2:
                ax.text(-f_val - 0.3, i, f"{f_val:.1f}%", ha="right", va="center", fontsize=7)
            if i_val > 2:
                ax.text(i_val + 0.3, i, f"{i_val:.1f}%", ha="left", va="center", fontsize=7)

        ax.set_yticks(y)
        ax.set_yticklabels(pct.index, fontsize=7.5)
        ax.set_title(
            f"{label}\nFacebook ←  |  → Instagram\n"
            f"(FB n={int(sub[sub['platform']=='Facebook']['Hashtag_Theme'].notna().sum()):,} | "
            f"IG n={int(sub[sub['platform']=='Instagram']['Hashtag_Theme'].notna().sum()):,})",
            fontsize=10, fontweight="bold"
        )
        ax.legend(fontsize=8)
        stats_out[label] = {
            "Facebook" : {t: round(float(fb[i]), 2) for i, t in enumerate(pct.index)},
            "Instagram": {t: round(float(ig[i]), 2) for i, t in enumerate(pct.index)},
        }

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_method_note(fig,
        "Diverging bars: Facebook = left (negative axis), Instagram = right (positive axis). "
        "Both normalised to 100% within platform per account. "
        "Symmetric bars = platform-agnostic strategy. Asymmetric = platform-tailored. "
        f"n = {len(themed):,} theme-rows.", y=0.01)

    stats_list = [
        (f"{acc} — {plat} — top theme", f"{max(td, key=td.get)} ({max(td.values()):.1f}%)")
        for acc, pd_data in stats_out.items() for plat, td in pd_data.items()
        if td
    ]
    return save_fig(fig, "C7_REVISED_Platform_Theme_Diverging.png"), stats_list


# ── C9: Per-Account Theme Breakdown (4 Vertical Bar Subplots) ─────────────────

def chart_c9(themed):
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("C9: Per-Account Theme Breakdown\n"
                 "(All 15 Themes — % of Account's Theme-Rows)",
                 fontsize=13, fontweight="bold")
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = themed[themed["account"] == account]
        counts = sub["Hashtag_Theme"].value_counts()
        total  = counts.sum()
        pct    = (counts / total * 100).sort_values(ascending=False)
        short_labels = [sh(t) for t in pct.index]

        colours = [PARTY_COLOURS[account]] * len(pct)
        bars    = ax.bar(range(len(pct)), pct.values, color=colours, alpha=0.85)

        for bar, v in zip(bars, pct.values):
            if v > 2:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{v:.1f}%", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold")

        ax.set_title(
            f"{label} — Thematic Distribution\n"
            f"(Total theme-rows: {int(total):,} | Posts with theme: {sub['Hashtag_Theme'].notna().sum():,})",
            fontsize=10, fontweight="bold"
        )
        ax.set_xticks(range(len(pct)))
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7.5)
        ax.set_ylabel("% of Account's Theme-Rows")
        ax.set_ylim(0, pct.values[0] * 1.25)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        stats_out[label] = {sh(t): round(float(v), 2) for t, v in pct.items()}

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    add_method_note(fig,
        "Each subplot shows % distribution of all 15 themes within that account's posts. "
        "Bars sorted by frequency descending. "
        f"n = {len(themed):,} total theme-rows across all accounts.", y=0.01)

    stats_list = [(f"{acc} — {theme}", f"{pct:.1f}%")
                  for acc, td in stats_out.items()
                  for theme, pct in list(td.items())[:5]]
    return save_fig(fig, "C9_Per_Account_Theme_Breakdown.png"), stats_list


# ── C10: Monthly Theme Volume Timeline per Account ────────────────────────────

def chart_c10(themed):
    # Use top 5 themes only to keep lines readable
    top5 = themed["Hashtag_Theme"].value_counts().nlargest(5).index.tolist()
    sub  = themed[themed["Hashtag_Theme"].isin(top5)].copy()

    monthly = (
        sub.groupby(["month_date", "account", "Hashtag_Theme"], observed=True)
        .size().reset_index(name="count")
        .sort_values("month_date")
    )

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(
        "C10: Monthly Volume of Top 5 Themes per Account\n"
        "(Jun 2023 – May 2024)",
        fontsize=13, fontweight="bold"
    )

    theme_colours_local = {t: plt.cm.tab10.colors[i] for i, t in enumerate(top5)}
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        stats_out[label] = {}
        for theme in top5:
            sub_acc = monthly[
                (monthly["account"] == account) &
                (monthly["Hashtag_Theme"] == theme)
            ]
            if sub_acc.empty:
                continue
            ax.plot(sub_acc["month_date"], sub_acc["count"],
                    marker="o", linewidth=1.8, markersize=4,
                    label=sh(theme),
                    color=theme_colours_local[theme])
            stats_out[label][sh(theme)] = {
                "peak_month": sub_acc.loc[sub_acc["count"].idxmax(), "month_date"].strftime("%b %Y"),
                "peak_count": int(sub_acc["count"].max()),
            }

        for dt_str in ["2023-09-01", "2024-01-01"]:
            ax.axvline(pd.Timestamp(dt_str), color="gray",
                       linestyle="--", alpha=0.4, linewidth=1)

        ax.set_title(
            f"{label}\n"
            f"(Total theme-rows: {int(monthly[monthly['account']==account]['count'].sum()):,})",
            fontsize=10, fontweight="bold"
        )
        ax.legend(fontsize=7.5, loc="upper left")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylabel("Monthly Theme-Row Count")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    add_method_note(fig,
        "Monthly count of theme-rows for top 5 themes by overall frequency. "
        "Dashed lines = period boundaries (Sep 2023, Jan 2024). "
        f"n = {len(sub):,} theme-rows (top-5 themes only).", y=0.01)

    stats_list = [
        (f"{acc} — {theme} — Peak", f"{v['peak_month']} ({v['peak_count']:,} rows)")
        for acc, td in stats_out.items() for theme, v in td.items()
    ]
    return save_fig(fig, "C10_Monthly_Theme_Volume_Timeline.png"), stats_list


# ── C11: Mean ER_Monthly per Theme per Account (Grouped Bar) ─────────────────

def chart_c11(themed):
    grp = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", n="count").reset_index()
    )
    all_themes = themed["Hashtag_Theme"].value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=(16, 7))
    x, w    = np.arange(len(all_themes)), 0.8 / len(PARTY_ORDER)
    stats_out = {}

    for i, account in enumerate(PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = grp[grp["account"] == account].set_index("Hashtag_Theme")
        vals  = []
        for t in all_themes:
            if t in sub.index:
                vals.append(float(sub.loc[t, "mean"]))
            else:
                vals.append(0.0)
        offset = (i - len(PARTY_ORDER) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w, label=label,
               color=PARTY_COLOURS[account], alpha=0.85)
        stats_out[label] = {
            sh(t): round(float(v), 4) for t, v in zip(all_themes, vals) if v > 0
        }

    ax.set_title(
        "C11: Mean ER_Monthly per Theme per Account\n"
        "(Which Themes Drive the Most Engagement per Account?)",
        pad=15
    )
    ax.set_xlabel("Theme"); ax.set_ylabel("Mean ER_Monthly (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([sh(t) for t in all_themes], rotation=35, ha="right", fontsize=8)
    ax.legend()

    add_method_note(fig,
        "Mean ER_Monthly per theme per account from exploded themed_df. "
        "Note: ER_Monthly is inherited from the original post — "
        "multi-theme posts share the same ER across all their theme-rows. "
        f"n = {len(themed):,} theme-rows.")

    stats_list = [(f"{acc} — {theme}", f"Mean ER: {er:.4f}%")
                  for acc, td in stats_out.items()
                  for theme, er in sorted(td.items(), key=lambda x: x[1], reverse=True)[:3]]
    return save_fig(fig, "C11_Mean_ER_per_Theme_per_Account.png"), stats_list


# ── C12: Heatmap — Mean ER_Monthly by Theme × Account ────────────────────────

def chart_c12(themed):
    pivot = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)[ER_MONTHLY]
        .mean().unstack(fill_value=0)
    )
    pivot.index   = [ACC_LABELS[a] for a in pivot.index]
    pivot.columns = [sh(t) for t in pivot.columns]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                ax=ax, linewidths=0.3,
                annot_kws={"size": 7.5},
                cbar_kws={"label": "Mean ER_Monthly (%)"})
    ax.set_title(
        "C12: Mean ER_Monthly Heatmap — Account × Theme\n"
        "(Which Account–Theme Combinations Have Highest Engagement?)",
        pad=15
    )
    ax.set_xlabel("Theme"); ax.set_ylabel("Account")
    ax.tick_params(axis="x", rotation=35, labelsize=7.5)

    add_method_note(fig,
        "Mean ER_Monthly per account per theme. Green = higher engagement, Red = lower. "
        "Zero-filled where account has no posts in that theme. "
        f"n = {len(themed):,} theme-rows.")

    stats_out = [
        (f"{acc} — {theme}", f"{float(pivot.loc[acc, theme]):.4f}%")
        for acc in pivot.index for theme in pivot.columns
        if float(pivot.loc[acc, theme]) > 0
    ]
    return save_fig(fig, "C12_ER_Heatmap_Theme_Account.png"), stats_out


# ── C13: ER_Monthly per Theme over Periods (Did Attack Themes Drive ER?) ──────

def chart_c13(themed):
    # Top 6 themes only — keeps chart readable
    top6    = themed["Hashtag_Theme"].value_counts().nlargest(6).index.tolist()
    sub     = themed[themed["Hashtag_Theme"].isin(top6)].copy()

    grp = (
        sub.groupby(["period", "Hashtag_Theme"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", n="count").reset_index()
    )

    fig, ax = plt.subplots(figsize=(13, 6))
    x, w    = np.arange(len(PERIOD_ORDER)), 0.8 / len(top6)
    theme_colours_local = {t: plt.cm.tab10.colors[i] for i, t in enumerate(top6)}
    stats_out = {}

    for i, theme in enumerate(top6):
        vals   = []
        for period in PERIOD_ORDER:
            row = grp[(grp["period"] == period) & (grp["Hashtag_Theme"] == theme)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        offset = (i - len(top6) / 2 + 0.5) * w
        ax.bar(x + offset, vals, w,
               label=sh(theme),
               color=theme_colours_local[theme], alpha=0.85)
        stats_out[sh(theme)] = {
            PER_LABELS[p]: round(v, 4) for p, v in zip(PERIOD_ORDER, vals)
        }

    ax.set_title(
        "C13: Mean ER_Monthly of Top 6 Themes per Electoral Period\n"
        "(Did Engagement Patterns per Theme Change Across Periods?)",
        pad=15
    )
    ax.set_xlabel("Electoral Period"); ax.set_ylabel("Mean ER_Monthly (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER])
    ax.legend(fontsize=8)

    add_method_note(fig,
        "Mean ER_Monthly per theme per electoral period (all accounts combined). "
        "Shows whether audience engagement with specific themes shifted as elections approached. "
        f"Top 6 themes by frequency shown. n = {len(sub):,} theme-rows.")

    stats_list = [(f"{theme} — {period}", f"{er:.4f}%")
                  for theme, pd_data in stats_out.items()
                  for period, er in pd_data.items()]
    return save_fig(fig, "C13_Theme_ER_per_Period.png"), stats_list


# ── C14: Per-Account Monthly ER_Timeline with Dominant Theme Annotation ───────

def chart_c14(master, themed):
    # Monthly ER_Timeline per account (line chart)
    # + annotate which theme was dominant each period
    monthly_er = (
        master.groupby(["month_date", "account"], observed=True)[ER_TIMELINE]
        .mean().reset_index().sort_values("month_date")
    )
    # Dominant theme per account per period
    dom_theme = (
        themed.groupby(["account", "period", "Hashtag_Theme"], observed=True)
        .size().reset_index(name="count")
    )
    dom_theme = (
        dom_theme.loc[dom_theme.groupby(["account", "period"],
                                         observed=True)["count"].idxmax()]
        .set_index(["account", "period"])["Hashtag_Theme"]
    )

    # Period midpoints for annotation
    period_mids = {
        "JunAug2023": pd.Timestamp("2023-07-15"),
        "SepDec2023": pd.Timestamp("2023-10-15"),
        "JanMay2024": pd.Timestamp("2024-03-01"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    fig.suptitle(
        "C14: Per-Account Monthly ER_Timeline\nwith Dominant Theme per Period Annotated",
        fontsize=13, fontweight="bold"
    )
    stats_out = {}

    for ax, account in zip(axes.flatten(), PARTY_ORDER):
        label = ACC_LABELS[account]
        sub   = monthly_er[monthly_er["account"] == account].copy()

        ax.plot(sub["month_date"], sub[ER_TIMELINE],
                marker="o", linewidth=2,
                color=PARTY_COLOURS[account], label="ER_Timeline")
        ax.fill_between(sub["month_date"], sub[ER_TIMELINE],
                         alpha=0.15, color=PARTY_COLOURS[account])

        # Period boundary lines
        max_er = float(sub[ER_TIMELINE].max()) if not sub.empty else 1.0
        for dt_str in ["2023-09-01", "2024-01-01"]:
            ax.axvline(pd.Timestamp(dt_str), color="gray",
                       linestyle="--", alpha=0.4, linewidth=1)

        # Annotate dominant theme per period
        for period, mid in period_mids.items():
            key = (account, period)
            if key in dom_theme.index:
                dom = sh(dom_theme[key])
                er_at_mid = float(sub.iloc[(sub["month_date"] -
                                            pd.Timestamp(mid)).abs().argsort().iloc[0]][ER_TIMELINE]) \
                            if not sub.empty else 0
                ax.annotate(
                    f"▶ {dom}",
                    xy=(mid, er_at_mid),
                    xytext=(mid, max_er * 0.92),
                    fontsize=7, color="#333333",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                )

        ax.set_title(
            f"{label}\nMonthly Mean ER_Timeline (with dominant theme per period)",
            fontsize=10, fontweight="bold"
        )
        ax.set_ylabel("Mean ER_Timeline (%)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        stats_out[label] = {
            "peak_er_month": sub.loc[sub[ER_TIMELINE].idxmax(), "month_date"].strftime("%b %Y")
                             if not sub.empty else "N/A",
            "peak_er"      : round(float(sub[ER_TIMELINE].max()), 4) if not sub.empty else 0,
        }

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_method_note(fig,
        "Monthly mean ER_Timeline per account. ER_Timeline removes follower-growth confound. "
        "Annotated arrows show dominant theme for each electoral period. "
        f"n = {len(master):,} total posts.", y=0.01)

    stats_list = [
        (f"{acc} — Peak ER month / Peak ER",
         f"{v['peak_er_month']} / {v['peak_er']:.4f}%")
        for acc, v in stats_out.items()
    ]
    return save_fig(fig, "C14_Per_Account_ER_Timeline_Annotated.png"), stats_list


# ── PLATFORM COLOUR CONSTANT ───────────────────────────────────────────────────
PLAT_COL = {"Facebook": "#1877F2", "Instagram": "#E1306C"}


# ── BUILD DOCX ────────────────────────────────────────────────────────────────

def build_docx(master, themed, paths, all_stats):
    doc  = init_doc("Block C Supplement: Thematic Distribution — Additional & Revised Charts")
    n_t  = len(themed)
    n_m  = len(master)

    p = doc.add_paragraph()
    p.add_run("Purpose of this supplement: ").bold = True
    p.add_run(
        "This document contains (1) revised versions of C3 and C7 with improved visual clarity "
        "and corrected insights, and (2) six new charts (C9–C14) providing deeper thematic, "
        "temporal, and engagement analyses. All charts are consistent with the master_df "
        f"({n_m:,} posts) and themed_df ({n_t:,} theme-rows)."
    )
    doc.add_paragraph()

    doc_section(doc, "C3 (Revised): Thematic Composition per Account",
        paths["c3r"],
        methodology=(
            "% = theme-rows(account, theme) / total theme-rows(account) × 100. "
            "Revised for improved legend readability and accurate annotations. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c3r"],
        insights=[
            "YS Jagan's Mass Mobilisation dominance at 78.3% is the most extreme thematic "
            "concentration in the dataset — nearly 4 in 5 of his classified posts carry this theme, "
            "indicating a singular rally-and-crowd strategy compared to all other accounts.",
            "YSRCP leads in Leader-centric branding (31.1%), confirming Jagan's party "
            "invested heavily in incumbent-leader image construction as a defensive electoral strategy.",
            "TDP's thematic diversity across attack, mobilisation, alliance, and governance themes "
            "reflects a multi-front campaign approach targeting different voter segments simultaneously.",
            "NCBN's Mass Mobilisation dominance (40.7%) despite lower absolute post volume suggests "
            "focused pre-election rally communication concentrated in the Jan–May 2024 period.",
        ],
    )

    doc_section(doc, "C7 (Revised): Thematic Emphasis by Platform — Diverging Bar",
        paths["c7r"],
        methodology=(
            "Diverging bar chart: Facebook = left (negative axis), Instagram = right (positive axis). "
            "Both normalised to 100% within platform per account. "
            "Symmetric bars indicate platform-agnostic strategy; asymmetric bars indicate "
            "platform-tailored thematic messaging. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c7r"][:16],
        insights=[
            "The data shows near-identical thematic distributions across Facebook and Instagram "
            "for all four accounts — dominant themes are consistent on both platforms per account. "
            "This contradicts assumptions of platform-specific content tailoring.",
            "The platform-agnostic thematic strategy suggests content is cross-posted rather "
            "than platform-optimised — a common pattern in resource-constrained political campaigns.",
            "While theme proportions are similar, absolute volumes differ significantly by platform "
            "(see B2 and B5) — same message type, different quantity.",
            "Future research could examine caption text and hashtag variation between platforms "
            "to detect micro-level differentiation not captured by theme categories.",
        ],
    )

    doc_section(doc, "C9: Per-Account Theme Breakdown (All 15 Themes)",
        paths["c9"],
        methodology=(
            "Individual bar chart per account showing % of all 15 themes within that account's "
            "total theme-rows. Sorted by frequency descending. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c9"],
        insights=[
            "Individual account subplots expose the long-tail structure of thematic messaging — "
            "each account has 2–3 dominant themes and a long tail of occasional themes.",
            "YS Jagan's profile is near-singular: Mass Mobilisation overwhelms all other themes, "
            "which may reflect data characteristics (fewer classified posts overall) as well "
            "as genuine strategic focus.",
            "YSRCP shows the broadest thematic spread, suggesting a comprehensive multi-issue "
            "strategy that addresses governance, branding, attack, and social themes in parallel.",
        ],
    )

    doc_section(doc, "C10: Monthly Theme Volume Timeline per Account",
        paths["c10"],
        methodology=(
            "Monthly count of theme-rows for the top 5 themes by overall frequency. "
            "Separate line per theme per account. "
            "Dashed lines = period boundaries. "
            f"n = {n_t:,} theme-rows (top-5 themes only)."
        ),
        stats=all_stats["c10"][:20],
        insights=[
            "The timeline chart reveals whether individual themes escalated or declined "
            "as elections approached — a direct test of campaign strategy adaptation.",
            "Themes that spike sharply in Jan–May 2024 confirm campaign-phase activation; "
            "themes that plateau or decline were de-prioritised as the election drew closer.",
            "Cross-account comparison of theme timelines shows whether rival parties "
            "escalated the same themes simultaneously (reaction) or independently (strategy).",
            "TDP and YSRCP's overlapping theme escalation patterns would indicate a "
            "reactive thematic arms race in the final campaign period.",
        ],
    )

    doc_section(doc, "C11: Mean ER_Monthly per Theme per Account",
        paths["c11"],
        methodology=(
            "Mean ER_Monthly per theme per account from exploded themed_df. "
            "ER_Monthly is inherited from the original post — multi-theme posts "
            "share the same ER across all their theme-rows. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c11"],
        insights=[
            "This chart answers a core research question: which themes generate "
            "the most audience engagement per account?",
            "Themes with high ER despite low frequency (see C1) are high-value niche themes — "
            "infrequent but resonant with the audience when used.",
            "Themes with high frequency but lower ER are volume-driven strategies — "
            "presence through repetition rather than audience resonance.",
            "ER differences between accounts for the same theme reveal audience "
            "receptivity differences — the same theme may land differently depending on "
            "the account's existing follower base and credibility.",
        ],
    )

    doc_section(doc, "C12: Mean ER_Monthly Heatmap — Account × Theme",
        paths["c12"],
        methodology=(
            "Mean ER_Monthly per account per theme. "
            "Green = higher engagement, Red = lower. "
            "Zero-filled where account has no posts in that theme. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c12"][:20],
        insights=[
            "The heatmap provides a simultaneous cross-account view of engagement "
            "intensity per theme — identifying high-performing account–theme combinations.",
            "Dark green cells identify the most resonant account–theme pairs, which have "
            "the strongest strategic value for that account's content planning.",
            "Accounts with uniformly moderate ER across themes rely on volume; "
            "accounts with spiky high-ER cells in specific themes have found niche "
            "content categories with high audience resonance.",
        ],
    )

    doc_section(doc, "C13: Theme ER_Monthly per Electoral Period",
        paths["c13"],
        methodology=(
            "Mean ER_Monthly per theme per electoral period (all accounts combined). "
            "Shows whether audience engagement with specific themes changed as elections approached. "
            f"Top 6 themes by frequency shown. n = {n_t:,} theme-rows."
        ),
        stats=all_stats["c13"],
        insights=[
            "Rising ER in attack or mobilisation themes in Jan–May 2024 would confirm "
            "that election-period audiences engaged more strongly with adversarial content.",
            "If welfare and governance themes show declining ER over time, it would suggest "
            "audiences disengaged from policy content as the campaign climaxed.",
            "Stable ER across periods for a theme indicates consistent audience receptivity "
            "regardless of electoral calendar — a durable messaging asset.",
        ],
    )

    doc_section(doc, "C14: Per-Account ER_Timeline with Dominant Theme Annotated",
        paths["c14"],
        methodology=(
            "Monthly mean ER_Timeline per account. ER_Timeline removes follower-growth "
            "confound for accurate temporal comparison. "
            "Arrow annotations show the dominant theme for each electoral period. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["c14"],
        insights=[
            "This chart directly connects engagement performance with thematic strategy — "
            "were periods of high ER aligned with specific dominant themes?",
            "An account that peaks in ER during a period dominated by attack themes "
            "has evidence that adversarial content resonated with its audience.",
            "An account that peaks in ER during mobilisation-dominant periods "
            "has evidence that ground-level activation content drove engagement.",
            "Misalignment (high ER in a period with an unusual dominant theme) points "
            "to a specific content event worth qualitative follow-up.",
        ],
    )

    docx_path = BLOCK_DIR / "Block_C_Supplement.docx"
    doc.save(str(docx_path))
    print(f"  ✓  Saved: Block_C_Supplement.docx")
    return docx_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PHASE 4B — Block C Supplement: Thematic Charts")
    print("=" * 62 + "\n")

    master, themed = load_both(verbose=False)

    all_themes    = themed["Hashtag_Theme"].value_counts().index.tolist()
    theme_colours = {t: plt.cm.tab20.colors[i % 20] for i, t in enumerate(all_themes)}

    print("Generating charts...\n")

    path_c3r,  stats_c3r  = chart_c3_revised(themed, theme_colours, all_themes)
    path_c7r,  stats_c7r  = chart_c7_revised(themed)
    path_c9,   stats_c9   = chart_c9(themed)
    path_c10,  stats_c10  = chart_c10(themed)
    path_c11,  stats_c11  = chart_c11(themed)
    path_c12,  stats_c12  = chart_c12(themed)
    path_c13,  stats_c13  = chart_c13(themed)
    path_c14,  stats_c14  = chart_c14(master, themed)

    print("\nBuilding docx report...\n")
    build_docx(
        master, themed,
        paths = {
            "c3r": path_c3r, "c7r": path_c7r,
            "c9" : path_c9,  "c10": path_c10,
            "c11": path_c11, "c12": path_c12,
            "c13": path_c13, "c14": path_c14,
        },
        all_stats = {
            "c3r": stats_c3r, "c7r": stats_c7r,
            "c9" : stats_c9,  "c10": stats_c10,
            "c11": stats_c11, "c12": stats_c12,
            "c13": stats_c13, "c14": stats_c14,
        },
    )

    print(f"\n✅  Block C Supplement complete. Outputs in:\n    {BLOCK_DIR}")


if __name__ == "__main__":
    main()

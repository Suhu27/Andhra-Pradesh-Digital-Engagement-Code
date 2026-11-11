# phase6_crossparty.py
# ─────────────────────────────────────────────────────────────────────────────
# Block E: Cross-Party Comparative Analysis
# E1:  Ruling vs Opposition — post volume comparison
# E2:  Ruling vs Opposition — mean ER_Monthly comparison
# E3:  Side-by-side account radar chart (thematic composition)
# E4:  Attack theme intensity per account over time
# E5:  Incumbent defence vs challenger attack — ER comparison
# E6:  Leader account vs Party account — volume & ER comparison
# E7:  TDP + NCBN (alliance) combined vs YSRCP — volume & ER
# E8:  Posting consistency — coefficient of variation in monthly volume
# E9:  Multi-metric comparison table (docx only — all key metrics)
# E10: Correlation matrix — ER, likes, shares, comments, theme_count
# Output: Analysis_Output/E_CrossParty/
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
BLOCK_DIR = ANALYSIS_DIR / "E_CrossParty"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

ACC_LABELS = {"ncbn": "NCBN", "tdp": "TDP", "ysjagan": "YS Jagan", "ysrcp": "YSRCP"}
ACC_TO_KEY = {v: k for k, v in ACC_LABELS.items()}
PER_LABELS = {
    "JunAug2023": "Jun–Aug 2023",
    "SepDec2023": "Sep–Dec 2023",
    "JanMay2024": "Jan–May 2024",
}

# Political context groupings
RULING   = ["ysrcp", "ysjagan"]   # YSRCP in power; ysjagan = CM's personal account
CHALLENG = ["tdp", "ncbn"]        # TDP = party; ncbn = Chandrababu personal account
ALLIANCE = ["tdp", "ncbn"]        # TDP–NCBN alliance
LEADER   = ["ncbn", "ysjagan"]    # Personal leader accounts
PARTY    = ["tdp", "ysrcp"]       # Party accounts

ATTACK_THEMES = {
    "Attacking Leadership/Demonisation",
    "Attacking Party",
    "Anti-incumbency/Attacking Governance",
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


# ── E1: Ruling vs Opposition — Post Volume ────────────────────────────────────

def chart_e1(master):
    master["bloc"] = master["account"].map(
        lambda a: "Ruling (YSRCP + YS Jagan)"
        if a in RULING else "Opposition (TDP + NCBN)"
    )
    grp = (
        master.groupby(["period", "bloc"], observed=False)
        .size().reset_index(name="posts")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "E1: Ruling vs Opposition — Post Volume per Period\n"
        "(Aggregate of YSRCP+YS Jagan vs TDP+NCBN)",
        fontsize=12, fontweight="bold"
    )

    blocs   = ["Ruling (YSRCP + YS Jagan)", "Opposition (TDP + NCBN)"]
    colours = {"Ruling (YSRCP + YS Jagan)": "#C0392B",
               "Opposition (TDP + NCBN)"  : "#2271B3"}

    # Left: grouped bar per period
    ax  = axes[0]
    x   = np.arange(len(PERIOD_ORDER))
    w   = 0.35
    stats_out = {}
    for j, bloc in enumerate(blocs):
        vals = [int(grp[(grp["period"] == p) & (grp["bloc"] == bloc)]["posts"].sum())
                for p in PERIOD_ORDER]
        offset = (j - 1) * w + w / 2
        bars = ax.bar(x + offset, vals, w,
                      label=bloc, color=colours[bloc], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 30, f"{v:,}",
                    ha="center", va="bottom", fontsize=8)
        stats_out[bloc] = {PER_LABELS[p]: v for p, v in zip(PERIOD_ORDER, vals)}
    ax.set_title("Per Period", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER], rotation=10)
    ax.set_ylabel("Number of Posts"); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Right: overall totals
    ax2 = axes[1]
    totals  = {b: sum(v for v in stats_out[b].values()) for b in blocs}
    colours2 = [colours[b] for b in blocs]
    bars2    = ax2.bar([b.split(" ")[0] for b in blocs],
                       list(totals.values()), color=colours2, alpha=0.85, width=0.4)
    for bar, v in zip(bars2, totals.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 50, f"{v:,}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_title("Overall Total", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Total Posts"); ax2.set_ylim(0, max(totals.values()) * 1.2)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "Ruling bloc = YSRCP + YS Jagan (incumbent government, Jun 2023–May 2024). "
        "Opposition bloc = TDP + NCBN (challenger alliance). "
        f"Total n = {len(master):,} posts.", y=0.01)

    stats_list = [(f"{b} — {p}", f"{v:,}") for b, pd in stats_out.items()
                  for p, v in pd.items()]
    stats_list += [(f"{b} — Overall Total", f"{v:,}") for b, v in totals.items()]
    return save_fig(fig, "E1_Ruling_vs_Opposition_Volume.png"), stats_list


# ── E2: Ruling vs Opposition — Mean ER_Monthly ────────────────────────────────

def chart_e2(master):
    master["bloc"] = master["account"].map(
        lambda a: "Ruling" if a in RULING else "Opposition"
    )
    grp = (
        master.groupby(["period", "bloc"], observed=False)[ER_MONTHLY]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "E2: Ruling vs Opposition — Mean ER_Monthly per Period\n"
        "(YSRCP+YS Jagan vs TDP+NCBN)",
        fontsize=12, fontweight="bold"
    )
    blocs   = ["Ruling", "Opposition"]
    colours = {"Ruling": "#C0392B", "Opposition": "#2271B3"}
    stats_out = {}

    for ax, metric, title_str in zip(axes, ["mean", "median"],
                                      ["Mean ER_Monthly (%)", "Median ER_Monthly (%)"]):
        x, w = np.arange(len(PERIOD_ORDER)), 0.35
        for j, bloc in enumerate(blocs):
            vals = []
            for p in PERIOD_ORDER:
                row = grp[(grp["period"] == p) & (grp["bloc"] == bloc)]
                vals.append(float(row[metric].iloc[0]) if not row.empty else 0.0)
            offset = (j - 1) * w + w / 2
            bars = ax.bar(x + offset, vals, w,
                          label=bloc, color=colours[bloc], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.4f}%", ha="center", va="bottom", fontsize=7.5)
            if metric == "mean":
                stats_out[bloc] = {
                    PER_LABELS[p]: round(v, 4) for p, v in zip(PERIOD_ORDER, vals)
                }
        ax.set_title(title_str, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER], rotation=10)
        ax.set_ylabel(title_str); ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "Mean and median ER_Monthly per political bloc per electoral period. "
        "Ruling = YSRCP + YS Jagan. Opposition = TDP + NCBN. "
        f"n = {len(master):,} posts.", y=0.01)

    stats_list = [(f"{b} — {p}", f"Mean ER: {v:.4f}%")
                  for b, pd in stats_out.items() for p, v in pd.items()]
    return save_fig(fig, "E2_Ruling_vs_Opposition_ER.png"), stats_list


# ── E3: Radar Chart — Thematic Composition per Account ───────────────────────

def chart_e3(themed):
    all_themes = themed["Hashtag_Theme"].value_counts().index.tolist()
    pivot = (
        themed.groupby(["account", "Hashtag_Theme"], observed=True)
        .size().unstack(fill_value=0)
    )
    pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    N          = len(all_themes)
    angles     = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles    += angles[:1]
    labels     = [sh(t) for t in all_themes]

    fig, ax = plt.subplots(figsize=(12, 12),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for account in PARTY_ORDER:
        values  = [float(pct.loc[account, t]) if t in pct.columns else 0.0
                   for t in all_themes]
        values += values[:1]
        ax.plot(angles, values, linewidth=2,
                label=ACC_LABELS[account], color=PARTY_COLOURS[account])
        ax.fill(angles, values, alpha=0.08, color=PARTY_COLOURS[account])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_title(
        "E3: Thematic Composition Radar Chart\n"
        "(% of Theme-Rows per Account for All 15 Themes)",
        size=13, fontweight="bold", pad=25
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    add_method_note(fig,
        "Radar axes = 15 themes. Each vertex = % of that account's theme-rows. "
        "Shape = thematic identity of each account. "
        f"n = {len(themed):,} theme-rows.", y=0.0)

    stats_out = [
        (f"{ACC_LABELS[acc]} — {sh(t)}",
         f"{float(pct.loc[acc, t]):.1f}%" if t in pct.columns else "0.0%")
        for acc in PARTY_ORDER for t in all_themes[:5]
    ]
    return save_fig(fig, "E3_Radar_Thematic_Composition.png"), stats_out


# ── E4: Attack Theme Intensity per Account over Time ─────────────────────────

def chart_e4(themed):
    df = themed.copy()
    df["is_attack"] = df["Hashtag_Theme"].isin(ATTACK_THEMES)

    monthly = (
        df.groupby(["month_date", "account"], observed=True)
        .agg(attack_rows=("is_attack", "sum"), total_rows=("is_attack", "count"))
        .reset_index()
    )
    monthly["attack_pct"] = monthly["attack_rows"] / monthly["total_rows"] * 100
    monthly = monthly.sort_values("month_date")

    fig, ax = plt.subplots(figsize=(14, 6))
    max_pct = float(monthly["attack_pct"].max())

    for account in PARTY_ORDER:
        sub = monthly[monthly["account"] == account]
        ax.plot(sub["month_date"], sub["attack_pct"],
                marker="o", linewidth=2,
                label=ACC_LABELS[account],
                color=PARTY_COLOURS[account])

    for dt_str, lbl in [("2023-09-01", "Sep 2023 →"), ("2024-01-01", "Jan 2024 →")]:
        dt = pd.Timestamp(dt_str)
        ax.axvline(dt, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(dt, max_pct * 0.97, lbl, fontsize=8, color="gray", ha="left", va="top")

    ax.set_title(
        "E4: Monthly Attack Theme Intensity per Account\n"
        "(%  of Monthly Theme-Rows that are Attack Themes)",
        pad=15
    )
    ax.set_xlabel("Month (IST)")
    ax.set_ylabel("Attack Theme % of Monthly Theme-Rows")
    ax.legend(); fig.autofmt_xdate()

    add_method_note(fig,
        "Attack themes: Attacking Leadership/Demonisation, Attacking Party, "
        "Anti-incumbency/Attacking Governance. "
        "Attack % = attack theme-rows / total theme-rows per account per month × 100. "
        f"n = {len(themed):,} theme-rows.")

    stats_out = []
    for account in PARTY_ORDER:
        sub = monthly[monthly["account"] == account]
        if not sub.empty:
            peak_row = sub.loc[sub["attack_pct"].idxmax()]
            stats_out.append((
                f"{ACC_LABELS[account]} — Peak attack month / Peak %",
                f"{peak_row['month_date'].strftime('%b %Y')} / {peak_row['attack_pct']:.1f}%"
            ))
            stats_out.append((
                f"{ACC_LABELS[account]} — Overall mean attack %",
                f"{sub['attack_pct'].mean():.1f}%"
            ))
    return save_fig(fig, "E4_Attack_Theme_Intensity_Timeline.png"), stats_out


# ── E5: Incumbent Defence vs Challenger Attack ────────────────────────────────

def chart_e5(themed):
    """Compare: ruling accounts' branding/welfare vs opposition's attack ER."""
    INCUMBENT_THEMES = {
        "Leader-centric branding",
        "Welfare&Governance",
        "Welfare & Governance",
        "Promoting Party",
        "Leadership legacy",
    }
    CHALLENGER_THEMES = ATTACK_THEMES | {"Anti-incumbency/Attacking Governance"}

    df = themed.copy()
    df["strategy"] = "Other"
    df.loc[
        (df["account"].isin(RULING)) &
        (df["Hashtag_Theme"].isin(INCUMBENT_THEMES)), "strategy"
    ] = "Ruling — Defence/Branding"
    df.loc[
        (df["account"].isin(CHALLENG)) &
        (df["Hashtag_Theme"].isin(CHALLENGER_THEMES)), "strategy"
    ] = "Opposition — Attack"

    sub = df[df["strategy"] != "Other"]
    grp = (
        sub.groupby(["strategy", "period"], observed=True)[ER_MONTHLY]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    strategies = ["Ruling — Defence/Branding", "Opposition — Attack"]
    colours    = {"Ruling — Defence/Branding": "#C0392B",
                  "Opposition — Attack"      : "#2271B3"}
    x, w       = np.arange(len(PERIOD_ORDER)), 0.35
    stats_out  = {}

    for j, strat in enumerate(strategies):
        vals = []
        for p in PERIOD_ORDER:
            row = grp[(grp["strategy"] == strat) & (grp["period"] == p)]
            vals.append(float(row["mean"].iloc[0]) if not row.empty else 0.0)
        offset = (j - 1) * w + w / 2
        bars   = ax.bar(x + offset, vals, w,
                        label=strat, color=colours[strat], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{v:.4f}%", ha="center", va="bottom", fontsize=8)
        stats_out[strat] = {PER_LABELS[p]: round(v, 4) for p, v in zip(PERIOD_ORDER, vals)}

    ax.set_title(
        "E5: Incumbent Defence/Branding vs Challenger Attack\n"
        "(Mean ER_Monthly per Strategic Theme Category per Period)",
        pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER])
    ax.set_ylabel("Mean ER_Monthly (%)"); ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    add_method_note(fig,
        "Ruling Defence themes: Leader-centric branding, Welfare&Governance, "
        "Promoting Party, Leadership legacy — from YSRCP/YS Jagan only. "
        "Opposition Attack themes: Attacking Leadership, Attacking Party, "
        "Anti-incumbency — from TDP/NCBN only. "
        f"n = {len(sub):,} theme-rows.")

    stats_list = [(f"{s} — {p}", f"Mean ER: {v:.4f}%")
                  for s, pd in stats_out.items() for p, v in pd.items()]
    return save_fig(fig, "E5_Incumbent_vs_Challenger_ER.png"), stats_list


# ── E6: Leader vs Party Account ───────────────────────────────────────────────

def chart_e6(master):
    master["acct_type"] = master["account"].map(
        lambda a: "Leader Account (NCBN / YS Jagan)" if a in LEADER
        else "Party Account (TDP / YSRCP)"
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "E6: Leader Account vs Party Account\n"
        "(Post Volume, Mean ER_Monthly, Mean Interactions per Post)",
        fontsize=12, fontweight="bold"
    )

    types   = ["Leader Account (NCBN / YS Jagan)", "Party Account (TDP / YSRCP)"]
    colours = {"Leader Account (NCBN / YS Jagan)": "#8E44AD",
               "Party Account (TDP / YSRCP)"     : "#1A5276"}
    stats_out = {}

    # Post volume
    vol = master.groupby("acct_type", observed=False).size()
    axes[0].bar([t.split(" ")[0] for t in types],
                [int(vol.get(t, 0)) for t in types],
                color=[colours[t] for t in types], alpha=0.85, width=0.4)
    for i, t in enumerate(types):
        v = int(vol.get(t, 0))
        axes[0].text(i, v + 50, f"{v:,}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[0].set_title("Post Volume", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Number of Posts")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # Mean ER_Monthly
    er = master.groupby("acct_type", observed=False)[ER_MONTHLY].mean()
    axes[1].bar([t.split(" ")[0] for t in types],
                [float(er.get(t, 0)) for t in types],
                color=[colours[t] for t in types], alpha=0.85, width=0.4)
    for i, t in enumerate(types):
        v = float(er.get(t, 0))
        axes[1].text(i, v + 0.005, f"{v:.4f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[1].set_title("Mean ER_Monthly", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Mean ER_Monthly (%)")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    # Mean interactions per post
    inter = master.groupby("acct_type", observed=False)[
        ["likes", "comments_count", "shares"]
    ].mean().sum(axis=1)
    axes[2].bar([t.split(" ")[0] for t in types],
                [float(inter.get(t, 0)) for t in types],
                color=[colours[t] for t in types], alpha=0.85, width=0.4)
    for i, t in enumerate(types):
        v = float(inter.get(t, 0))
        axes[2].text(i, v + 2, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[2].set_title("Mean Total Interactions\nper Post", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Mean Interactions (likes+comments+shares)")
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "Leader accounts: NCBN (N. Chandrababu Naidu personal) + YS Jagan (CM personal). "
        "Party accounts: TDP (party page) + YSRCP (party page). "
        f"n = {len(master):,} posts.", y=0.01)

    stats_out = {
        t: {
            "volume"    : int(vol.get(t, 0)),
            "mean_er"   : round(float(er.get(t, 0)), 4),
            "mean_inter": round(float(inter.get(t, 0)), 2),
        }
        for t in types
    }
    stats_list = [(f"{t.split(' ')[0]} — Volume / Mean ER / Mean Interactions",
                   f"{v['volume']:,} / {v['mean_er']:.4f}% / {v['mean_inter']:.1f}")
                  for t, v in stats_out.items()]
    return save_fig(fig, "E6_Leader_vs_Party_Account.png"), stats_list


# ── E7: Alliance (TDP+NCBN) vs YSRCP ─────────────────────────────────────────

def chart_e7(master, themed):
    master["group"] = master["account"].map(
        lambda a: "TDP+NCBN Alliance" if a in ALLIANCE else
                  "YSRCP" if a == "ysrcp" else "YS Jagan"
    )
    themed["group"] = themed["account"].map(
        lambda a: "TDP+NCBN Alliance" if a in ALLIANCE else
                  "YSRCP" if a == "ysrcp" else "YS Jagan"
    )

    groups  = ["TDP+NCBN Alliance", "YSRCP"]
    colours = {"TDP+NCBN Alliance": "#2271B3", "YSRCP": "#1A4FA0"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(
        "E7: TDP+NCBN Alliance vs YSRCP\n"
        "(Post Volume | Mean ER_Monthly | Thematic Attack Intensity)",
        fontsize=12, fontweight="bold"
    )

    # Volume per period
    vol = master[master["group"].isin(groups)].groupby(
        ["period", "group"], observed=True
    ).size().reset_index(name="posts")
    x, w = np.arange(len(PERIOD_ORDER)), 0.35
    for j, grp_name in enumerate(groups):
        vals   = [int(vol[(vol["period"] == p) & (vol["group"] == grp_name)]["posts"].sum())
                  for p in PERIOD_ORDER]
        offset = (j - 1) * w + w / 2
        bars   = axes[0].bar(x + offset, vals, w,
                             label=grp_name, color=colours[grp_name], alpha=0.85)
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 30, f"{v:,}",
                         ha="center", va="bottom", fontsize=7.5)
    axes[0].set_title("Post Volume per Period", fontsize=10, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER], rotation=10)
    axes[0].set_ylabel("Posts"); axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # ER per period
    er = master[master["group"].isin(groups)].groupby(
        ["period", "group"], observed=True
    )[ER_MONTHLY].mean().reset_index()
    for j, grp_name in enumerate(groups):
        vals   = [float(er[(er["period"] == p) & (er["group"] == grp_name)][ER_MONTHLY].iloc[0])
                  if not er[(er["period"] == p) & (er["group"] == grp_name)].empty else 0.0
                  for p in PERIOD_ORDER]
        offset = (j - 1) * w + w / 2
        bars   = axes[1].bar(x + offset, vals, w,
                             label=grp_name, color=colours[grp_name], alpha=0.85)
        for bar, v in zip(bars, vals):
            axes[1].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.003, f"{v:.4f}%",
                         ha="center", va="bottom", fontsize=7.5)
    axes[1].set_title("Mean ER_Monthly per Period", fontsize=10, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([PER_LABELS[p] for p in PERIOD_ORDER], rotation=10)
    axes[1].set_ylabel("Mean ER_Monthly (%)"); axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    # Attack theme intensity overall
    att = themed[themed["group"].isin(groups)].copy()
    att["is_attack"] = att["Hashtag_Theme"].isin(ATTACK_THEMES)
    att_grp = att.groupby("group", observed=True).agg(
        attack=("is_attack", "sum"), total=("is_attack", "count")
    ).reset_index()
    att_grp["pct"] = att_grp["attack"] / att_grp["total"] * 100

    att_grp_filtered = att_grp[att_grp["group"].isin(groups)]
    bars3 = axes[2].bar(att_grp_filtered["group"],
                        att_grp_filtered["pct"],
                        color=[colours[g] for g in att_grp_filtered["group"]],
                        alpha=0.85, width=0.4)
    for bar, v in zip(bars3, att_grp_filtered["pct"]):
        axes[2].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[2].set_title("Overall Attack Theme Intensity\n(% of Theme-Rows)",
                      fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Attack Theme %")
    axes[2].grid(axis="y", alpha=0.3, linestyle="--")
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "TDP+NCBN Alliance = combined TDP party + NCBN personal account posts. "
        "YSRCP = YSRCP party account only (YS Jagan personal excluded for cleaner party-level comparison). "
        f"n = {len(master):,} posts | n = {len(themed):,} theme-rows.", y=0.01)

    stats_list = []
    for g in att_grp_filtered["group"]:
        row = att_grp_filtered[att_grp_filtered["group"] == g].iloc[0]
        stats_list.append((f"{g} — Attack theme %", f"{row['pct']:.1f}%"))
    return save_fig(fig, "E7_Alliance_vs_YSRCP.png"), stats_list


# ── E8: Posting Consistency (CV in Monthly Volume) ───────────────────────────

def chart_e8(master):
    monthly = (
        master.groupby(["month_date", "account"], observed=True)
        .size().reset_index(name="posts")
    )
    cv_stats = {}
    for account in PARTY_ORDER:
        sub = monthly[monthly["account"] == account]["posts"]
        mean_v = float(sub.mean())
        std_v  = float(sub.std())
        cv     = std_v / mean_v * 100 if mean_v > 0 else 0
        cv_stats[ACC_LABELS[account]] = {
            "mean_monthly": round(mean_v, 1),
            "std_monthly" : round(std_v,  1),
            "cv_pct"      : round(cv,     1),
            "min_monthly" : int(sub.min()),
            "max_monthly" : int(sub.max()),
        }

    labels  = list(cv_stats.keys())
    cvs     = [cv_stats[l]["cv_pct"] for l in labels]
    colours = [PARTY_COLOURS[ACC_TO_KEY[l]] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "E8: Posting Consistency per Account\n"
        "(Coefficient of Variation in Monthly Post Volume)",
        fontsize=12, fontweight="bold"
    )

    # Left: CV bar
    bars = axes[0].bar(labels, cvs, color=colours, alpha=0.85, width=0.5)
    for bar, v in zip(bars, cvs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f"{v:.1f}%",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[0].set_title("Coefficient of Variation (%)\n(Lower = More Consistent)",
                      fontsize=10, fontweight="bold")
    axes[0].set_ylabel("CV (%)"); axes[0].set_xlabel("Account")
    axes[0].grid(axis="y", alpha=0.3, linestyle="--")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    # Right: mean ± std error bars
    means = [cv_stats[l]["mean_monthly"] for l in labels]
    stds  = [cv_stats[l]["std_monthly"]  for l in labels]
    axes[1].bar(labels, means, color=colours, alpha=0.75, width=0.5)
    axes[1].errorbar(labels, means, yerr=stds, fmt="none",
                     color="black", capsize=6, linewidth=2)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[1].text(i, m + s + 5, f"{m:.0f}±{s:.0f}",
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_title("Mean ± Std Dev Monthly Post Volume",
                      fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Monthly Posts"); axes[1].set_xlabel("Account")
    axes[1].grid(axis="y", alpha=0.3, linestyle="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "CV = std dev / mean monthly post count × 100. "
        "Lower CV = more consistent posting cadence. "
        f"Computed over {master['month_date'].nunique()} months "
        f"(Jun 2023–May 2024). n = {len(master):,} posts.", y=0.01)

    stats_list = [(f"{l} — Mean monthly / Std / CV",
                   f"{v['mean_monthly']:.0f} / {v['std_monthly']:.0f} / {v['cv_pct']:.1f}%")
                  for l, v in cv_stats.items()]
    return save_fig(fig, "E8_Posting_Consistency_CV.png"), stats_list


# ── E9: Multi-Metric Summary Table (docx only) ───────────────────────────────

def build_summary_table(doc, master, themed):
    doc.add_heading("E9: Multi-Metric Comparative Summary Table", level=1)
    p = doc.add_paragraph()
    p.add_run("Methodology: ").bold = True
    p.add_run(
        "Summary of all key metrics per account across Blocks A–E. "
        "Engagement metrics use ER_Monthly (monthly follower denominator). "
        "Attack % = attack theme-rows / total theme-rows × 100."
    )
    doc.add_paragraph()

    metrics = []
    for account in PARTY_ORDER:
        lbl     = ACC_LABELS[account]
        m_sub   = master[master["account"] == account]
        t_sub   = themed[themed["account"] == account]
        atk_sub = t_sub[t_sub["Hashtag_Theme"].isin(ATTACK_THEMES)]
        dom_t   = t_sub["Hashtag_Theme"].value_counts().idxmax() if len(t_sub) > 0 else "—"
        monthly_cv = (
            master[master["account"] == account]
            .groupby("month_date", observed=True).size()
        )
        cv = float(monthly_cv.std() / monthly_cv.mean() * 100) if monthly_cv.mean() > 0 else 0

        metrics.append({
            "Account"           : lbl,
            "Total Posts"       : f"{len(m_sub):,}",
            "Theme-Rows"        : f"{len(t_sub):,}",
            "Unclassified %"    : f"{m_sub['Hashtag_Theme'].isna().mean()*100:.1f}%",
            "Mean ER_Monthly"   : f"{m_sub[ER_MONTHLY].mean():.4f}%",
            "Median ER_Monthly" : f"{m_sub[ER_MONTHLY].median():.4f}%",
            "Mean Likes/Post"   : f"{m_sub['likes'].mean():.1f}",
            "Mean Comments/Post": f"{m_sub['comments_count'].mean():.1f}",
            "Mean Shares/Post"  : f"{m_sub['shares'].mean():.1f}",
            "Attack Theme %"    : f"{len(atk_sub)/len(t_sub)*100:.1f}%" if len(t_sub) > 0 else "—",
            "Dominant Theme"    : sh(dom_t),
            "Posting CV %"      : f"{cv:.1f}%",
        })

    cols  = list(metrics[0].keys())
    table = doc.add_table(rows=1, cols=len(cols))
    table.style = "Light List Accent 1"
    for i, col in enumerate(cols):
        cell = table.rows[0].cells[i]
        cell.text = col
        for run in cell.paragraphs[0].runs:
            run.bold = True

    for row_data in metrics:
        cells = table.add_row().cells
        for i, col in enumerate(cols):
            cells[i].text = row_data[col]

    doc.add_paragraph()
    doc.add_page_break()


# ── E10: Correlation Matrix ───────────────────────────────────────────────────

def chart_e10(master, themed):
    # Post-level: deduplicate themed to post level for theme_count
    tc = (
        themed.drop_duplicates(subset=["account", "date", "platform"])
        [["account", "date", "platform", "theme_count"]]
    )
    df = master.merge(tc, on=["account", "date", "platform"], how="left")
    df["theme_count"] = df["theme_count"].fillna(0).astype(int)

    cols_corr = {
        ER_MONTHLY        : "ER_Monthly",
        "likes"           : "Likes",
        "comments_count"  : "Comments",
        "shares"          : "Shares",
        "theme_count"     : "Theme Count",
        ER_TIMELINE       : "ER_Timeline",
    }
    sub = df[list(cols_corr.keys())].rename(columns=cols_corr)
    corr = sub.corr(method="pearson")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "E10: Correlation Matrix — ER, Interactions & Theme Count\n"
        "(Pearson r — All Posts Combined + Per-Account Subplots)",
        fontsize=12, fontweight="bold"
    )

    # Left: overall
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r",
                ax=axes[0], linewidths=0.5, center=0,
                annot_kws={"size": 9},
                cbar_kws={"label": "Pearson r"})
    axes[0].set_title("All Accounts Combined", fontsize=10, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    axes[0].tick_params(axis="y", labelsize=8)

    # Right: r(ER_Monthly, Likes/Comments/Shares) per account
    corr_by_acc = []
    for account in PARTY_ORDER:
        a_sub = df[df["account"] == account]
        tc2   = a_sub[list(cols_corr.keys())].rename(columns=cols_corr)
        row   = {"Account": ACC_LABELS[account]}
        for col in ["Likes", "Comments", "Shares", "Theme Count"]:
            row[f"r(ER, {col})"] = round(float(tc2[["ER_Monthly", col]].corr().iloc[0, 1]), 3)
        corr_by_acc.append(row)
    ca_df = pd.DataFrame(corr_by_acc).set_index("Account")

    sns.heatmap(ca_df, annot=True, fmt=".3f", cmap="RdBu_r",
                ax=axes[1], linewidths=0.5, center=0,
                annot_kws={"size": 10},
                cbar_kws={"label": "Pearson r"})
    axes[1].set_title("r(ER_Monthly, metric) per Account",
                      fontsize=10, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=20, labelsize=9)
    axes[1].tick_params(axis="y", labelsize=9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    add_method_note(fig,
        "Pearson r correlation. +1 = perfect positive, -1 = perfect negative, 0 = no linear relationship. "
        "Right panel: r between ER_Monthly and each metric per account. "
        f"n = {len(df):,} posts.", y=0.01)

    stats_list = [(f"r(ER_Monthly, {col})",
                   f"{float(corr.loc['ER_Monthly', col]):.3f}")
                  for col in ["Likes", "Comments", "Shares", "Theme Count"]]
    return save_fig(fig, "E10_Correlation_Matrix.png"), stats_list


# ── BUILD DOCX ────────────────────────────────────────────────────────────────

def build_docx(master, themed, paths, all_stats):
    doc = init_doc("Block E: Cross-Party Comparative Analysis")
    n_m = len(master); n_t = len(themed)

    p = doc.add_paragraph()
    p.add_run("Dataset Overview: ").bold = True
    p.add_run(
        f"This block places the four accounts in their political context: "
        f"YSRCP and YS Jagan = ruling bloc (YSRCP in power, Jun 2023–May 2024); "
        f"TDP and NCBN = opposition/challenger bloc (TDP–NCBN alliance). "
        f"Analyses compare digital communication strategies by political role, "
        f"account type (leader vs party), and alliance grouping. "
        f"n = {n_m:,} posts | n = {n_t:,} theme-rows."
    )
    doc.add_paragraph()

    doc_section(doc, "E1: Ruling vs Opposition — Post Volume",
        paths["e1"],
        methodology=(
            "Ruling bloc = YSRCP + YS Jagan. Opposition = TDP + NCBN. "
            "Post counts aggregated per bloc per electoral period. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["e1"],
        insights=[
            "Volume comparison between blocs reveals whether the ruling party or "
            "opposition mounted a more aggressive digital presence.",
            "Ruling parties in incumbency often have more institutional resources "
            "for digital communication — a higher ruling bloc volume would reflect this.",
            "Opposition over-production of content relative to the ruling party would "
            "suggest a digitally-driven challenger campaign strategy.",
            "Period-by-period divergence shows whether volume parity shifted as "
            "elections approached — acceleration by either bloc in Jan–May 2024 "
            "indicates campaign-phase escalation.",
        ],
    )

    doc_section(doc, "E2: Ruling vs Opposition — Mean ER_Monthly",
        paths["e2"],
        methodology=(
            "Mean and median ER_Monthly per political bloc per electoral period. "
            "Ruling = YSRCP + YS Jagan. Opposition = TDP + NCBN. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["e2"],
        insights=[
            "ER comparison reveals which bloc had more audience-resonant content — "
            "independent of volume.",
            "Higher opposition ER would suggest challenger content is more emotionally "
            "activating for audiences than incumbent communication.",
            "Higher ruling bloc ER may indicate governance communication or "
            "welfare messaging drives strong engagement from beneficiary audiences.",
            "Divergence in ER trajectory across periods reveals whether one bloc "
            "gained or lost audience resonance as election day approached.",
        ],
    )

    doc_section(doc, "E3: Thematic Composition Radar Chart",
        paths["e3"],
        methodology=(
            "Radar chart with 15 axes (one per theme). Each axis = % of that account's "
            "theme-rows. Shape = thematic identity of each account. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["e3"],
        insights=[
            "The radar chart is the most intuitive representation of each account's "
            "thematic identity — the shape of the polygon = communication persona.",
            "Overlapping shapes indicate thematic convergence between accounts; "
            "divergent shapes confirm distinct communication strategies.",
            "Ruling bloc accounts (YSRCP, YS Jagan) and opposition accounts (TDP, NCBN) "
            "should show clear shape differences reflecting incumbent vs challenger postures.",
            "Within-bloc shape similarity (YSRCP vs YS Jagan, TDP vs NCBN) reveals "
            "coordination of messaging between the party and its leader's personal accounts.",
        ],
    )

    doc_section(doc, "E4: Attack Theme Intensity Timeline per Account",
        paths["e4"],
        methodology=(
            "Attack theme % = attack theme-rows / total theme-rows per account per month × 100. "
            "Attack themes: Attacking Leadership/Demonisation, Attacking Party, "
            "Anti-incumbency/Attacking Governance. "
            f"n = {n_t:,} theme-rows."
        ),
        stats=all_stats["e4"],
        insights=[
            "Attack intensity rising over time confirms a campaign-phase escalation "
            "strategy — adversarial content increases as elections approach.",
            "Opposition accounts are expected to show higher attack intensity than ruling "
            "accounts, but ruling-party attacks on opposition records are also common.",
            "Spikes in attack % aligned with specific events (alliance announcements, "
            "policy controversies) would indicate reactive attack content.",
            "Sustained high attack intensity from the very start of the study period "
            "indicates a pre-planned adversarial campaign strategy rather than reactive attacks.",
        ],
    )

    doc_section(doc, "E5: Incumbent Defence/Branding vs Challenger Attack ER",
        paths["e5"],
        methodology=(
            "Ruling defence themes (YSRCP/YS Jagan only): Leader-centric branding, "
            "Welfare&Governance, Promoting Party, Leadership legacy. "
            "Challenger attack themes (TDP/NCBN only): Attacking Leadership, "
            "Attacking Party, Anti-incumbency. "
            f"Mean ER_Monthly per category per period. n = {n_t:,} theme-rows."
        ),
        stats=all_stats["e5"],
        insights=[
            "This chart directly tests the electoral communication hypothesis: "
            "does attack content outperform defence/branding content in terms of audience engagement?",
            "If challenger attack ER > ruling defence ER, it supports the theory that "
            "opposition attack messaging is more audience-activating than incumbent positional content.",
            "The reverse would suggest ruling party governance and welfare messaging "
            "builds stronger engagement — possibly from beneficiary audience segments.",
            "Period-by-period trends reveal whether the ER gap between attack and defence "
            "content widened or narrowed as election day approached.",
        ],
    )

    doc_section(doc, "E6: Leader Account vs Party Account",
        paths["e6"],
        methodology=(
            "Leader accounts: NCBN (personal) + YS Jagan (personal). "
            "Party accounts: TDP (party page) + YSRCP (party page). "
            "Volume, mean ER_Monthly, and mean total interactions compared. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["e6"],
        insights=[
            "Leader accounts typically generate higher ER per post despite lower "
            "absolute volume — personalised content drives stronger individual audience connection.",
            "Party accounts produce higher post volumes as institutional communication "
            "machines, but may generate lower per-post ER.",
            "Higher mean interactions per post for leader accounts would confirm that "
            "voters engage more with individual personalities than party institutions.",
            "This comparison has direct implications for campaign strategy: "
            "investing in leader-centric content may yield higher returns per post "
            "than equivalent party-page content.",
        ],
    )

    doc_section(doc, "E7: TDP+NCBN Alliance vs YSRCP",
        paths["e7"],
        methodology=(
            "TDP+NCBN Alliance = combined TDP party + NCBN personal posts. "
            "YSRCP = YSRCP party page only (YS Jagan personal excluded for clean party comparison). "
            "Volume, ER, and attack intensity compared per period. "
            f"n = {n_m:,} posts | n = {n_t:,} theme-rows."
        ),
        stats=all_stats["e7"],
        insights=[
            "The TDP–NCBN alliance forms the principal electoral opposition to YSRCP — "
            "this comparison is the core ruling vs challenger contrast in the dataset.",
            "Alliance volume exceeding YSRCP would indicate the opposition mounted "
            "a more aggressive digital campaign, consistent with challenger dynamics.",
            "If YSRCP's attack % is higher than the alliance, it would indicate the "
            "ruling party adopted an unusually adversarial posture despite incumbency.",
            "ER parity between the alliance and YSRCP would suggest comparable "
            "audience engagement efficiency despite different volume levels.",
        ],
    )

    doc_section(doc, "E8: Posting Consistency (Coefficient of Variation)",
        paths["e8"],
        methodology=(
            "CV = std dev / mean monthly post count × 100. "
            "Computed over all months Jun 2023–May 2024. "
            "Lower CV = more consistent posting cadence. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["e8"],
        insights=[
            "Low CV accounts maintain a stable posting cadence — "
            "indicative of a planned, professionally managed digital strategy.",
            "High CV accounts show erratic posting patterns — bursts of activity "
            "followed by quiet periods — suggesting event-driven or reactive content strategy.",
            "An account with high CV and low overall volume likely campaigns only "
            "during politically significant moments rather than maintaining continuous presence.",
            "Consistency is a marker of campaign professionalism: institutionalised "
            "parties (TDP, YSRCP) are expected to show lower CV than personal accounts.",
        ],
    )

    # E9 — table embedded directly
    build_summary_table(doc, master, themed)

    doc_section(doc, "E10: Correlation Matrix — ER, Interactions & Theme Count",
        paths["e10"],
        methodology=(
            "Pearson r correlation between ER_Monthly, ER_Timeline, likes, comments, "
            "shares, and theme_count at post level. "
            "Left panel: all accounts combined. Right panel: per account. "
            f"n = {n_m:,} posts."
        ),
        stats=all_stats["e10"],
        insights=[
            "High r(ER, likes) confirms that likes are the primary driver of ER_Monthly — "
            "expected given likes dominate the numerator.",
            "r(ER, shares) provides a cleaner signal of content reach: shares extend "
            "content beyond the follower base, adding external engagement.",
            "r(ER, theme_count) tests whether thematic complexity helps or hurts engagement "
            "at the post level — a negative r would indicate focused posts outperform scattered ones.",
            "Per-account variation in correlations reveals structural differences in "
            "how each account's audience engages — some followers are sharers, "
            "others are commenters, shaping which interaction type drives ER most.",
        ],
    )

    docx_path = BLOCK_DIR / "Block_E_CrossParty_Analysis.docx"
    doc.save(str(docx_path))
    print(f"  ✓  Saved: Block_E_CrossParty_Analysis.docx")
    return docx_path


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PHASE 6 — Block E: Cross-Party Comparative Analysis")
    print("=" * 62 + "\n")

    master, themed = load_both(verbose=False)
    print("Generating charts...\n")

    path_e1,  stats_e1  = chart_e1(master.copy())
    path_e2,  stats_e2  = chart_e2(master.copy())
    path_e3,  stats_e3  = chart_e3(themed)
    path_e4,  stats_e4  = chart_e4(themed)
    path_e5,  stats_e5  = chart_e5(themed)
    path_e6,  stats_e6  = chart_e6(master.copy())
    path_e7,  stats_e7  = chart_e7(master.copy(), themed.copy())
    path_e8,  stats_e8  = chart_e8(master.copy())
    path_e10, stats_e10 = chart_e10(master.copy(), themed.copy())

    print("\nBuilding docx report...\n")
    build_docx(
        master, themed,
        paths = {
            "e1": path_e1, "e2": path_e2, "e3": path_e3, "e4": path_e4,
            "e5": path_e5, "e6": path_e6, "e7": path_e7, "e8": path_e8,
            "e10": path_e10,
        },
        all_stats = {
            "e1": stats_e1, "e2": stats_e2, "e3": stats_e3, "e4": stats_e4,
            "e5": stats_e5, "e6": stats_e6, "e7": stats_e7, "e8": stats_e8,
            "e10": stats_e10,
        },
    )

    print(f"\n✅  Block E complete. All outputs saved to:\n    {BLOCK_DIR}")


if __name__ == "__main__":
    main()

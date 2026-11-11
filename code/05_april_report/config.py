# ============================================================
#  CONFIG.PY — Shared Configuration & Helpers
#  AP Political Sentiment Analysis — Insights Extraction
#  All scripts import from this file.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script usage
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import sys
from datetime import datetime

# ── PATHS ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "phase3_classified_patched.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
CHART_DIR  = os.path.join(OUTPUT_DIR, "charts")
TEXT_DIR   = os.path.join(OUTPUT_DIR, "text")

for _d in [OUTPUT_DIR, CHART_DIR, TEXT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── ACCOUNT CONFIGURATION ────────────────────────────────────────────────────
ACC_COLORS = {
    'ysrcp':   '#2E8B57',   # Sea green
    'tdp':     '#DAA520',   # Goldenrod
    'ncbn':    '#4682B4',   # Steel blue
    'ysjagan': '#CD853F',   # Peru
}
ACC_LABELS = {
    'ysrcp': 'YSRCP', 'tdp': 'TDP',
    'ncbn': 'NCBN', 'ysjagan': 'YS Jagan',
}
ACC_ORDER = ['ysrcp', 'tdp', 'ncbn', 'ysjagan']

# ── ALLIANCE CONFIGURATION ───────────────────────────────────────────────────
ALLIANCE_MAP = {
    'ysrcp': 'YSRCP Alliance', 'ysjagan': 'YSRCP Alliance',
    'tdp': 'TDP Alliance',     'ncbn': 'TDP Alliance',
}
ALLIANCE_COLORS = {'YSRCP Alliance': '#2E8B57', 'TDP Alliance': '#DAA520'}
ALLIANCE_ORDER  = ['YSRCP Alliance', 'TDP Alliance']

# ── SENTIMENT COLOURS ────────────────────────────────────────────────────────
SENT_COLORS = {'Positive': '#2ECC71', 'Neutral': '#95A5A6', 'Negative': '#E74C3C'}
SENT_ORDER  = ['Positive', 'Neutral', 'Negative']

# ── EMOTION COLOURS ──────────────────────────────────────────────────────────
EMO_COLORS = {
    'Anger': '#E74C3C', 'Pride': '#F39C12', 'Fear': '#8E44AD',
    'Hope':  '#2ECC71', 'Trust': '#3498DB', 'Informational': '#95A5A6',
}
EMO_ORDER = ['Anger', 'Trust', 'Pride', 'Hope', 'Informational', 'Fear']

# ── TONE COLOURS ─────────────────────────────────────────────────────────────
TONE_COLORS = {
    'Aggressive':       '#E74C3C',
    'Persuasive':       '#3498DB',
    'Inspirational':    '#2ECC71',
    'Defensive':        '#F39C12',
    'Victim_Narrative': '#8E44AD',
    'Neutral':          '#BDC3C7',
}
TONE_ORDER = ['Aggressive', 'Persuasive', 'Inspirational', 'Victim_Narrative', 'Defensive']

# ── SUBJECTIVITY COLOURS ─────────────────────────────────────────────────────
SUBJ_COLORS = {'Fact': '#3498DB', 'Opinion': '#E74C3C', 'Mixed': '#F39C12'}
SUBJ_ORDER  = ['Fact', 'Mixed', 'Opinion']

# ── TIMELINE CONFIGURATION ───────────────────────────────────────────────────
TL_ORDER  = ['JunAug2023', 'SepDec2023', 'JanMay2024']
TL_LABELS = {
    'JunAug2023': 'Jun–Aug 2023',
    'SepDec2023': 'Sep–Dec 2023',
    'JanMay2024': 'Jan–May 2024',
}
TL_COLORS = {
    'JunAug2023': '#3498DB',
    'SepDec2023': '#F39C12',
    'JanMay2024': '#2ECC71',
}

# ── PLATFORM COLOURS ─────────────────────────────────────────────────────────
PLAT_COLORS = {'Facebook': '#4267B2', 'Instagram': '#E1306C'}
PLAT_ORDER  = ['Facebook', 'Instagram']

# ── HASHTAG SENTIMENT COLOURS ────────────────────────────────────────────────
HSENT_COLORS = {'Positive': '#2ECC71', 'Neutral': '#95A5A6', 'Negative': '#E74C3C', 'NoHashtags': '#BDC3C7'}
HSENT_ORDER  = ['Positive', 'Neutral', 'Negative']


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART STYLE SETUP
# ═══════════════════════════════════════════════════════════════════════════════
def setup_style():
    """Apply a clean, professional chart style across all plots."""
    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Segoe UI', 'DejaVu Sans', 'Helvetica', 'Arial'],
        'font.size':         10,
        'axes.titlesize':    13,
        'axes.titleweight':  'bold',
        'axes.labelsize':    11,
        'axes.labelweight':  'normal',
        'xtick.labelsize':   9,
        'ytick.labelsize':   9,
        'legend.fontsize':   9,
        'figure.dpi':        150,
        'savefig.dpi':       200,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.4,
        'figure.facecolor':  '#ffffff',
        'axes.facecolor':    '#f8f9fa',
        'axes.edgecolor':    '#dee2e6',
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.color':        '#dee2e6',
        'grid.linestyle':    '--',
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    """
    Returns
    -------
    hc : DataFrame
        All high-confidence rows (needs_review=False).
        21,909 rows.  Has sentiment_label for every row.
        Use for: caption sentiment, hashtag sentiment, engagement, metadata.

    hc_clf : DataFrame
        High-confidence rows EXCLUDING _insufficient=True / SKIPPED.
        19,790 rows.
        Use for: tone_primary, emo_primary, subj_label analyses.
    """
    print("  Loading dataset...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    hc = df[df['needs_review'] == False].copy()
    hc['_alliance'] = hc['_account'].map(ALLIANCE_MAP)

    hc_clf = hc[hc['_insufficient'] == False].copy()
    print(f"  → {len(hc):,} high-confidence rows loaded.")
    print(f"  → {len(hc_clf):,} classified rows (excl. SKIPPED/insufficient).")
    return hc, hc_clf


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def pct(n, total):
    """Safe percentage calculation."""
    return (n / total * 100) if total > 0 else 0.0


def add_value_labels(ax, bars, fmt='{:.1f}%', fontsize=8, counts=None, vertical_offset=None):
    """
    Add value labels on top of bars.
    If counts is provided, shows 'value\\n(n=count)' format.
    """
    ymin, ymax = ax.get_ylim()
    if vertical_offset is None:
        vertical_offset = (ymax - ymin) * 0.015
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if h > 0:
            label = fmt.format(h)
            if counts is not None:
                label += f'\n(n={counts[i]:,})'
            ax.text(
                bar.get_x() + bar.get_width() / 2., h + vertical_offset,
                label, ha='center', va='bottom',
                fontsize=fontsize, fontweight='bold', color='#333333'
            )


def add_value_labels_inside(ax, bars, fmt='{:.1f}%', fontsize=8, counts=None):
    """Add value labels inside bars (for stacked or shorter bars)."""
    for i, bar in enumerate(bars):
        h = bar.get_height()
        if h > 2:  # only label if tall enough
            label = fmt.format(h)
            if counts is not None:
                label += f'\n(n={counts[i]:,})'
            ax.text(
                bar.get_x() + bar.get_width() / 2., bar.get_y() + h / 2.,
                label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white'
            )


def add_methodology(fig, text, y=0.01, fontsize=7.5):
    """Add methodology annotation at the bottom of the figure."""
    fig.text(
        0.5, y, f'Methodology: {text}',
        ha='center', va='bottom', fontsize=fontsize,
        color='#888888', style='italic',
        transform=fig.transFigure, wrap=True,
        linespacing=1.4,
    )


def save_chart(fig, filename):
    """Save chart to the charts/ output directory."""
    path = os.path.join(CHART_DIR, filename)
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"    ✓ Saved: {filename}")
    return path


def format_title(chart_id, title, subtitle=None):
    """Format a chart title with optional subtitle."""
    if subtitle:
        return f"{chart_id}: {title}\n({subtitle})"
    return f"{chart_id}: {title}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT INSIGHT REPORT WRITER
# ═══════════════════════════════════════════════════════════════════════════════

def write_insight_report(filename, report_title, sections):
    """
    Write a structured text insight report.

    Parameters
    ----------
    filename : str
        Output filename (saved to TEXT_DIR).
    report_title : str
        Title of the report.
    sections : list of tuples
        Each tuple: (section_title, insights_list)
        Each insight in insights_list: (insight_id, insight_title, finding_text, methodology_text)
    """
    path = os.path.join(TEXT_DIR, filename)
    sep  = '=' * 88
    sep2 = '─' * 88

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{sep}\n')
        f.write(f'  {report_title}\n')
        f.write(f'  AP Political Sentiment Analysis — Research-Grade Insights Report\n')
        f.write(f'  Dataset: phase3_classified_patched.csv\n')
        f.write(f'  Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'{sep}\n\n')

        for sec_title, insights in sections:
            f.write(f'{sep2}\n')
            f.write(f'  SECTION: {sec_title}\n')
            f.write(f'{sep2}\n\n')

            for insight_id, insight_title, finding, methodology in insights:
                f.write(f'  {insight_id}: {insight_title}\n')
                f.write(f'  {"─" * (len(insight_id) + len(insight_title) + 2)}\n\n')
                # Wrap finding text
                for line in finding.split('\n'):
                    f.write(f'  {line}\n')
                f.write(f'\n')
                f.write(f'  Methodology: {methodology}\n')
                f.write(f'\n{"─" * 40}\n\n')

            f.write('\n')

        f.write(f'{sep}\n')
        f.write(f'  END OF REPORT\n')
        f.write(f'{sep}\n')

    print(f"    ✓ Report saved: {filename}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED COMPUTATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_distribution(series, order=None):
    """
    Compute distribution counts and percentages for a categorical series.
    Returns dict: {label: {'count': int, 'pct': float}}
    """
    vc = series.value_counts()
    total = vc.sum()
    result = {}
    labels = order if order else vc.index.tolist()
    for label in labels:
        c = vc.get(label, 0)
        result[label] = {'count': int(c), 'pct': pct(c, total)}
    return result, int(total)


def compute_grouped_distribution(df, group_col, value_col, value_order=None):
    """
    Compute distribution of value_col for each group in group_col.
    Returns dict: {group: {label: {'count': int, 'pct': float}, ...}, ...}
    """
    result = {}
    for grp, sub_df in df.groupby(group_col):
        vc = sub_df[value_col].value_counts()
        total = vc.sum()
        labels = value_order if value_order else vc.index.tolist()
        grp_data = {}
        for label in labels:
            c = vc.get(label, 0)
            grp_data[label] = {'count': int(c), 'pct': pct(c, total)}
        grp_data['_total'] = int(total)
        result[grp] = grp_data
    return result


def compute_mean_engagement(df, group_col, metric='Engagement_Rate_Monthly'):
    """Compute mean engagement metric for each group."""
    result = {}
    for grp, sub_df in df.groupby(group_col):
        result[grp] = {
            'mean': sub_df[metric].mean(),
            'count': len(sub_df),
        }
    return result

# ============================================================
#  SCRIPT 05 — SUBJECTIVITY ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: subj_label (Fact / Mixed / Opinion)
#  Supporting: subj_conf (confidence score)
#  Subset: needs_review=False AND _insufficient=False → 19,790 classified rows
#
#  Run: python insight_05_subjectivity.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 05: SUBJECTIVITY ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_clf = len(hc_clf)

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB1: Overall Subjectivity Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    dist, _ = compute_distribution(hc_clf['subj_label'], SUBJ_ORDER)
    labels = SUBJ_ORDER
    counts = [dist[l]['count'] for l in labels]
    pcts   = [dist[l]['pct'] for l in labels]
    colors = [SUBJ_COLORS[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle(format_title('SB1', 'Overall Subjectivity Distribution',
                              f'n = {n_clf:,} classified posts'),
                 fontsize=14, fontweight='bold', y=1.02)

    bars = ax1.bar(labels, pcts, color=colors, edgecolor='white', linewidth=0.8, width=0.55)
    add_value_labels(ax1, bars, fmt='{:.1f}%', counts=counts, fontsize=10)
    ax1.set_ylabel('Percentage of Posts (%)')
    ax1.set_xlabel('Subjectivity Label')
    ax1.set_ylim(0, max(pcts) * 1.25)
    ax1.set_title('Distribution (Bar)', fontsize=11)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax2.set_title('Distribution (Pie)', fontsize=11)

    add_methodology(fig,
        f'Source: subj_label column (Fact/Mixed/Opinion classifier). '
        f'Subset: classified posts only (n={n_clf:,}). '
        f'Fact = primarily fact-based assertions. Opinion = subjective/opinion-based. '
        f'Mixed = blend of factual and opinion content. '
        f'Percentages = count ÷ {n_clf:,} × 100.',
        y=-0.04)
    save_chart(fig, 'SB1_overall_subjectivity.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB2: Subjectivity by Account
    # ═════════════════════════════════════════════════════════════════════════
    grp = compute_grouped_distribution(hc_clf, '_account', 'subj_label', SUBJ_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('SB2', 'Subjectivity Distribution by Account',
                              f'Fact vs Opinion vs Mixed per account'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        acc_data = grp.get(acc, {})
        acc_total = acc_data.get('_total', 0)
        vals = [acc_data.get(s, {}).get('pct', 0) for s in SUBJ_ORDER]
        cnts = [acc_data.get(s, {}).get('count', 0) for s in SUBJ_ORDER]
        bars = ax.bar(SUBJ_ORDER, vals, color=[SUBJ_COLORS[s] for s in SUBJ_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={acc_total:,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    add_methodology(fig,
        f'Source: subj_label × _account. Classified posts only. '
        f'Percentages = count per subj_label ÷ total classified posts for that account × 100.',
        y=-0.06)
    save_chart(fig, 'SB2_subjectivity_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB3: Subjectivity by Alliance
    # ═════════════════════════════════════════════════════════════════════════
    grp_al = compute_grouped_distribution(hc_clf, '_alliance', 'subj_label', SUBJ_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(format_title('SB3', 'Subjectivity by Alliance',
                              f'YSRCP Alliance vs TDP Alliance'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        al_data  = grp_al.get(al, {})
        al_total = al_data.get('_total', 0)
        vals = [al_data.get(s, {}).get('pct', 0) for s in SUBJ_ORDER]
        cnts = [al_data.get(s, {}).get('count', 0) for s in SUBJ_ORDER]
        bars = ax.bar(SUBJ_ORDER, vals, color=[SUBJ_COLORS[s] for s in SUBJ_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=10)
        ax.set_title(f'{al}\n(n={al_total:,})', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 50)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    add_methodology(fig,
        f'Source: subj_label × _alliance. Alliance: YSRCP+YS Jagan vs TDP+NCBN.',
        y=-0.04)
    save_chart(fig, 'SB3_subjectivity_by_alliance.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB4: Subjectivity by Timeline
    # ═════════════════════════════════════════════════════════════════════════
    grp_tl = compute_grouped_distribution(hc_clf, '_timeline', 'subj_label', SUBJ_ORDER)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('SB4', 'Subjectivity Evolution Across Timelines',
                              f'Is discourse becoming more fact-based or opinion-based over time?'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    width = 0.25
    for j, subj in enumerate(SUBJ_ORDER):
        vals = [grp_tl.get(tl, {}).get(subj, {}).get('pct', 0) for tl in TL_ORDER]
        cnts = [grp_tl.get(tl, {}).get(subj, {}).get('count', 0) for tl in TL_ORDER]
        bars = ax.bar(x + j * width, vals, width, label=subj,
                      color=SUBJ_COLORS[subj], edgecolor='white', linewidth=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{TL_LABELS[tl]}\n(n={grp_tl.get(tl, {}).get("_total", 0):,})'
                        for tl in TL_ORDER])
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 55)
    ax.legend(title='Subjectivity', framealpha=0.9)

    add_methodology(fig,
        f'Source: subj_label × _timeline. Classified posts only. '
        f'Percentages = count per subj_label per timeline ÷ total classified posts in that timeline × 100.',
        y=-0.04)
    save_chart(fig, 'SB4_subjectivity_by_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB5: Subjectivity × Sentiment Cross-Tab (Heatmap)
    # ═════════════════════════════════════════════════════════════════════════
    ct = pd.crosstab(hc_clf['subj_label'], hc_clf['sentiment_label'], normalize='index') * 100
    ct = ct.reindex(index=SUBJ_ORDER, columns=SENT_ORDER, fill_value=0)
    ct_counts = pd.crosstab(hc_clf['subj_label'], hc_clf['sentiment_label'])
    ct_counts = ct_counts.reindex(index=SUBJ_ORDER, columns=SENT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('SB5', 'Subjectivity × Caption Sentiment Cross-Tab',
                              f'Do fact-based posts lean more negative? Are opinions more positive?'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(SENT_ORDER)))
    ax.set_xticklabels(SENT_ORDER)
    ax.set_yticks(range(len(SUBJ_ORDER)))
    ax.set_yticklabels(SUBJ_ORDER)
    ax.set_xlabel('Caption Sentiment')
    ax.set_ylabel('Subjectivity')

    for r in range(len(SUBJ_ORDER)):
        for c in range(len(SENT_ORDER)):
            val = ct.values[r, c]
            cnt = ct_counts.values[r, c]
            color = 'white' if val > 50 else 'black'
            ax.text(c, r, f'{val:.1f}%\n(n={cnt:,})', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row-normalised %')

    add_methodology(fig,
        f'Source: cross-tab of subj_label × sentiment_label, row-normalised. '
        f'Cell = % of that sentiment within that subjectivity category. '
        f'Each row sums to 100%. n-counts = raw counts in parentheses.',
        y=-0.04)
    save_chart(fig, 'SB5_subjectivity_sentiment_crosstab.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB6: Subjectivity × Tone Cross-Tab (Heatmap)
    # ═════════════════════════════════════════════════════════════════════════
    hc_tone_valid = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]
    ct2 = pd.crosstab(hc_tone_valid['subj_label'], hc_tone_valid['tone_primary'],
                       normalize='index') * 100
    ct2 = ct2.reindex(index=SUBJ_ORDER, columns=TONE_ORDER, fill_value=0)
    ct2_counts = pd.crosstab(hc_tone_valid['subj_label'], hc_tone_valid['tone_primary'])
    ct2_counts = ct2_counts.reindex(index=SUBJ_ORDER, columns=TONE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('SB6', 'Subjectivity × Rhetorical Tone Cross-Tab',
                              f'Are aggressive posts more opinion-based? Are facts used persuasively?'),
                 fontsize=14, fontweight='bold', y=1.02)

    tone_display = [t.replace('_', ' ') for t in TONE_ORDER]
    im = ax.imshow(ct2.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(TONE_ORDER)))
    ax.set_xticklabels(tone_display, rotation=15)
    ax.set_yticks(range(len(SUBJ_ORDER)))
    ax.set_yticklabels(SUBJ_ORDER)
    ax.set_xlabel('Rhetorical Tone')
    ax.set_ylabel('Subjectivity')

    for r in range(len(SUBJ_ORDER)):
        for c in range(len(TONE_ORDER)):
            val = ct2.values[r, c]
            cnt = ct2_counts.values[r, c]
            color = 'white' if val > ct2.values.max() * 0.6 else 'black'
            ax.text(c, r, f'{val:.1f}%\n(n={cnt:,})', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row-normalised %')

    add_methodology(fig,
        f'Source: cross-tab of subj_label × tone_primary, row-normalised. '
        f'Neutral-tone excluded. Cell = % of that tone within that subjectivity category. '
        f'Each row sums to 100%.',
        y=-0.04)
    save_chart(fig, 'SB6_subjectivity_tone_crosstab.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB7: Subjectivity by Platform
    # ═════════════════════════════════════════════════════════════════════════
    grp_pl = compute_grouped_distribution(hc_clf, '_platform', 'subj_label', SUBJ_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(format_title('SB7', 'Subjectivity by Platform',
                              f'Facebook vs Instagram content basis'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, plat in enumerate(PLAT_ORDER):
        ax = axes[i]
        pl_data  = grp_pl.get(plat, {})
        pl_total = pl_data.get('_total', 0)
        vals = [pl_data.get(s, {}).get('pct', 0) for s in SUBJ_ORDER]
        cnts = [pl_data.get(s, {}).get('count', 0) for s in SUBJ_ORDER]
        bars = ax.bar(SUBJ_ORDER, vals, color=[SUBJ_COLORS[s] for s in SUBJ_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=10)
        ax.set_title(f'{plat}\n(n={pl_total:,})', fontsize=12,
                     color=PLAT_COLORS[plat], fontweight='bold')
        ax.set_ylim(0, 50)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    add_methodology(fig,
        f'Source: subj_label × _platform. Classified posts only. '
        f'Percentages = count per subj ÷ total classified on that platform × 100.',
        y=-0.04)
    save_chart(fig, 'SB7_subjectivity_by_platform.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB8: Subjectivity Confidence Distribution
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('SB8', 'Subjectivity Classifier Confidence by Label',
                              f'Mean confidence for Fact vs Opinion vs Mixed'),
                 fontsize=14, fontweight='bold', y=1.02)

    conf_vals = []
    conf_cnts = []
    for s in SUBJ_ORDER:
        sub = hc_clf[hc_clf['subj_label'] == s]['subj_conf']
        conf_vals.append(sub.mean())
        conf_cnts.append(len(sub))

    bars = ax.bar(SUBJ_ORDER, conf_vals, color=[SUBJ_COLORS[s] for s in SUBJ_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.3f}', counts=conf_cnts, fontsize=10)
    ax.set_ylabel('Mean Confidence Score')
    ax.set_ylim(0, max(conf_vals) * 1.2 if conf_vals else 1)

    add_methodology(fig,
        f'Source: subj_conf column grouped by subj_label. '
        f'Value = mean(subj_conf) per label. Higher = more certain classification.',
        y=-0.04)
    save_chart(fig, 'SB8_subjectivity_confidence.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SB9: Fact-to-Opinion Ratio by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('SB9', 'Fact-to-Opinion Ratio by Account',
                              f'Ratio > 1 = more fact-based posts; < 1 = more opinion-based'),
                 fontsize=14, fontweight='bold', y=1.02)

    fo_ratios = []
    fo_labels = []
    for acc in ACC_ORDER:
        d = grp.get(acc, {})
        n_fact = d.get('Fact', {}).get('count', 0)
        n_opin = d.get('Opinion', {}).get('count', 0)
        ratio = n_fact / n_opin if n_opin > 0 else 0
        fo_ratios.append(ratio)
        fo_labels.append(f'{ACC_LABELS[acc]}\n(F={n_fact:,}, O={n_opin:,})')

    bars = ax.bar(fo_labels, fo_ratios, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.2f}', fontsize=11)
    ax.axhline(y=1.0, color='#888888', linestyle='--', linewidth=1, alpha=0.7,
               label='Ratio = 1.0 (balanced)')
    ax.set_ylabel('Fact / Opinion Ratio')
    ax.set_ylim(0, max(fo_ratios) * 1.35 if fo_ratios else 2)
    ax.legend(fontsize=9)

    add_methodology(fig,
        f'Source: subj_label column. Ratio = count(Fact) ÷ count(Opinion) per account. '
        f'Mixed posts excluded from this ratio. '
        f'> 1.0 = more fact-based; < 1.0 = more opinion-based.',
        y=-0.04)
    save_chart(fig, 'SB9_fact_opinion_ratio_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    overall_subj = compute_distribution(hc_clf['subj_label'], SUBJ_ORDER)[0]
    by_acc_subj  = compute_grouped_distribution(hc_clf, '_account', 'subj_label', SUBJ_ORDER)
    by_al_subj   = compute_grouped_distribution(hc_clf, '_alliance', 'subj_label', SUBJ_ORDER)
    by_tl_subj   = compute_grouped_distribution(hc_clf, '_timeline', 'subj_label', SUBJ_ORDER)

    # Subj × Sentiment
    subj_sent = {}
    for subj in SUBJ_ORDER:
        sub = hc_clf[hc_clf['subj_label'] == subj]
        total = len(sub)
        subj_sent[subj] = {s: pct((sub['sentiment_label']==s).sum(), total) for s in SENT_ORDER}

    # Subj × Tone
    subj_tone = {}
    for subj in SUBJ_ORDER:
        sub = hc_tone_valid[hc_tone_valid['subj_label'] == subj]
        total = len(sub)
        subj_tone[subj] = {t: pct((sub['tone_primary']==t).sum(), total) for t in TONE_ORDER}

    sec_insights = []

    # 5.1 Overall
    sec_insights.append((
        'INSIGHT 5.1',
        'Overall Subjectivity Distribution',
        f'Among {n_clf:,} classified posts:\n\n'
        + '\n'.join([f'    • {s}: {overall_subj[s]["count"]:,} ({overall_subj[s]["pct"]:.1f}%)'
                     for s in SUBJ_ORDER]) + '\n\n'
        f'  Mixed content is the most prevalent category, indicating that AP political posts '
        f'frequently blend factual claims with opinion-based commentary. Pure opinion posts '
        f'outnumber pure fact-based posts, suggesting the discourse leans more toward '
        f'subjective interpretation than objective reporting.',
        f'Counts of subj_label per label ÷ {n_clf:,} × 100. '
        f'Fact = primarily factual assertions. Opinion = subjective commentary. '
        f'Mixed = blend of both.'
    ))

    # 5.2 By account
    acc_subj_lines = []
    for acc in ACC_ORDER:
        d = by_acc_subj.get(acc, {})
        acc_subj_lines.append(
            f'    • {ACC_LABELS[acc]} (n={d.get("_total", 0):,}): '
            f'Fact={d.get("Fact", {}).get("pct", 0):.1f}%, '
            f'Mixed={d.get("Mixed", {}).get("pct", 0):.1f}%, '
            f'Opinion={d.get("Opinion", {}).get("pct", 0):.1f}%'
        )

    most_fact = max(ACC_ORDER, key=lambda a: by_acc_subj.get(a, {}).get('Fact', {}).get('pct', 0))
    most_opin = max(ACC_ORDER, key=lambda a: by_acc_subj.get(a, {}).get('Opinion', {}).get('pct', 0))

    sec_insights.append((
        'INSIGHT 5.2',
        'Subjectivity Breakdown by Account',
        f'Per-account subjectivity profiles:\n\n'
        + '\n'.join(acc_subj_lines) + '\n\n'
        f'  {ACC_LABELS[most_fact]} has the highest fact-based proportion at '
        f'{by_acc_subj.get(most_fact, {}).get("Fact", {}).get("pct", 0):.1f}%, while '
        f'{ACC_LABELS[most_opin]} has the highest opinion proportion at '
        f'{by_acc_subj.get(most_opin, {}).get("Opinion", {}).get("pct", 0):.1f}%. '
        f'This reveals fundamentally different content strategies: some accounts position '
        f'themselves as information sources, others as opinion leaders.',
        f'subj_label counts per _account ÷ total classified posts per account × 100.'
    ))

    # 5.3 Subjectivity × Sentiment
    sec_insights.append((
        'INSIGHT 5.3',
        'Do Fact-Based Posts Lean More Negative?',
        f'Sentiment distribution within each subjectivity category:\n\n'
        + '\n'.join([f'    • {subj}: Pos={subj_sent[subj]["Positive"]:.1f}%, '
                     f'Neu={subj_sent[subj]["Neutral"]:.1f}%, '
                     f'Neg={subj_sent[subj]["Negative"]:.1f}%'
                     for subj in SUBJ_ORDER]) + '\n\n'
        f'  This cross-analysis reveals whether facts are used more for negative framing '
        f'(e.g., citing statistics about governance failures) or positive framing '
        f'(e.g., citing development achievements). Similarly, it shows whether opinion '
        f'content skews positive (cheerleading) or negative (criticism).',
        f'Cross-tab of subj_label × sentiment_label, row-normalised (each row sums to 100%). '
        f'Percentages = count of posts with that subjectivity+sentiment pair ÷ '
        f'total posts with that subjectivity × 100.'
    ))

    # 5.4 Subjectivity × Tone
    sec_insights.append((
        'INSIGHT 5.4',
        'Subjectivity × Rhetorical Tone Relationship',
        f'Tone distribution within each subjectivity category:\n\n'
        + '\n'.join([f'    • {subj}: Agg={subj_tone[subj].get("Aggressive", 0):.1f}%, '
                     f'Per={subj_tone[subj].get("Persuasive", 0):.1f}%, '
                     f'Ins={subj_tone[subj].get("Inspirational", 0):.1f}%, '
                     f'VN={subj_tone[subj].get("Victim_Narrative", 0):.1f}%'
                     for subj in SUBJ_ORDER]) + '\n\n'
        f'  This reveals the rhetorical function of facts vs opinions: e.g., are facts '
        f'primarily wielded aggressively (as weapons in political attacks) or persuasively '
        f'(as evidence for policy proposals)?',
        f'Cross-tab of subj_label × tone_primary, row-normalised. Neutral-tone excluded. '
        f'Percentages = count of posts with subj+tone pair ÷ total posts with that subjectivity × 100.'
    ))

    # 5.5 Alliance comparison
    ysrcp_subj = by_al_subj.get('YSRCP Alliance', {})
    tdp_subj   = by_al_subj.get('TDP Alliance', {})

    sec_insights.append((
        'INSIGHT 5.5',
        'Alliance Subjectivity Comparison',
        f'YSRCP Alliance (n={ysrcp_subj.get("_total", 0):,}):\n'
        + '\n'.join([f'    {s}: {ysrcp_subj.get(s, {}).get("pct", 0):.1f}%' for s in SUBJ_ORDER])
        + f'\n\n  TDP Alliance (n={tdp_subj.get("_total", 0):,}):\n'
        + '\n'.join([f'    {s}: {tdp_subj.get(s, {}).get("pct", 0):.1f}%' for s in SUBJ_ORDER])
        + f'\n\n  The alliance with a higher Fact percentage may be leveraging '
        f'data and evidence more in its messaging, while a higher Opinion percentage '
        f'suggests more rhetorical and persuasion-based communication.',
        f'subj_label counts per _alliance ÷ total classified posts per alliance × 100.'
    ))

    # 5.6 Timeline trend
    tl_subj_lines = []
    for tl in TL_ORDER:
        d = by_tl_subj.get(tl, {})
        tl_subj_lines.append(
            f'    • {TL_LABELS[tl]}: Fact={d.get("Fact", {}).get("pct", 0):.1f}%, '
            f'Mixed={d.get("Mixed", {}).get("pct", 0):.1f}%, '
            f'Opinion={d.get("Opinion", {}).get("pct", 0):.1f}%'
        )

    first_fact = by_tl_subj.get(TL_ORDER[0], {}).get('Fact', {}).get('pct', 0)
    last_fact  = by_tl_subj.get(TL_ORDER[-1], {}).get('Fact', {}).get('pct', 0)

    sec_insights.append((
        'INSIGHT 5.6',
        'Subjectivity Temporal Trends',
        f'Subjectivity evolution:\n\n'
        + '\n'.join(tl_subj_lines) + '\n\n'
        f'  Fact-based content shifts from {first_fact:.1f}% ({TL_LABELS[TL_ORDER[0]]}) '
        f'to {last_fact:.1f}% ({TL_LABELS[TL_ORDER[-1]]}), a change of {last_fact - first_fact:+.1f}pp. '
        f'This may reflect changing campaign strategies as elections approach — '
        f'parties may rely more on data/facts (governance track record) or shift to '
        f'more emotional/opinion-based appeals for voter mobilisation.',
        f'subj_label counts per _timeline ÷ total classified posts per timeline × 100.'
    ))

    # 5.7 Fact-to-Opinion ratio
    fo_ratio_lines = []
    for acc in ACC_ORDER:
        d = by_acc_subj.get(acc, {})
        n_f = d.get('Fact', {}).get('count', 0)
        n_o = d.get('Opinion', {}).get('count', 0)
        r = n_f / n_o if n_o > 0 else float('inf')
        fo_ratio_lines.append(f'    • {ACC_LABELS[acc]}: {r:.2f} (Fact={n_f:,}, Opinion={n_o:,})')

    sec_insights.append((
        'INSIGHT 5.7',
        'Fact-to-Opinion Ratio per Account',
        f'F/O Ratio (>1 = more fact-based; <1 = more opinion-based):\n\n'
        + '\n'.join(fo_ratio_lines) + '\n\n'
        f'  This single metric captures the "information vs persuasion" balance of each '
        f'account. A higher ratio suggests a more evidence-driven communication approach; '
        f'a lower ratio suggests a more rhetoric-driven approach.',
        f'Ratio = count(Fact) ÷ count(Opinion) per account. Mixed posts excluded.'
    ))

    # 5.8 Confidence analysis
    subj_conf = {s: hc_clf[hc_clf['subj_label']==s]['subj_conf'].mean() for s in SUBJ_ORDER}
    most_conf_subj = max(SUBJ_ORDER, key=lambda s: subj_conf[s])

    sec_insights.append((
        'INSIGHT 5.8',
        'Subjectivity Classifier Confidence',
        f'Mean classifier confidence by subjectivity label:\n\n'
        + '\n'.join([f'    • {s}: {subj_conf[s]:.3f}' for s in SUBJ_ORDER]) + '\n\n'
        f'  {most_conf_subj} posts have the highest mean confidence ({subj_conf[most_conf_subj]:.3f}), '
        f'indicating the model finds the clearest linguistic signals for that category. '
        f'Mixed posts typically have lower confidence as they contain signals for both '
        f'fact and opinion, creating classifier ambiguity.',
        f'Mean of subj_conf per subj_label. Higher = more certain classification.'
    ))

    # 5.9 Conclusion
    sec_insights.append((
        'INSIGHT 5.9',
        'Conclusion — Subjectivity Landscape',
        f'The subjectivity landscape of AP political social media reveals:\n\n'
        f'  1. Mixed content dominates at {overall_subj["Mixed"]["pct"]:.1f}%, reflecting the '
        f'inherently blended nature of political messaging.\n'
        f'  2. Opinion content ({overall_subj["Opinion"]["pct"]:.1f}%) exceeds pure fact content '
        f'({overall_subj["Fact"]["pct"]:.1f}%), indicating a discourse leaning toward subjective commentary.\n'
        f'  3. {ACC_LABELS[most_fact]} is the most fact-oriented account.\n'
        f'  4. Fact-based posts show distinct sentiment and tone profiles compared to opinion-based posts.\n'
        f'  5. The relationship between subjectivity, sentiment, and tone reveals how political '
        f'actors strategically deploy facts and opinions in their communication.\n\n'
        f'  These findings inform understanding of political information quality and the '
        f'fact-vs-opinion balance that characterises AP political social media discourse.',
        f'Synthesis of insights 5.1–5.8.'
    ))

    all_sections.append(('Subjectivity Analysis', sec_insights))

    write_insight_report('05_subjectivity_insights.txt',
                         'SUBJECTIVITY ANALYSIS', all_sections)

    print(f"\n  Done! Generated 9 charts + 1 text report (9 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

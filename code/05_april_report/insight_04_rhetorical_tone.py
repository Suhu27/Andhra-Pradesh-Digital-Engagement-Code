# ============================================================
#  SCRIPT 04 — RHETORICAL TONE ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: tone_primary (Aggressive / Persuasive / Inspirational /
#            Victim_Narrative / Defensive)
#  Supporting: tone_primary_c (confidence)
#  Subset: needs_review=False AND _insufficient=False → 19,790 classified rows
#
#  Run: python insight_04_rhetorical_tone.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 04: RHETORICAL TONE ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_clf = len(hc_clf)

    # Exclude the rare 'Neutral' tone (only 12 posts) from main analysis
    # but note it in text insights
    neutral_tone_count = (hc_clf['tone_primary'] == 'Neutral').sum()
    hc_tone = hc_clf[hc_clf['tone_primary'] != 'Neutral'].copy()
    n_tone = len(hc_tone)

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T1: Overall Tone Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    dist, _ = compute_distribution(hc_tone['tone_primary'], TONE_ORDER)
    labels = TONE_ORDER
    counts = [dist[l]['count'] for l in labels]
    pcts   = [dist[l]['pct'] for l in labels]
    colors = [TONE_COLORS[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1.3, 1]})
    fig.suptitle(format_title('T1', 'Overall Rhetorical Tone Distribution',
                              f'n = {n_tone:,} classified posts (excl. {neutral_tone_count:,} Neutral-tone posts)'),
                 fontsize=14, fontweight='bold', y=1.02)

    tone_display = [t.replace('_', ' ') for t in labels]

    bars = ax1.bar(tone_display, pcts, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    add_value_labels(ax1, bars, fmt='{:.1f}%', counts=counts, fontsize=9)
    ax1.set_ylabel('Percentage of Posts (%)')
    ax1.set_xlabel('Rhetorical Tone')
    ax1.set_ylim(0, max(pcts) * 1.25)
    ax1.set_title('Distribution (Bar)', fontsize=11)
    ax1.tick_params(axis='x', rotation=15)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=tone_display, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax2.set_title('Distribution (Pie)', fontsize=11)

    add_methodology(fig,
        f'Source: tone_primary column (rhetorical tone classifier). '
        f'Subset: classified posts (n={n_tone:,}). {neutral_tone_count:,} "Neutral"-tone posts excluded '
        f'(too few for meaningful analysis). '
        f'Percentages = count ÷ {n_tone:,} × 100.',
        y=-0.06)
    save_chart(fig, 'T1_overall_tone_distribution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T2: Tone by Account
    # ═════════════════════════════════════════════════════════════════════════
    grp = compute_grouped_distribution(hc_tone, '_account', 'tone_primary', TONE_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('T2', 'Rhetorical Tone Distribution by Account',
                              f'Each account\'s rhetorical strategy profile'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        acc_data = grp.get(acc, {})
        acc_total = acc_data.get('_total', 0)
        vals = [acc_data.get(t, {}).get('pct', 0) for t in TONE_ORDER]
        cnts = [acc_data.get(t, {}).get('count', 0) for t in TONE_ORDER]
        bars = ax.bar(tone_display, vals, color=[TONE_COLORS[t] for t in TONE_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.6)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={acc_total:,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 65)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=25, labelsize=8)

    add_methodology(fig,
        f'Source: tone_primary × _account. Neutral-tone posts excluded. '
        f'Percentages = count per tone ÷ total classified posts (excl. Neutral) for that account × 100.',
        y=-0.12)
    save_chart(fig, 'T2_tone_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T3: Tone by Alliance
    # ═════════════════════════════════════════════════════════════════════════
    grp_al = compute_grouped_distribution(hc_tone, '_alliance', 'tone_primary', TONE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(format_title('T3', 'Rhetorical Tone by Alliance',
                              f'YSRCP Alliance vs TDP Alliance'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        al_data  = grp_al.get(al, {})
        al_total = al_data.get('_total', 0)
        vals = [al_data.get(t, {}).get('pct', 0) for t in TONE_ORDER]
        cnts = [al_data.get(t, {}).get('count', 0) for t in TONE_ORDER]
        bars = ax.bar(tone_display, vals, color=[TONE_COLORS[t] for t in TONE_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=9)
        ax.set_title(f'{al}\n(n={al_total:,})', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=15)

    add_methodology(fig,
        f'Source: tone_primary × _alliance. Alliance: YSRCP+YS Jagan vs TDP+NCBN. '
        f'Neutral-tone posts excluded. '
        f'Percentages = count per tone ÷ total classified posts (excl. Neutral) per alliance × 100.',
        y=-0.06)
    save_chart(fig, 'T3_tone_by_alliance.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T4: Tone Evolution by Timeline
    # ═════════════════════════════════════════════════════════════════════════
    grp_tl = compute_grouped_distribution(hc_tone, '_timeline', 'tone_primary', TONE_ORDER)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('T4', 'Rhetorical Tone Evolution Across Timelines',
                              f'How the political rhetoric shifts over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    n_tones = len(TONE_ORDER)
    width = 0.14
    offset = -(n_tones - 1) * width / 2

    for j, tone in enumerate(TONE_ORDER):
        vals = [grp_tl.get(tl, {}).get(tone, {}).get('pct', 0) for tl in TL_ORDER]
        cnts = [grp_tl.get(tl, {}).get(tone, {}).get('count', 0) for tl in TL_ORDER]
        bars = ax.bar(x + offset + j * width, vals, width,
                      label=tone.replace('_', ' '),
                      color=TONE_COLORS[tone], edgecolor='white', linewidth=0.3)
        add_value_labels(ax, bars, fmt='{:.1f}%', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{TL_LABELS[tl]}\n(n={grp_tl.get(tl, {}).get("_total", 0):,})'
                        for tl in TL_ORDER])
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 55)
    ax.legend(title='Tone', ncol=3, fontsize=8, title_fontsize=9, framealpha=0.9)

    add_methodology(fig,
        f'Source: tone_primary × _timeline. Neutral-tone posts excluded. '
        f'Percentages = count per tone per timeline ÷ total classified posts (excl. Neutral) in that timeline × 100.',
        y=-0.04)
    save_chart(fig, 'T4_tone_by_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T5: Tone by Platform
    # ═════════════════════════════════════════════════════════════════════════
    grp_pl = compute_grouped_distribution(hc_tone, '_platform', 'tone_primary', TONE_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(format_title('T5', 'Rhetorical Tone by Platform',
                              f'Facebook vs Instagram: Do platforms differ in rhetorical approach?'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, plat in enumerate(PLAT_ORDER):
        ax = axes[i]
        pl_data  = grp_pl.get(plat, {})
        pl_total = pl_data.get('_total', 0)
        vals = [pl_data.get(t, {}).get('pct', 0) for t in TONE_ORDER]
        cnts = [pl_data.get(t, {}).get('count', 0) for t in TONE_ORDER]
        bars = ax.bar(tone_display, vals, color=[TONE_COLORS[t] for t in TONE_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{plat}\n(n={pl_total:,})', fontsize=12,
                     color=PLAT_COLORS[plat], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=15)

    add_methodology(fig,
        f'Source: tone_primary × _platform. Neutral-tone posts excluded. '
        f'Percentages = count per tone per platform ÷ total classified posts (excl. Neutral) on that platform × 100.',
        y=-0.06)
    save_chart(fig, 'T5_tone_by_platform.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T6: Aggressive-to-Persuasive Ratio by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('T6', 'Aggressive-to-Persuasive Ratio by Account',
                              f'Ratio > 1 = more aggressive than persuasive posts'),
                 fontsize=14, fontweight='bold', y=1.02)

    ap_ratios = []
    ap_labels = []
    for acc in ACC_ORDER:
        sub = hc_tone[hc_tone['_account'] == acc]
        n_agg  = (sub['tone_primary'] == 'Aggressive').sum()
        n_pers = (sub['tone_primary'] == 'Persuasive').sum()
        ratio = n_agg / n_pers if n_pers > 0 else 0
        ap_ratios.append(ratio)
        ap_labels.append(f'{ACC_LABELS[acc]}\n(Agg={n_agg:,}, Per={n_pers:,})')

    bars = ax.bar(ap_labels, ap_ratios, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.2f}', fontsize=11)
    ax.axhline(y=1.0, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.7,
               label='Ratio = 1.0 (balanced)')
    ax.set_ylabel('Aggressive / Persuasive Ratio')
    ax.set_ylim(0, max(ap_ratios) * 1.35 if ap_ratios else 3)
    ax.legend(fontsize=9)

    add_methodology(fig,
        f'Source: tone_primary column. '
        f'Ratio = count(Aggressive) ÷ count(Persuasive) per account. '
        f'> 1.0 = more attack-oriented rhetoric; < 1.0 = more constructive persuasion.',
        y=-0.04)
    save_chart(fig, 'T6_aggressive_persuasive_ratio.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T7: Mean Engagement by Tone
    # ═════════════════════════════════════════════════════════════════════════
    tone_er = compute_mean_engagement(hc_tone, 'tone_primary', 'Engagement_Rate_Monthly')

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('T7', 'Mean Engagement Rate by Rhetorical Tone',
                              f'Which rhetorical approach drives the most engagement?'),
                 fontsize=14, fontweight='bold', y=1.02)

    tone_sorted = sorted(TONE_ORDER, key=lambda t: tone_er.get(t, {}).get('mean', 0), reverse=True)
    vals = [tone_er.get(t, {}).get('mean', 0) * 100 for t in tone_sorted]
    cnts = [tone_er.get(t, {}).get('count', 0) for t in tone_sorted]
    td   = [t.replace('_', ' ') for t in tone_sorted]
    bars = ax.bar(td, vals, color=[TONE_COLORS[t] for t in tone_sorted],
                  edgecolor='white', linewidth=0.8, width=0.55)
    add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=9)
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.set_xlabel('Rhetorical Tone')
    ax.set_ylim(0, max(vals) * 1.3 if vals else 0.1)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly column grouped by tone_primary. '
        f'Mean ER = mean(Engagement_Rate_Monthly) per tone × 100 for %. '
        f'n-counts = number of posts with each tone. Neutral-tone posts excluded.',
        y=-0.04)
    save_chart(fig, 'T7_engagement_by_tone.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T8: Tone Confidence by Label
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('T8', 'Mean Tone Classifier Confidence by Tone Label',
                              f'How confident is the classifier for each rhetorical tone?'),
                 fontsize=14, fontweight='bold', y=1.02)

    conf_vals = []
    conf_cnts = []
    for t in TONE_ORDER:
        sub = hc_tone[hc_tone['tone_primary'] == t]['tone_primary_c']
        conf_vals.append(sub.mean())
        conf_cnts.append(len(sub))

    bars = ax.bar(tone_display, conf_vals, color=[TONE_COLORS[t] for t in TONE_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.55)
    add_value_labels(ax, bars, fmt='{:.3f}', counts=conf_cnts, fontsize=9)
    ax.set_ylabel('Mean Confidence Score')
    ax.set_xlabel('Rhetorical Tone')
    ax.set_ylim(0, max(conf_vals) * 1.2 if conf_vals else 1)
    ax.tick_params(axis='x', rotation=10)

    add_methodology(fig,
        f'Source: tone_primary_c column grouped by tone_primary. '
        f'Value = mean(tone_primary_c) per label. Neutral-tone posts excluded. '
        f'Higher confidence = the classifier was more certain about this tone classification.',
        y=-0.04)
    save_chart(fig, 'T8_tone_confidence_by_label.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T9: Victim Narrative Usage by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('T9', 'Victim Narrative Usage by Account',
                              f'Percentage of posts using victim narrative rhetoric'),
                 fontsize=14, fontweight='bold', y=1.02)

    vn_pcts = []
    vn_labels = []
    vn_ns = []
    for acc in ACC_ORDER:
        sub = hc_tone[hc_tone['_account'] == acc]
        c = (sub['tone_primary'] == 'Victim_Narrative').sum()
        total = len(sub)
        vn_pcts.append(pct(c, total))
        vn_labels.append(f'{ACC_LABELS[acc]}\n(n={total:,})')
        vn_ns.append(c)

    bars = ax.bar(vn_labels, vn_pcts, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                f'{h:.1f}%\n({vn_ns[i]:,} posts)', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#333333')
    ax.set_ylabel('Victim Narrative (%)')
    ax.set_ylim(0, max(vn_pcts) * 1.4 if vn_pcts else 10)

    add_methodology(fig,
        f'Source: tone_primary column. '
        f'Victim Narrative % = count(tone_primary="Victim_Narrative") ÷ total classified posts (excl. Neutral) per account × 100. '
        f'Victim narrative = rhetoric where the speaker/party positions itself or its supporters as victims.',
        y=-0.04)
    save_chart(fig, 'T9_victim_narrative_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART T10: Tone Trajectory per Account Across Timelines
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('T10', 'Rhetorical Tone Trajectory per Account Across Timelines',
                              f'How each account\'s rhetorical strategy evolves'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_tone[hc_tone['_account'] == acc]
        x = np.arange(len(TL_ORDER))
        n_tones_plot = len(TONE_ORDER)
        w = 0.14
        off = -(n_tones_plot - 1) * w / 2

        for j, tone in enumerate(TONE_ORDER):
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['tone_primary'] == tone).sum()
                vals.append(pct(c, total))
            bars = ax.bar(x + off + j * w, vals, w,
                          label=tone.replace('_', ' ') if i == 0 else '',
                          color=TONE_COLORS[tone], edgecolor='white', linewidth=0.3)
            for bar in bars:
                h = bar.get_height()
                if h > 3:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                            f'{h:.0f}%', ha='center', fontsize=5, color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]} (n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 65)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Tone', fontsize=6, title_fontsize=7, ncol=2, loc='upper left')

    add_methodology(fig,
        f'Source: tone_primary × _timeline × _account. Neutral-tone excluded. '
        f'Each bar = % of that tone within that account-timeline combination.',
        y=-0.10)
    save_chart(fig, 'T10_tone_trajectory_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    overall_tone = compute_distribution(hc_tone['tone_primary'], TONE_ORDER)[0]
    by_acc_tone  = compute_grouped_distribution(hc_tone, '_account', 'tone_primary', TONE_ORDER)
    by_al_tone   = compute_grouped_distribution(hc_tone, '_alliance', 'tone_primary', TONE_ORDER)
    by_tl_tone   = compute_grouped_distribution(hc_tone, '_timeline', 'tone_primary', TONE_ORDER)

    sec_insights = []

    # 4.1 Overall
    sec_insights.append((
        'INSIGHT 4.1',
        'Overall Rhetorical Tone Distribution',
        f'Among {n_tone:,} classified posts (excluding {neutral_tone_count:,} Neutral-tone posts):\n\n'
        + '\n'.join([f'    • {t.replace("_", " ")}: {overall_tone[t]["count"]:,} ({overall_tone[t]["pct"]:.1f}%)'
                     for t in TONE_ORDER]) + '\n\n'
        f'  Aggressive rhetoric is the most common tone at {overall_tone["Aggressive"]["pct"]:.1f}%, '
        f'followed by Persuasive at {overall_tone["Persuasive"]["pct"]:.1f}%. Together, these two '
        f'tones account for {overall_tone["Aggressive"]["pct"] + overall_tone["Persuasive"]["pct"]:.1f}% '
        f'of all classified posts, indicating that AP political social media is dominated by '
        f'either attack-oriented or constructive-persuasion content. Inspirational content '
        f'({overall_tone["Inspirational"]["pct"]:.1f}%) is a distant third, while Victim Narrative '
        f'({overall_tone["Victim_Narrative"]["pct"]:.1f}%) and Defensive ({overall_tone["Defensive"]["pct"]:.1f}%) '
        f'are minority strategies.',
        f'Counts of tone_primary per label ÷ {n_tone:,} × 100. Neutral-tone posts ({neutral_tone_count:,}) excluded. '
        f'SKIPPED/_insufficient posts also excluded upfront.'
    ))

    # 4.2 By account
    acc_tone_lines = []
    for acc in ACC_ORDER:
        d = by_acc_tone.get(acc, {})
        dominant = max(TONE_ORDER, key=lambda t: d.get(t, {}).get('pct', 0))
        acc_tone_lines.append(
            f'    • {ACC_LABELS[acc]} (n={d.get("_total", 0):,}): '
            f'Dominant={dominant.replace("_", " ")} ({d.get(dominant, {}).get("pct", 0):.1f}%) | '
            f'Agg={d.get("Aggressive", {}).get("pct", 0):.1f}%, '
            f'Per={d.get("Persuasive", {}).get("pct", 0):.1f}%, '
            f'Ins={d.get("Inspirational", {}).get("pct", 0):.1f}%'
        )

    most_agg = max(ACC_ORDER, key=lambda a: by_acc_tone.get(a, {}).get('Aggressive', {}).get('pct', 0))
    most_per = max(ACC_ORDER, key=lambda a: by_acc_tone.get(a, {}).get('Persuasive', {}).get('pct', 0))
    most_ins = max(ACC_ORDER, key=lambda a: by_acc_tone.get(a, {}).get('Inspirational', {}).get('pct', 0))

    sec_insights.append((
        'INSIGHT 4.2',
        'Rhetorical Tone Breakdown by Account',
        f'Per-account tone profiles:\n\n'
        + '\n'.join(acc_tone_lines) + '\n\n'
        f'  {ACC_LABELS[most_agg]} leads in aggression ({by_acc_tone.get(most_agg, {}).get("Aggressive", {}).get("pct", 0):.1f}%), '
        f'{ACC_LABELS[most_per]} in persuasion ({by_acc_tone.get(most_per, {}).get("Persuasive", {}).get("pct", 0):.1f}%), and '
        f'{ACC_LABELS[most_ins]} in inspiration ({by_acc_tone.get(most_ins, {}).get("Inspirational", {}).get("pct", 0):.1f}%). '
        f'These profiles reveal fundamentally different communication strategies '
        f'across the political spectrum in AP.',
        f'Counts of tone_primary per _account ÷ total classified posts (excl. Neutral) for that account × 100.'
    ))

    # 4.3 Aggressive-to-Persuasive ratio
    ratio_lines = []
    for acc in ACC_ORDER:
        sub = hc_tone[hc_tone['_account'] == acc]
        n_a = (sub['tone_primary'] == 'Aggressive').sum()
        n_p = (sub['tone_primary'] == 'Persuasive').sum()
        r = n_a / n_p if n_p > 0 else float('inf')
        ratio_lines.append(f'    • {ACC_LABELS[acc]}: {r:.2f} (Agg={n_a:,}, Per={n_p:,})')

    sec_insights.append((
        'INSIGHT 4.3',
        'Aggressive-to-Persuasive Ratio — Attack vs Constructive Strategy',
        f'Aggressive/Persuasive ratio by account:\n\n'
        + '\n'.join(ratio_lines) + '\n\n'
        f'  A ratio > 1.0 indicates a strategy that favours attack rhetoric over constructive '
        f'persuasion. A ratio < 1.0 indicates a focus on positive messaging. '
        f'This metric is a concise summary of each account\'s rhetorical orientation.',
        f'Ratio = count(Aggressive) ÷ count(Persuasive) per account. '
        f'Source: tone_primary column, Neutral excluded.'
    ))

    # 4.4 Alliance
    ysrcp_tone = by_al_tone.get('YSRCP Alliance', {})
    tdp_tone   = by_al_tone.get('TDP Alliance', {})

    sec_insights.append((
        'INSIGHT 4.4',
        'Alliance-Level Rhetorical Strategy Comparison',
        f'YSRCP Alliance (n={ysrcp_tone.get("_total", 0):,}):\n'
        + '\n'.join([f'    {t.replace("_", " ")}: {ysrcp_tone.get(t, {}).get("pct", 0):.1f}% ({ysrcp_tone.get(t, {}).get("count", 0):,})'
                     for t in TONE_ORDER])
        + f'\n\n  TDP Alliance (n={tdp_tone.get("_total", 0):,}):\n'
        + '\n'.join([f'    {t.replace("_", " ")}: {tdp_tone.get(t, {}).get("pct", 0):.1f}% ({tdp_tone.get(t, {}).get("count", 0):,})'
                     for t in TONE_ORDER])
        + f'\n\n  The alliance comparison reveals the aggregate rhetorical balance of each '
        f'political camp, smoothing out individual account differences.',
        f'tone_primary counts per _alliance ÷ total classified posts (excl. Neutral) per alliance × 100.'
    ))

    # 4.5 Victim Narrative
    vn_rates = {acc: pct((hc_tone[hc_tone['_account']==acc]['tone_primary']=='Victim_Narrative').sum(),
                         len(hc_tone[hc_tone['_account']==acc]))
                for acc in ACC_ORDER}
    most_vn = max(ACC_ORDER, key=lambda a: vn_rates[a])

    sec_insights.append((
        'INSIGHT 4.5',
        'Victim Narrative Usage — Who Plays the Victim?',
        f'Victim Narrative percentage by account:\n\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {vn_rates[a]:.1f}%' for a in ACC_ORDER]) + '\n\n'
        f'  {ACC_LABELS[most_vn]} uses victim narrative rhetoric the most ({vn_rates[most_vn]:.1f}%). '
        f'Victim narrative is a powerful political tool used to generate sympathy and '
        f'frame the opposition as oppressors. Its strategic deployment varies significantly '
        f'across accounts.',
        f'Count of tone_primary="Victim_Narrative" per account ÷ total classified posts (excl. Neutral) per account × 100.'
    ))

    # 4.6 Defensive tone
    def_rates = {acc: pct((hc_tone[hc_tone['_account']==acc]['tone_primary']=='Defensive').sum(),
                          len(hc_tone[hc_tone['_account']==acc]))
                 for acc in ACC_ORDER}

    sec_insights.append((
        'INSIGHT 4.6',
        'Defensive Tone Analysis — A Rare Strategy',
        f'Defensive tone is the rarest rhetorical strategy, appearing in only '
        f'{overall_tone["Defensive"]["count"]:,} posts ({overall_tone["Defensive"]["pct"]:.1f}%).\n\n'
        f'  Per-account Defensive rates:\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {def_rates[a]:.1f}%' for a in ACC_ORDER]) + '\n\n'
        f'  The rarity of defensive rhetoric suggests that political actors in AP prefer '
        f'to go on the offensive rather than defend against attacks. When defensive tone '
        f'does appear, it is typically in response to specific allegations or scandals.',
        f'Count of tone_primary="Defensive" per account ÷ total classified posts (excl. Neutral) per account × 100.'
    ))

    # 4.7 Timeline evolution
    first_agg = by_tl_tone.get(TL_ORDER[0], {}).get('Aggressive', {}).get('pct', 0)
    last_agg  = by_tl_tone.get(TL_ORDER[-1], {}).get('Aggressive', {}).get('pct', 0)
    agg_trend = 'increasing' if last_agg > first_agg else 'decreasing'

    tl_lines = []
    for tl in TL_ORDER:
        d = by_tl_tone.get(tl, {})
        tl_lines.append(
            f'    • {TL_LABELS[tl]} (n={d.get("_total", 0):,}): '
            f'Agg={d.get("Aggressive", {}).get("pct", 0):.1f}%, '
            f'Per={d.get("Persuasive", {}).get("pct", 0):.1f}%, '
            f'VN={d.get("Victim_Narrative", {}).get("pct", 0):.1f}%'
        )

    sec_insights.append((
        'INSIGHT 4.7',
        'Rhetorical Tone Temporal Trends',
        f'Tone evolution across timelines:\n\n'
        + '\n'.join(tl_lines) + '\n\n'
        f'  Aggression shows a {agg_trend} trend: {first_agg:.1f}% ({TL_LABELS[TL_ORDER[0]]}) → '
        f'{last_agg:.1f}% ({TL_LABELS[TL_ORDER[-1]]}), a shift of {last_agg - first_agg:+.1f}pp. '
        f'The pre-election period (Jan–May 2024) is particularly noteworthy for any '
        f'escalation or de-escalation in rhetorical aggression.',
        f'tone_primary counts per _timeline ÷ total classified posts (excl. Neutral) in that period × 100.'
    ))

    # 4.8 Engagement
    tone_engagement = {t: tone_er.get(t, {}).get('mean', 0) for t in TONE_ORDER}
    top_eng = max(TONE_ORDER, key=lambda t: tone_engagement[t])
    low_eng = min(TONE_ORDER, key=lambda t: tone_engagement[t])

    sec_insights.append((
        'INSIGHT 4.8',
        'Tone–Engagement Relationship',
        f'Mean Engagement Rate (Monthly) by tone:\n\n'
        + '\n'.join([f'    • {t.replace("_", " ")}: {tone_engagement[t]*100:.4f}% '
                     f'(n={tone_er.get(t, {}).get("count", 0):,})'
                     for t in sorted(TONE_ORDER, key=lambda t: tone_engagement[t], reverse=True)])
        + f'\n\n  {top_eng.replace("_", " ")} generates the highest mean engagement ({tone_engagement[top_eng]*100:.4f}%), '
        f'while {low_eng.replace("_", " ")} the lowest ({tone_engagement[low_eng]*100:.4f}%). '
        f'This suggests audiences reward certain rhetorical approaches more than others.',
        f'Mean of Engagement_Rate_Monthly per tone_primary label × 100 for %. '
        f'Neutral-tone excluded.'
    ))

    # 4.9 Platform difference
    fb_tone = grp_pl.get('Facebook', {})
    ig_tone = grp_pl.get('Instagram', {})

    sec_insights.append((
        'INSIGHT 4.9',
        'Platform Tone Differences — Facebook vs Instagram',
        f'Facebook (n={fb_tone.get("_total", 0):,}):\n'
        + '\n'.join([f'    {t.replace("_", " ")}: {fb_tone.get(t, {}).get("pct", 0):.1f}%' for t in TONE_ORDER])
        + f'\n\n  Instagram (n={ig_tone.get("_total", 0):,}):\n'
        + '\n'.join([f'    {t.replace("_", " ")}: {ig_tone.get(t, {}).get("pct", 0):.1f}%' for t in TONE_ORDER])
        + f'\n\n  Platform-level differences may reflect strategic content adaptation: '
        f'Instagram audiences (younger, visual-oriented) may respond differently to '
        f'aggressive rhetoric compared to Facebook audiences.',
        f'tone_primary counts per _platform ÷ total classified posts (excl. Neutral) on that platform × 100.'
    ))

    # 4.10 Confidence
    tone_conf = {t: hc_tone[hc_tone['tone_primary']==t]['tone_primary_c'].mean() for t in TONE_ORDER}
    most_conf = max(TONE_ORDER, key=lambda t: tone_conf[t])

    sec_insights.append((
        'INSIGHT 4.10',
        'Classifier Confidence by Tone',
        f'Mean classifier confidence score by tone:\n\n'
        + '\n'.join([f'    • {t.replace("_", " ")}: {tone_conf[t]:.3f}' for t in TONE_ORDER]) + '\n\n'
        f'  {most_conf.replace("_", " ")} has the highest mean confidence ({tone_conf[most_conf]:.3f}), '
        f'suggesting the model finds the clearest linguistic markers for that tone. '
        f'Lower confidence tones are more ambiguous and may benefit from further validation.',
        f'Mean of tone_primary_c per tone_primary label. '
        f'Higher values = more certain classification.'
    ))

    # 4.11 Conclusion
    sec_insights.append((
        'INSIGHT 4.11',
        'Conclusion — Rhetorical Tone Landscape',
        f'The AP political rhetorical landscape is characterised by:\n\n'
        f'  1. Aggressive dominance at {overall_tone["Aggressive"]["pct"]:.1f}% — the prevailing strategy.\n'
        f'  2. Persuasive rhetoric ({overall_tone["Persuasive"]["pct"]:.1f}%) serves as the primary '
        f'counterweight to aggression.\n'
        f'  3. Victim Narrative ({overall_tone["Victim_Narrative"]["pct"]:.1f}%) and Defensive '
        f'({overall_tone["Defensive"]["pct"]:.1f}%) are niche strategies with uneven adoption.\n'
        f'  4. {top_eng.replace("_", " ")} content generates the highest engagement.\n'
        f'  5. Aggression is {agg_trend} toward the pre-election period ({last_agg - first_agg:+.1f}pp).\n'
        f'  6. Significant inter-account variation reveals distinct rhetorical identities.\n\n'
        f'  These findings indicate a highly confrontational political communication '
        f'environment where attack-based rhetoric is the default, but with meaningful '
        f'variation in how each account balances aggression with persuasion and inspiration.',
        f'Synthesis of insights 4.1–4.10.'
    ))

    all_sections.append(('Rhetorical Tone Analysis', sec_insights))

    write_insight_report('04_rhetorical_tone_insights.txt',
                         'RHETORICAL TONE ANALYSIS', all_sections)

    print(f"\n  Done! Generated 10 charts + 1 text report (11 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

# ============================================================
#  SCRIPT 07 — TIMELINE EVOLUTION ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: How all sentiment dimensions evolve across timelines
#  Timelines: Jun–Aug 2023, Sep–Dec 2023, Jan–May 2024
#  Focus: Temporal trends, pre-election shifts, per-account trajectories
#
#  Run: python insight_07_timeline_evolution.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 07: TIMELINE EVOLUTION ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)
    n_clf = len(hc_clf)

    hc_tone = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL1: Post Volume by Timeline × Account
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('TL1', 'Post Volume by Timeline × Account',
                              f'Who posts more as elections approach?'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    n_acc = len(ACC_ORDER)
    width = 0.18
    offset = -(n_acc - 1) * width / 2

    for j, acc in enumerate(ACC_ORDER):
        vals = []
        for tl in TL_ORDER:
            c = ((hc['_timeline'] == tl) & (hc['_account'] == acc)).sum()
            vals.append(c)
        bars = ax.bar(x + offset + j * width, vals, width, label=ACC_LABELS[acc],
                      color=ACC_COLORS[acc], edgecolor='white', linewidth=0.5)
        add_value_labels(ax, bars, fmt='{:,.0f}', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([TL_LABELS[tl] for tl in TL_ORDER])
    ax.set_ylabel('Number of Posts')
    ax.legend(title='Account', framealpha=0.9)

    add_methodology(fig,
        f'Source: _timeline × _account columns. All {n_total:,} HC posts. '
        f'Bar heights = raw count of posts per account per timeline.',
        y=-0.04)
    save_chart(fig, 'TL1_post_volume_by_timeline_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL2: Sentiment Evolution per Account (Line Chart)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('TL2', 'Caption Sentiment Evolution per Account',
                              f'How each account\'s sentiment shifts across timelines'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]

        for sent in SENT_ORDER:
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['sentiment_label'] == sent).sum()
                vals.append(pct(c, total))
            ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2,
                    color=SENT_COLORS[sent], label=sent if i == 0 else '',
                    markersize=6)
            for k, v in enumerate(vals):
                ax.text(k, v + 1, f'{v:.1f}%', ha='center', fontsize=7,
                        color=SENT_COLORS[sent], fontweight='bold')

        ax.set_xticks(range(len(TL_ORDER)))
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]} (n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Sentiment', fontsize=7, title_fontsize=8)

    add_methodology(fig,
        f'Source: sentiment_label × _timeline × _account. '
        f'% = count of label per account-timeline ÷ total posts for that account-timeline × 100.',
        y=-0.08)
    save_chart(fig, 'TL2_sentiment_evolution_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL3: Emotion Evolution per Account (Anger + Hope focus)
    # ═════════════════════════════════════════════════════════════════════════
    focus_emos = ['Anger', 'Hope', 'Trust', 'Pride']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('TL3', 'Key Emotion Evolution per Account',
                              f'Tracking Anger, Hope, Trust, Pride over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_clf[hc_clf['_account'] == acc]

        for emo in focus_emos:
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['emo_primary'] == emo).sum()
                vals.append(pct(c, total))
            ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2,
                    color=EMO_COLORS[emo], label=emo if i == 0 else '',
                    markersize=6)
            for k, v in enumerate(vals):
                ax.text(k, v + 0.7, f'{v:.0f}%', ha='center', fontsize=6,
                        color=EMO_COLORS[emo])

        ax.set_xticks(range(len(TL_ORDER)))
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]} (n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Emotion', fontsize=7, title_fontsize=8)

    add_methodology(fig,
        f'Source: emo_primary × _timeline × _account. Classified posts only. '
        f'Focus on 4 key emotions (Anger/Hope/Trust/Pride). '
        f'% = count of emotion per account-timeline ÷ total classified per account-timeline × 100.',
        y=-0.08)
    save_chart(fig, 'TL3_emotion_evolution_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL4: Tone Evolution per Account (Aggressive + Persuasive focus)
    # ═════════════════════════════════════════════════════════════════════════
    focus_tones = ['Aggressive', 'Persuasive', 'Inspirational', 'Victim_Narrative']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('TL4', 'Key Tone Evolution per Account',
                              f'Tracking Aggressive, Persuasive, Inspirational, Victim Narrative'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_tone[hc_tone['_account'] == acc]

        for tone in focus_tones:
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['tone_primary'] == tone).sum()
                vals.append(pct(c, total))
            ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2,
                    color=TONE_COLORS[tone],
                    label=tone.replace('_', ' ') if i == 0 else '',
                    markersize=6)
            for k, v in enumerate(vals):
                ax.text(k, v + 0.7, f'{v:.0f}%', ha='center', fontsize=6,
                        color=TONE_COLORS[tone])

        ax.set_xticks(range(len(TL_ORDER)))
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]} (n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 70)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Tone', fontsize=6, title_fontsize=7)

    add_methodology(fig,
        f'Source: tone_primary × _timeline × _account. Neutral-tone excluded. '
        f'% = count per tone per account-timeline ÷ total classified (excl. Neutral) per account-timeline × 100.',
        y=-0.08)
    save_chart(fig, 'TL4_tone_evolution_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL5: Engagement Rate Evolution per Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('TL5', 'Mean Engagement Rate Evolution per Account',
                              f'How engagement changes across timelines'),
                 fontsize=14, fontweight='bold', y=1.02)

    for acc in ACC_ORDER:
        vals = []
        ns = []
        for tl in TL_ORDER:
            sub = hc[(hc['_account'] == acc) & (hc['_timeline'] == tl)]
            vals.append(sub['Engagement_Rate_Monthly'].mean() * 100)
            ns.append(len(sub))
        ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2.5,
                color=ACC_COLORS[acc], label=ACC_LABELS[acc], markersize=8)
        for k, v in enumerate(vals):
            ax.text(k, v + 0.02, f'{v:.4f}%\n(n={ns[k]:,})', ha='center',
                    fontsize=7, color=ACC_COLORS[acc], fontweight='bold')

    ax.set_xticks(range(len(TL_ORDER)))
    ax.set_xticklabels([TL_LABELS[tl] for tl in TL_ORDER])
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.legend(title='Account', framealpha=0.9)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × _timeline × _account. '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 per account-timeline pair.',
        y=-0.04)
    save_chart(fig, 'TL5_engagement_evolution_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL6: Alliance Sentiment Evolution
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(format_title('TL6', 'Alliance Sentiment Evolution Across Timelines',
                              f'How each alliance\'s sentiment shifts over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        sub = hc[hc['_alliance'] == al]

        for sent in SENT_ORDER:
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['sentiment_label'] == sent).sum()
                vals.append(pct(c, total))
            ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2,
                    color=SENT_COLORS[sent], label=sent if i == 0 else '',
                    markersize=7)
            for k, v in enumerate(vals):
                ax.text(k, v + 1.2, f'{v:.1f}%', ha='center', fontsize=8,
                        color=SENT_COLORS[sent], fontweight='bold')

        ax.set_xticks(range(len(TL_ORDER)))
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=8)
        ax.set_title(f'{al}\n(n={len(sub):,})', fontsize=11,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Sentiment', fontsize=8, title_fontsize=9)

    add_methodology(fig,
        f'Source: sentiment_label × _timeline × _alliance. '
        f'% = count per sentiment per alliance-timeline ÷ total per alliance-timeline × 100.',
        y=-0.06)
    save_chart(fig, 'TL6_alliance_sentiment_evolution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL7: Platform Volume and Sentiment Over Time
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(format_title('TL7', 'Platform Sentiment Evolution Across Timelines',
                              f'Facebook vs Instagram over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, plat in enumerate(PLAT_ORDER):
        ax = axes[i]
        sub = hc[hc['_platform'] == plat]

        for sent in SENT_ORDER:
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['sentiment_label'] == sent).sum()
                vals.append(pct(c, total))
            ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2,
                    color=SENT_COLORS[sent], label=sent if i == 0 else '',
                    markersize=7)
            for k, v in enumerate(vals):
                ax.text(k, v + 1, f'{v:.1f}%', ha='center', fontsize=7,
                        color=SENT_COLORS[sent])

        ax.set_xticks(range(len(TL_ORDER)))
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{plat}\n(n={len(sub):,})', fontsize=11,
                     color=PLAT_COLORS[plat], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Sentiment', fontsize=7, title_fontsize=8)

    add_methodology(fig,
        f'Source: sentiment_label × _timeline × _platform. '
        f'% = count per sentiment per platform-timeline ÷ total per platform-timeline × 100.',
        y=-0.06)
    save_chart(fig, 'TL7_platform_sentiment_evolution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART TL8: Subjectivity Evolution Over Time
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('TL8', 'Subjectivity Evolution Across Timelines',
                              f'Is discourse becoming more fact-based or opinion-based?'),
                 fontsize=14, fontweight='bold', y=1.02)

    for subj in SUBJ_ORDER:
        vals = []
        for tl in TL_ORDER:
            tl_sub = hc_clf[hc_clf['_timeline'] == tl]
            total = len(tl_sub)
            c = (tl_sub['subj_label'] == subj).sum()
            vals.append(pct(c, total))
        ax.plot(range(len(TL_ORDER)), vals, marker='o', linewidth=2.5,
                color=SUBJ_COLORS[subj], label=subj, markersize=8)
        for k, v in enumerate(vals):
            ax.text(k, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9,
                    color=SUBJ_COLORS[subj], fontweight='bold')

    ax.set_xticks(range(len(TL_ORDER)))
    ax.set_xticklabels([TL_LABELS[tl] for tl in TL_ORDER])
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 50)
    ax.legend(title='Subjectivity', framealpha=0.9)

    add_methodology(fig,
        f'Source: subj_label × _timeline. Classified posts only. '
        f'% = count per subj per timeline ÷ total classified per timeline × 100.',
        y=-0.04)
    save_chart(fig, 'TL8_subjectivity_evolution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    sec_insights = []

    # 7.1 Post volume trends
    vol_lines = []
    for acc in ACC_ORDER:
        acc_vols = []
        for tl in TL_ORDER:
            c = ((hc['_timeline'] == tl) & (hc['_account'] == acc)).sum()
            acc_vols.append(c)
        trend = 'increasing' if acc_vols[-1] > acc_vols[0] else 'decreasing'
        vol_lines.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:,}" for v in acc_vols])} ({trend})'
        )

    sec_insights.append((
        'INSIGHT 7.1',
        'Post Volume Trends Across Timelines',
        f'Number of posts per account per timeline:\n\n'
        + '\n'.join(vol_lines) + '\n\n'
        f'  Changes in posting frequency reveal strategic intensification or de-escalation '
        f'as elections approach. An increase in the final period (Jan–May 2024) may indicate '
        f'pre-election campaign ramping.',
        f'Count of rows per _account per _timeline. All HC posts (n={n_total:,}).'
    ))

    # 7.2 Negativity trajectory
    neg_trajectories = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        neg_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            total = len(tl_sub)
            n_neg = (tl_sub['sentiment_label'] == 'Negative').sum()
            neg_vals.append(pct(n_neg, total))
        change = neg_vals[-1] - neg_vals[0]
        neg_trajectories.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:.1f}%" for v in neg_vals])} '
            f'(Δ={change:+.1f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.2',
        'Negativity Trajectory per Account',
        f'Negative sentiment % across timelines:\n\n'
        + '\n'.join(neg_trajectories) + '\n\n'
        f'  Tracking negativity over time reveals whether discourse is escalating or '
        f'de-escalating. A rising negativity trajectory in the pre-election period '
        f'suggests intensifying political attacks.',
        f'Count of sentiment_label="Negative" per account-timeline ÷ total per account-timeline × 100.'
    ))

    # 7.3 Anger trajectory
    anger_trajectories = []
    for acc in ACC_ORDER:
        sub = hc_clf[hc_clf['_account'] == acc]
        anger_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            total = len(tl_sub)
            n_ang = (tl_sub['emo_primary'] == 'Anger').sum()
            anger_vals.append(pct(n_ang, total))
        change = anger_vals[-1] - anger_vals[0]
        anger_trajectories.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:.1f}%" for v in anger_vals])} '
            f'(Δ={change:+.1f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.3',
        'Anger Trajectory per Account',
        f'Anger (emo_primary) % across timelines:\n\n'
        + '\n'.join(anger_trajectories) + '\n\n'
        f'  Anger is the key emotion driving political discourse. Its trajectory '
        f'reveals whether the emotional temperature is rising or cooling.',
        f'Count of emo_primary="Anger" per account-timeline ÷ total classified per account-timeline × 100.'
    ))

    # 7.4 Aggression trajectory
    agg_trajectories = []
    for acc in ACC_ORDER:
        sub = hc_tone[hc_tone['_account'] == acc]
        agg_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            total = len(tl_sub)
            n_agg = (tl_sub['tone_primary'] == 'Aggressive').sum()
            agg_vals.append(pct(n_agg, total))
        change = agg_vals[-1] - agg_vals[0]
        agg_trajectories.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:.1f}%" for v in agg_vals])} '
            f'(Δ={change:+.1f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.4',
        'Aggression Trajectory per Account',
        f'Aggressive tone % across timelines:\n\n'
        + '\n'.join(agg_trajectories) + '\n\n'
        f'  Escalation or de-escalation of aggressive rhetoric is a key indicator '
        f'of how the political climate is evolving.',
        f'Count of tone_primary="Aggressive" per account-timeline ÷ total classified (excl. Neutral) '
        f'per account-timeline × 100.'
    ))

    # 7.5 Hope/Trust trajectory
    hope_trajectories = []
    for acc in ACC_ORDER:
        sub = hc_clf[hc_clf['_account'] == acc]
        hope_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            total = len(tl_sub)
            n_h = tl_sub['emo_primary'].isin(['Hope', 'Trust']).sum()
            hope_vals.append(pct(n_h, total))
        change = hope_vals[-1] - hope_vals[0]
        hope_trajectories.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:.1f}%" for v in hope_vals])} '
            f'(Δ={change:+.1f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.5',
        'Hope+Trust (Positive Emotion) Trajectory per Account',
        f'Combined Hope+Trust % across timelines:\n\n'
        + '\n'.join(hope_trajectories) + '\n\n'
        f'  An increase in hope/trust suggests a shift toward aspirational messaging; '
        f'a decrease suggests deepening negativity and attack-oriented strategy.',
        f'Count of emo_primary in (Hope, Trust) per account-timeline ÷ total classified '
        f'per account-timeline × 100.'
    ))

    # 7.6 Engagement trajectory
    er_trajectories = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        er_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            er_vals.append(tl_sub['Engagement_Rate_Monthly'].mean() * 100)
        change = er_vals[-1] - er_vals[0]
        er_trajectories.append(
            f'    • {ACC_LABELS[acc]}: {" → ".join([f"{v:.4f}%" for v in er_vals])} '
            f'(Δ={change:+.4f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.6',
        'Engagement Rate Trajectory per Account',
        f'Mean Engagement Rate (Monthly) across timelines:\n\n'
        + '\n'.join(er_trajectories) + '\n\n'
        f'  Engagement trajectory shows whether audiences are becoming more or less '
        f'responsive to political content over time. Pre-election periods often see '
        f'heightened engagement due to increased political interest.',
        f'Mean(Engagement_Rate_Monthly) per account-timeline × 100.'
    ))

    # 7.7 Pre-election period analysis
    pre_election = hc[hc['_timeline'] == 'JanMay2024']
    earlier = hc[hc['_timeline'] != 'JanMay2024']

    pre_neg = pct((pre_election['sentiment_label'] == 'Negative').sum(), len(pre_election))
    earlier_neg = pct((earlier['sentiment_label'] == 'Negative').sum(), len(earlier))

    pre_clf = hc_clf[hc_clf['_timeline'] == 'JanMay2024']
    earlier_clf = hc_clf[hc_clf['_timeline'] != 'JanMay2024']

    pre_anger = pct((pre_clf['emo_primary'] == 'Anger').sum(), len(pre_clf))
    earlier_anger = pct((earlier_clf['emo_primary'] == 'Anger').sum(), len(earlier_clf))

    sec_insights.append((
        'INSIGHT 7.7',
        'Pre-Election Period (Jan–May 2024) vs Earlier Periods',
        f'Comparing the pre-election period to earlier periods:\n\n'
        f'  Negative sentiment: Pre-election={pre_neg:.1f}%, Earlier={earlier_neg:.1f}%, '
        f'Δ={pre_neg - earlier_neg:+.1f}pp\n'
        f'  Anger emotion: Pre-election={pre_anger:.1f}%, Earlier={earlier_anger:.1f}%, '
        f'Δ={pre_anger - earlier_anger:+.1f}pp\n'
        f'  Pre-election posts: {len(pre_election):,} ({pct(len(pre_election), n_total):.1f}% of total)\n'
        f'  Earlier posts: {len(earlier):,} ({pct(len(earlier), n_total):.1f}% of total)\n\n'
        f'  The pre-election period (AP state elections, Jun 2024) is crucial for understanding '
        f'campaign intensification. '
        f'{"Negativity increased" if pre_neg > earlier_neg else "Negativity decreased"} '
        f'and {"anger escalated" if pre_anger > earlier_anger else "anger de-escalated"} '
        f'as the election approached.',
        f'Pre-election = _timeline="JanMay2024". Earlier = Jun-Aug 2023 + Sep-Dec 2023. '
        f'Percentages computed separately for each period.'
    ))

    # 7.8 Alliance temporal dynamics
    alliance_tl_lines = []
    for al in ALLIANCE_ORDER:
        sub = hc[hc['_alliance'] == al]
        neg_vals = []
        for tl in TL_ORDER:
            tl_sub = sub[sub['_timeline'] == tl]
            total = len(tl_sub)
            n_neg = (tl_sub['sentiment_label'] == 'Negative').sum()
            neg_vals.append(pct(n_neg, total))
        change = neg_vals[-1] - neg_vals[0]
        alliance_tl_lines.append(
            f'    • {al}: {" → ".join([f"{v:.1f}%" for v in neg_vals])} '
            f'(Δ={change:+.1f}pp)'
        )

    sec_insights.append((
        'INSIGHT 7.8',
        'Alliance Negativity Trajectory',
        f'Negative sentiment % by alliance across timelines:\n\n'
        + '\n'.join(alliance_tl_lines) + '\n\n'
        f'  The alliance-level trajectory reveals the aggregate strategic direction '
        f'of each political camp over time.',
        f'Count of sentiment_label="Negative" per alliance-timeline ÷ total per alliance-timeline × 100.'
    ))

    # 7.9 Conclusion
    sec_insights.append((
        'INSIGHT 7.9',
        'Conclusion — Temporal Dynamics',
        f'Key temporal findings:\n\n'
        f'  1. Post volume shows distinct patterns across accounts as elections approach.\n'
        f'  2. Negativity follows account-specific trajectories — not all accounts become '
        f'more negative simultaneously.\n'
        f'  3. Anger and aggression trajectories reveal escalation/de-escalation patterns.\n'
        f'  4. The pre-election period (Jan–May 2024) shows '
        f'{"heightened" if pre_neg > earlier_neg else "moderated"} negativity overall.\n'
        f'  5. Engagement rates evolve over time, potentially reflecting audience fatigue '
        f'or heightened political interest.\n\n'
        f'  These temporal patterns are essential for understanding the dynamics of political '
        f'social media communication in the context of the AP state elections.',
        f'Synthesis of insights 7.1–7.8.'
    ))

    all_sections.append(('Timeline Evolution Analysis', sec_insights))

    write_insight_report('07_timeline_evolution_insights.txt',
                         'TIMELINE EVOLUTION ANALYSIS', all_sections)

    print(f"\n  Done! Generated 8 charts + 1 text report (9 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

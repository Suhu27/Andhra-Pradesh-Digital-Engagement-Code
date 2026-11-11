# ============================================================
#  SCRIPT 03 — EMOTION DETECTION ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: emo_primary (Anger / Trust / Pride / Hope / Fear / Informational)
#            emo_secondary (secondary emotion co-occurrence)
#  Supporting: emo_primary_c, emo_secondary_c (confidence scores)
#  Subset: needs_review=False AND _insufficient=False → 19,790 classified rows
#
#  Run: python insight_03_emotion_detection.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 03: EMOTION DETECTION ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_clf = len(hc_clf)

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E1: Overall Emotion Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    dist, _ = compute_distribution(hc_clf['emo_primary'], EMO_ORDER)
    labels = EMO_ORDER
    counts = [dist[l]['count'] for l in labels]
    pcts   = [dist[l]['pct'] for l in labels]
    colors = [EMO_COLORS[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={'width_ratios': [1.3, 1]})
    fig.suptitle(format_title('E1', 'Overall Primary Emotion Distribution',
                              f'n = {n_clf:,} classified posts (SKIPPED/insufficient excluded)'),
                 fontsize=14, fontweight='bold', y=1.02)

    bars = ax1.bar(labels, pcts, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    add_value_labels(ax1, bars, fmt='{:.1f}%', counts=counts, fontsize=9)
    ax1.set_ylabel('Percentage of Posts (%)')
    ax1.set_xlabel('Primary Emotion')
    ax1.set_ylim(0, max(pcts) * 1.25)
    ax1.set_title('Distribution (Bar)', fontsize=11)

    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    ax2.set_title('Distribution (Pie)', fontsize=11)

    add_methodology(fig,
        f'Source: emo_primary column (6-class emotion classifier). '
        f'Subset: needs_review=False AND _insufficient=False (n={n_clf:,}). '
        f'Percentages = count of each emotion ÷ {n_clf:,} × 100. '
        f'Emotions: Anger, Trust, Pride, Hope, Fear, Informational.',
        y=-0.04)
    save_chart(fig, 'E1_overall_emotion_distribution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E2: Emotion by Account (Grouped Bars)
    # ═════════════════════════════════════════════════════════════════════════
    grp = compute_grouped_distribution(hc_clf, '_account', 'emo_primary', EMO_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('E2', 'Primary Emotion Distribution by Account',
                              f'How each account\'s emotional profile differs'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        acc_data = grp.get(acc, {})
        acc_total = acc_data.get('_total', 0)
        vals = [acc_data.get(e, {}).get('pct', 0) for e in EMO_ORDER]
        cnts = [acc_data.get(e, {}).get('count', 0) for e in EMO_ORDER]
        bars = ax.bar(EMO_ORDER, vals, color=[EMO_COLORS[e] for e in EMO_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.6)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={acc_total:,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=30)

    add_methodology(fig,
        f'Source: emo_primary × _account. Subset: classified posts (n={n_clf:,}). '
        f'Percentages = count per emotion ÷ total classified posts for that account × 100.',
        y=-0.10)
    save_chart(fig, 'E2_emotion_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E3: Emotion by Alliance
    # ═════════════════════════════════════════════════════════════════════════
    grp_al = compute_grouped_distribution(hc_clf, '_alliance', 'emo_primary', EMO_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(format_title('E3', 'Primary Emotion by Alliance',
                              f'YSRCP Alliance vs TDP Alliance'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        al_data  = grp_al.get(al, {})
        al_total = al_data.get('_total', 0)
        vals = [al_data.get(e, {}).get('pct', 0) for e in EMO_ORDER]
        cnts = [al_data.get(e, {}).get('count', 0) for e in EMO_ORDER]
        bars = ax.bar(EMO_ORDER, vals, color=[EMO_COLORS[e] for e in EMO_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{al}\n(n={al_total:,})', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=20)

    add_methodology(fig,
        f'Source: emo_primary × _alliance. Alliance: YSRCP+YS Jagan vs TDP+NCBN. '
        f'Percentages = count per emotion ÷ total classified posts for that alliance × 100.',
        y=-0.06)
    save_chart(fig, 'E3_emotion_by_alliance.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E4: Emotion by Timeline
    # ═════════════════════════════════════════════════════════════════════════
    grp_tl = compute_grouped_distribution(hc_clf, '_timeline', 'emo_primary', EMO_ORDER)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('E4', 'Primary Emotion Evolution Across Timelines',
                              f'How the emotional profile of political discourse shifts over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    n_emo = len(EMO_ORDER)
    width = 0.12
    offset_start = -(n_emo - 1) * width / 2

    for j, emo in enumerate(EMO_ORDER):
        vals = [grp_tl.get(tl, {}).get(emo, {}).get('pct', 0) for tl in TL_ORDER]
        cnts = [grp_tl.get(tl, {}).get(emo, {}).get('count', 0) for tl in TL_ORDER]
        bars = ax.bar(x + offset_start + j * width, vals, width, label=emo,
                      color=EMO_COLORS[emo], edgecolor='white', linewidth=0.3)
        add_value_labels(ax, bars, fmt='{:.0f}%', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{TL_LABELS[tl]}\n(n={grp_tl.get(tl, {}).get("_total", 0):,})'
                        for tl in TL_ORDER])
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 50)
    ax.legend(title='Emotion', ncol=3, fontsize=8, title_fontsize=9, framealpha=0.9)

    add_methodology(fig,
        f'Source: emo_primary × _timeline. Classified posts only. '
        f'Percentages = count per emotion per timeline ÷ total classified posts in that timeline × 100.',
        y=-0.04)
    save_chart(fig, 'E4_emotion_by_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E5: Secondary Emotion Co-occurrence Heatmap
    # ═════════════════════════════════════════════════════════════════════════
    # Filter to rows with a valid emo_secondary
    valid_emos = ['Anger', 'Pride', 'Fear', 'Hope', 'Trust']
    has_sec = hc_clf[hc_clf['emo_secondary'].isin(valid_emos)].copy()

    ct = pd.crosstab(has_sec['emo_primary'], has_sec['emo_secondary'])
    # Reorder
    emo_rows = [e for e in EMO_ORDER if e in ct.index]
    emo_cols = [e for e in valid_emos if e in ct.columns]
    ct = ct.reindex(index=emo_rows, columns=emo_cols, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(format_title('E5', 'Primary × Secondary Emotion Co-occurrence',
                              f'n = {len(has_sec):,} posts with a valid secondary emotion'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(emo_cols)))
    ax.set_xticklabels([f'2nd: {e}' for e in emo_cols], rotation=30, ha='right')
    ax.set_yticks(range(len(emo_rows)))
    ax.set_yticklabels([f'1st: {e}' for e in emo_rows])
    ax.set_xlabel('Secondary Emotion')
    ax.set_ylabel('Primary Emotion')

    for r in range(len(emo_rows)):
        for c in range(len(emo_cols)):
            val = ct.values[r, c]
            color = 'white' if val > ct.values.max() * 0.55 else 'black'
            ax.text(c, r, f'{val:,}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Number of Posts')

    add_methodology(fig,
        f'Source: emo_primary × emo_secondary. Only posts with a recognizable secondary emotion '
        f'(Anger/Pride/Fear/Hope/Trust) are included (n={len(has_sec):,}). '
        f'{len(hc_clf) - len(has_sec):,} posts have NaN or no secondary emotion and are excluded. '
        f'Cell values = raw counts of the primary-secondary emotion pair.',
        y=-0.06)
    save_chart(fig, 'E5_emotion_cooccurrence_heatmap.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E6: Mean Engagement Rate by Emotion
    # ═════════════════════════════════════════════════════════════════════════
    emo_er = compute_mean_engagement(hc_clf, 'emo_primary', 'Engagement_Rate_Monthly')

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('E6', 'Mean Engagement Rate by Primary Emotion',
                              f'Which emotions drive the most engagement?'),
                 fontsize=14, fontweight='bold', y=1.02)

    emo_sorted = sorted(EMO_ORDER, key=lambda e: emo_er.get(e, {}).get('mean', 0), reverse=True)
    vals = [emo_er.get(e, {}).get('mean', 0) * 100 for e in emo_sorted]
    cnts = [emo_er.get(e, {}).get('count', 0) for e in emo_sorted]
    bars = ax.bar(emo_sorted, vals, color=[EMO_COLORS[e] for e in emo_sorted],
                  edgecolor='white', linewidth=0.8, width=0.55)
    add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=9)
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.set_xlabel('Primary Emotion')
    ax.set_ylim(0, max(vals) * 1.3)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly column grouped by emo_primary. '
        f'Mean ER = mean(Engagement_Rate_Monthly) per emotion × 100 to show as %. '
        f'n-counts shown = number of posts with that emotion. Classified posts only (n={n_clf:,}).',
        y=-0.04)
    save_chart(fig, 'E6_engagement_by_emotion.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E7: Emotion by Platform
    # ═════════════════════════════════════════════════════════════════════════
    grp_pl = compute_grouped_distribution(hc_clf, '_platform', 'emo_primary', EMO_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    fig.suptitle(format_title('E7', 'Primary Emotion by Platform',
                              f'Facebook vs Instagram emotional profiles'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, plat in enumerate(PLAT_ORDER):
        ax = axes[i]
        pl_data  = grp_pl.get(plat, {})
        pl_total = pl_data.get('_total', 0)
        vals = [pl_data.get(e, {}).get('pct', 0) for e in EMO_ORDER]
        cnts = [pl_data.get(e, {}).get('count', 0) for e in EMO_ORDER]
        bars = ax.bar(EMO_ORDER, vals, color=[EMO_COLORS[e] for e in EMO_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{plat}\n(n={pl_total:,})', fontsize=12,
                     color=PLAT_COLORS[plat], fontweight='bold')
        ax.set_ylim(0, 50)
        if i == 0:
            ax.set_ylabel('Percentage (%)')
        ax.tick_params(axis='x', rotation=20)

    add_methodology(fig,
        f'Source: emo_primary × _platform. Classified posts only. '
        f'Percentages = count per emotion per platform ÷ total classified posts on that platform × 100.',
        y=-0.06)
    save_chart(fig, 'E7_emotion_by_platform.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E8: Emotion Confidence by Emotion Label
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('E8', 'Mean Emotion Classifier Confidence by Emotion',
                              f'Higher confidence = model more certain about the classification'),
                 fontsize=14, fontweight='bold', y=1.02)

    conf_vals = []
    conf_cnts = []
    for emo in EMO_ORDER:
        sub = hc_clf[hc_clf['emo_primary'] == emo]['emo_primary_c']
        conf_vals.append(sub.mean())
        conf_cnts.append(len(sub))

    bars = ax.bar(EMO_ORDER, conf_vals, color=[EMO_COLORS[e] for e in EMO_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.55)
    add_value_labels(ax, bars, fmt='{:.3f}', counts=conf_cnts, fontsize=9)
    ax.set_ylabel('Mean Confidence Score')
    ax.set_xlabel('Primary Emotion')
    ax.set_ylim(0, max(conf_vals) * 1.2 if conf_vals else 1)

    add_methodology(fig,
        f'Source: emo_primary_c column (confidence score for the primary emotion assignment). '
        f'Value = mean(emo_primary_c) per emo_primary label. '
        f'Higher values indicate the classifier was more certain about assigning that emotion. '
        f'n-counts per emotion shown above bars.',
        y=-0.04)
    save_chart(fig, 'E8_emotion_confidence_by_label.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E9: Positive vs Negative Emotion Basket by Account
    # ═════════════════════════════════════════════════════════════════════════
    pos_basket = ['Pride', 'Trust', 'Hope']
    neg_basket = ['Anger', 'Fear']

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('E9', 'Positive vs Negative Emotion Basket by Account',
                              f'Positive = Pride+Trust+Hope  |  Negative = Anger+Fear  |  Neutral = Informational'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(ACC_ORDER))
    width = 0.25

    for j, (basket_name, basket_emos, bcolor) in enumerate([
        ('Positive Emotions', pos_basket, '#2ECC71'),
        ('Negative Emotions', neg_basket, '#E74C3C'),
        ('Informational', ['Informational'], '#95A5A6'),
    ]):
        vals = []
        cnts = []
        for acc in ACC_ORDER:
            sub = hc_clf[hc_clf['_account'] == acc]
            total = len(sub)
            c = sub['emo_primary'].isin(basket_emos).sum()
            vals.append(pct(c, total))
            cnts.append(c)
        bars = ax.bar(x + j * width, vals, width, label=basket_name,
                      color=bcolor, edgecolor='white', linewidth=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=7)

    ax.set_xticks(x + width)
    acc_labels_n = [f'{ACC_LABELS[a]}\n(n={len(hc_clf[hc_clf["_account"]==a]):,})'
                    for a in ACC_ORDER]
    ax.set_xticklabels(acc_labels_n)
    ax.set_ylabel('Percentage of Posts (%)')
    ax.set_ylim(0, 70)
    ax.legend(fontsize=9, framealpha=0.9)

    add_methodology(fig,
        f'Source: emo_primary column grouped into baskets. '
        f'Positive basket = Pride + Trust + Hope. Negative basket = Anger + Fear. '
        f'Percentages = count of emotions in basket ÷ total classified posts per account × 100. '
        f'Informational shown separately as it is affectively neutral.',
        y=-0.04)
    save_chart(fig, 'E9_emotion_basket_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART E10: Emotion Trajectory per Account Across Timelines
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('E10', 'Emotion Trajectory per Account Across Timelines',
                              f'How each account\'s emotional tone shifts over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_clf[hc_clf['_account'] == acc]
        x = np.arange(len(TL_ORDER))
        n_emos = len(EMO_ORDER)
        w = 0.12
        off = -(n_emos - 1) * w / 2

        for j, emo in enumerate(EMO_ORDER):
            vals = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                total = len(tl_sub)
                c = (tl_sub['emo_primary'] == emo).sum()
                vals.append(pct(c, total))
            bars = ax.bar(x + off + j * w, vals, w, label=emo if i == 0 else '',
                          color=EMO_COLORS[emo], edgecolor='white', linewidth=0.3)
            for bar in bars:
                h = bar.get_height()
                if h > 3:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                            f'{h:.0f}%', ha='center', fontsize=5, color='#333')

        ax.set_xticks(x)
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        acc_n = len(sub)
        ax.set_title(f'{ACC_LABELS[acc]} (n={acc_n:,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Emotion', fontsize=6, title_fontsize=7, ncol=2, loc='upper left')

    add_methodology(fig,
        f'Source: emo_primary × _timeline × _account. Classified posts only. '
        f'Each bar = percentage of that emotion within that account-timeline combination. '
        f'Percentage = count ÷ total classified posts for that account in that timeline × 100.',
        y=-0.10)
    save_chart(fig, 'E10_emotion_trajectory_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    overall_emo = compute_distribution(hc_clf['emo_primary'], EMO_ORDER)[0]
    by_acc_emo  = compute_grouped_distribution(hc_clf, '_account', 'emo_primary', EMO_ORDER)
    by_al_emo   = compute_grouped_distribution(hc_clf, '_alliance', 'emo_primary', EMO_ORDER)
    by_tl_emo   = compute_grouped_distribution(hc_clf, '_timeline', 'emo_primary', EMO_ORDER)

    sec_insights = []

    # Insight 3.1
    sec_insights.append((
        'INSIGHT 3.1',
        'Overall Primary Emotion Distribution',
        f'Among {n_clf:,} classified posts, the primary emotion distribution is:\n\n'
        + '\n'.join([f'    • {e}: {overall_emo[e]["count"]:,} ({overall_emo[e]["pct"]:.1f}%)'
                     for e in EMO_ORDER]) + '\n\n'
        f'  Anger is the dominant emotion across AP political social media, comprising over '
        f'one-third of all classified posts. This is followed by Trust and Pride — '
        f'suggesting a discourse landscape polarised between attack/outrage content and '
        f'leader-loyalty/achievement-celebration content. Fear is rare ({overall_emo["Fear"]["pct"]:.1f}%), '
        f'indicating that fear-based messaging is not a primary tool in AP politics. '
        f'Informational posts ({overall_emo["Informational"]["pct"]:.1f}%) represent factual/news-style '
        f'content without strong emotional valence.',
        f'Counts of emo_primary per label ÷ {n_clf:,} (total classified posts) × 100. '
        f'Classifier: fine-tuned emotion detection model with 6-class output. '
        f'SKIPPED/_insufficient posts excluded.'
    ))

    # Insight 3.2 — By account
    acc_emo_lines = []
    for acc in ACC_ORDER:
        d = by_acc_emo.get(acc, {})
        dominant = max(EMO_ORDER, key=lambda e: d.get(e, {}).get('pct', 0))
        acc_emo_lines.append(
            f'    • {ACC_LABELS[acc]} (n={d.get("_total", 0):,}): '
            f'Dominant = {dominant} ({d.get(dominant, {}).get("pct", 0):.1f}%) | '
            f'Anger={d.get("Anger", {}).get("pct", 0):.1f}%, '
            f'Trust={d.get("Trust", {}).get("pct", 0):.1f}%, '
            f'Pride={d.get("Pride", {}).get("pct", 0):.1f}%, '
            f'Hope={d.get("Hope", {}).get("pct", 0):.1f}%'
        )

    most_anger = max(ACC_ORDER, key=lambda a: by_acc_emo.get(a, {}).get('Anger', {}).get('pct', 0))
    most_trust = max(ACC_ORDER, key=lambda a: by_acc_emo.get(a, {}).get('Trust', {}).get('pct', 0))
    most_hope  = max(ACC_ORDER, key=lambda a: by_acc_emo.get(a, {}).get('Hope', {}).get('pct', 0))

    sec_insights.append((
        'INSIGHT 3.2',
        'Emotion Profile by Account',
        f'Per-account emotion breakdown:\n\n'
        + '\n'.join(acc_emo_lines) + '\n\n'
        f'  {ACC_LABELS[most_anger]} has the highest anger percentage, '
        f'{ACC_LABELS[most_trust]} leads in trust, and {ACC_LABELS[most_hope]} in hope. '
        f'These profiles reveal distinct emotional strategies: some accounts lead with '
        f'outrage-driven content (attack messaging), while others emphasise loyalty/'
        f'achievement narratives.',
        f'emo_primary counts per _account ÷ total classified posts for that account × 100.'
    ))

    # Insight 3.3 — Anger rate comparison
    anger_rates = {acc: by_acc_emo.get(acc, {}).get('Anger', {}).get('pct', 0) for acc in ACC_ORDER}
    anger_sorted = sorted(ACC_ORDER, key=lambda a: anger_rates[a], reverse=True)

    sec_insights.append((
        'INSIGHT 3.3',
        'Anger Rate Ranking — Who Attacks Most?',
        f'Anger as a percentage of classified posts, ranked:\n\n'
        + '\n'.join([f'    {i+1}. {ACC_LABELS[a]}: {anger_rates[a]:.1f}% '
                     f'({by_acc_emo.get(a, {}).get("Anger", {}).get("count", 0):,} posts)'
                     for i, a in enumerate(anger_sorted)]) + '\n\n'
        f'  The spread between highest and lowest anger rate is '
        f'{anger_rates[anger_sorted[0]] - anger_rates[anger_sorted[-1]]:.1f} percentage points. '
        f'In political communication research, high anger content is typically associated with '
        f'opposition/attack strategies designed to mobilise discontented voters.',
        f'Count of emo_primary="Anger" per account ÷ total classified posts per account × 100.'
    ))

    # Insight 3.4 — Positive emotion basket
    for acc in ACC_ORDER:
        sub = hc_clf[hc_clf['_account'] == acc]
        pos_c = sub['emo_primary'].isin(pos_basket).sum()
        neg_c = sub['emo_primary'].isin(neg_basket).sum()

    basket_lines = []
    for acc in ACC_ORDER:
        sub = hc_clf[hc_clf['_account'] == acc]
        pos_c = sub['emo_primary'].isin(pos_basket).sum()
        neg_c = sub['emo_primary'].isin(neg_basket).sum()
        total = len(sub)
        basket_lines.append(
            f'    • {ACC_LABELS[acc]}: Positive basket={pct(pos_c, total):.1f}% (n={pos_c:,}), '
            f'Negative basket={pct(neg_c, total):.1f}% (n={neg_c:,}), '
            f'Ratio={pos_c/neg_c:.2f}' if neg_c > 0 else
            f'    • {ACC_LABELS[acc]}: Positive basket={pct(pos_c, total):.1f}%, Negative basket=0'
        )

    sec_insights.append((
        'INSIGHT 3.4',
        'Positive vs Negative Emotion Basket Analysis',
        f'Positive emotion basket (Pride + Trust + Hope) vs Negative (Anger + Fear):\n\n'
        + '\n'.join(basket_lines) + '\n\n'
        f'  A ratio > 1.0 means the account\'s posts have more positive than negative '
        f'emotions; < 1.0 means negative emotions dominate. This basket analysis provides '
        f'a simplified emotional polarity metric for each account.',
        f'Positive basket = count of emo_primary in (Pride, Trust, Hope) per account. '
        f'Negative basket = count of emo_primary in (Anger, Fear) per account. '
        f'Ratio = positive count ÷ negative count. Informational excluded from both baskets.'
    ))

    # Insight 3.5 — Fear analysis
    fear_data = hc_clf[hc_clf['emo_primary'] == 'Fear']
    fear_by_acc = fear_data['_account'].value_counts()

    sec_insights.append((
        'INSIGHT 3.5',
        'Fear Analysis — Rare but Significant',
        f'Fear appears in only {len(fear_data):,} posts ({pct(len(fear_data), n_clf):.1f}% of classified posts), '
        f'making it the rarest primary emotion.\n\n'
        f'  Fear distribution by account:\n'
        + '\n'.join([f'    • {ACC_LABELS.get(acc, acc)}: {count:,} posts'
                     for acc, count in fear_by_acc.items()]) + '\n\n'
        f'  Fear distribution by timeline:\n'
        + '\n'.join([f'    • {TL_LABELS[tl]}: {(fear_data["_timeline"]==tl).sum():,} posts'
                     for tl in TL_ORDER]) + '\n\n'
        f'  Despite its rarity, fear-based content can disproportionately affect voter '
        f'perceptions. In AP politics, the low fear rate suggests the discourse is more '
        f'confrontational (anger) than anxiety-driven (fear).',
        f'Count of emo_primary="Fear" = {len(fear_data):,}. Broken down by _account and _timeline. '
        f'Percentage = fear count ÷ {n_clf:,} × 100.'
    ))

    # Insight 3.6 — Informational posts
    info_data = hc_clf[hc_clf['emo_primary'] == 'Informational']
    info_rates = {acc: pct((info_data['_account'] == acc).sum(),
                           len(hc_clf[hc_clf['_account'] == acc]))
                  for acc in ACC_ORDER}

    sec_insights.append((
        'INSIGHT 3.6',
        'Informational Posts — Emotionally Neutral Content',
        f'Informational posts ({len(info_data):,}, {pct(len(info_data), n_clf):.1f}%) lack strong '
        f'emotional valence and represent factual/news-style content.\n\n'
        f'  Per-account Informational rates:\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {info_rates[a]:.1f}%' for a in ACC_ORDER]) + '\n\n'
        f'  The account with the highest informational rate may be positioning itself as '
        f'a credible news source rather than an emotional mobiliser.',
        f'Count of emo_primary="Informational" per account ÷ total classified posts per account × 100.'
    ))

    # Insight 3.7 — Secondary emotion co-occurrence
    if len(has_sec) > 0:
        top_pairs = ct.stack().sort_values(ascending=False).head(5)
        pair_lines = [f'    • {p}: {v:,} posts' for p, v in top_pairs.items()]
    else:
        pair_lines = ['    (No valid secondary emotions found)']

    sec_insights.append((
        'INSIGHT 3.7',
        'Top Secondary Emotion Co-occurrences',
        f'Among {len(has_sec):,} posts with a valid secondary emotion, '
        f'the most common primary-secondary pairings are:\n\n'
        + '\n'.join(pair_lines) + '\n\n'
        f'  These pairings reveal the emotional "texture" of posts. For instance, '
        f'an Anger-Fear pairing suggests anxiety-laden critique, while an Anger-Hope '
        f'pairing might represent frustrated optimism.',
        f'Cross-tab of emo_primary × emo_secondary for posts where emo_secondary is one of '
        f'(Anger, Pride, Fear, Hope, Trust). Posts with NaN secondary emotion excluded. '
        f'Top 5 pairs by raw count.'
    ))

    # Insight 3.8 — Engagement correlation
    emo_engagement = {e: emo_er.get(e, {}).get('mean', 0) for e in EMO_ORDER}
    top_eng_emo = max(EMO_ORDER, key=lambda e: emo_engagement[e])
    low_eng_emo = min(EMO_ORDER, key=lambda e: emo_engagement[e])

    sec_insights.append((
        'INSIGHT 3.8',
        'Emotion–Engagement Relationship',
        f'Mean Engagement Rate (Monthly) by primary emotion:\n\n'
        + '\n'.join([f'    • {e}: {emo_engagement[e]*100:.4f}% '
                     f'(n={emo_er.get(e, {}).get("count", 0):,})'
                     for e in sorted(EMO_ORDER, key=lambda e: emo_engagement[e], reverse=True)])
        + f'\n\n  {top_eng_emo} posts receive the highest mean engagement rate, while '
        f'{low_eng_emo} receives the lowest. '
        f'The difference is {(emo_engagement[top_eng_emo] - emo_engagement[low_eng_emo])*100:.4f} '
        f'percentage points. This indicates which emotional appeals resonate most '
        f'with social media audiences in AP political discourse.',
        f'Mean of Engagement_Rate_Monthly per emo_primary label. '
        f'Engagement_Rate_Monthly is a pre-computed metric. '
        f'Values shown as percentages (×100). Classified posts only.'
    ))

    # Insight 3.9 — Alliance comparison
    ysrcp_emo = by_al_emo.get('YSRCP Alliance', {})
    tdp_emo   = by_al_emo.get('TDP Alliance', {})

    sec_insights.append((
        'INSIGHT 3.9',
        'Alliance Emotion Comparison',
        f'YSRCP Alliance (n={ysrcp_emo.get("_total", 0):,}):\n'
        + '\n'.join([f'    {e}: {ysrcp_emo.get(e, {}).get("pct", 0):.1f}%' for e in EMO_ORDER])
        + f'\n\n  TDP Alliance (n={tdp_emo.get("_total", 0):,}):\n'
        + '\n'.join([f'    {e}: {tdp_emo.get(e, {}).get("pct", 0):.1f}%' for e in EMO_ORDER])
        + f'\n\n  The key emotional divergence between the two alliances lies in their '
        f'anger-to-trust balance. A higher anger ratio suggests a more attack-oriented '
        f'campaign strategy, while higher trust/hope suggests a governance-showcase approach.',
        f'emo_primary counts per _alliance (YSRCP+YS Jagan vs TDP+NCBN) ÷ total classified '
        f'posts per alliance × 100.'
    ))

    # Insight 3.10 — Temporal dynamics
    first_anger = by_tl_emo.get(TL_ORDER[0], {}).get('Anger', {}).get('pct', 0)
    last_anger  = by_tl_emo.get(TL_ORDER[-1], {}).get('Anger', {}).get('pct', 0)

    sec_insights.append((
        'INSIGHT 3.10',
        'Conclusion — Emotion Landscape',
        f'The AP political emotion landscape is characterised by:\n\n'
        f'  1. Anger dominance at {overall_emo["Anger"]["pct"]:.1f}% — the most common emotion.\n'
        f'  2. Significant inter-account variation in emotional profiles.\n'
        f'  3. Fear is rare ({overall_emo["Fear"]["pct"]:.1f}%), suggesting confrontation over anxiety.\n'
        f'  4. Anger shows {"increasing" if last_anger > first_anger else "decreasing"} trend '
        f'from {first_anger:.1f}% to {last_anger:.1f}% across the study period.\n'
        f'  5. {top_eng_emo} posts generate the highest engagement, suggesting that '
        f'audiences respond most to {top_eng_emo.lower()}-based content.\n'
        f'  6. Complex emotional textures exist in secondary emotion pairings.\n\n'
        f'  These findings establish the emotional underpinning of political discourse in AP, '
        f'revealing that attack-based (anger) messaging is the dominant strategy, but '
        f'trust-building and pride-based content also plays a significant role for certain accounts.',
        f'Synthesis of insights 3.1–3.9. No new calculations.'
    ))

    all_sections.append(('Emotion Detection Analysis', sec_insights))

    write_insight_report('03_emotion_detection_insights.txt',
                         'EMOTION DETECTION ANALYSIS', all_sections)

    print(f"\n  Done! Generated 10 charts + 1 text report (10 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

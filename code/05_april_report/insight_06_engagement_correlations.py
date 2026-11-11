# ============================================================
#  SCRIPT 06 — ENGAGEMENT × SENTIMENT CORRELATIONS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: How sentiment, emotion, and tone relate to engagement
#  Metrics: likes, comments_count, shares, views, Engagement_Rate_Monthly
#  Subset: needs_review=False → 21,909 rows (for sentiment)
#           + classified subset for tone/emotion
#
#  Run: python insight_06_engagement_correlations.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 06: ENGAGEMENT × SENTIMENT CORRELATIONS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)
    n_clf = len(hc_clf)

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC1: Mean Engagement Rate by Caption Sentiment
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('EC1', 'Mean Engagement Rate by Caption Sentiment',
                              f'Does sentiment polarity affect engagement? (n={n_total:,})'),
                 fontsize=14, fontweight='bold', y=1.02)

    er_sent = compute_mean_engagement(hc, 'sentiment_label', 'Engagement_Rate_Monthly')
    vals = [er_sent.get(s, {}).get('mean', 0) * 100 for s in SENT_ORDER]
    cnts = [er_sent.get(s, {}).get('count', 0) for s in SENT_ORDER]
    bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=10)
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.set_ylim(0, max(vals) * 1.3)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly grouped by sentiment_label. '
        f'Mean ER = mean(Engagement_Rate_Monthly) per sentiment × 100. '
        f'All {n_total:,} HC posts included.',
        y=-0.04)
    save_chart(fig, 'EC1_engagement_by_caption_sentiment.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC2: Multi-Metric Engagement by Sentiment (Likes, Comments, Shares, Views)
    # ═════════════════════════════════════════════════════════════════════════
    metrics = ['likes', 'comments_count', 'shares', 'views']
    metric_labels = ['Likes', 'Comments', 'Shares', 'Views']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    fig.suptitle(format_title('EC2', 'Mean Engagement Metrics by Caption Sentiment',
                              f'Likes, Comments, Shares, Views breakdown'),
                 fontsize=14, fontweight='bold', y=1.02)

    for idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        vals = []
        cnts = []
        for s in SENT_ORDER:
            sub = hc[hc['sentiment_label'] == s]
            vals.append(sub[metric].mean())
            cnts.append(len(sub))
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:,.0f}', counts=cnts if idx == 0 else None, fontsize=8)
        ax.set_title(f'Mean {mlabel}', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Mean Count')

    add_methodology(fig,
        f'Source: likes, comments_count, shares, views columns grouped by sentiment_label. '
        f'Values = mean of each metric per sentiment label. All HC posts (n={n_total:,}).',
        y=-0.06)
    save_chart(fig, 'EC2_multi_metric_by_sentiment.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC3: Mean ER by Sentiment × Account (Faceted)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('EC3', 'Mean Engagement Rate by Sentiment × Account',
                              f'How each account\'s engagement varies by sentiment'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]
        vals = []
        cnts = []
        for s in SENT_ORDER:
            s_sub = sub[sub['sentiment_label'] == s]
            vals.append(s_sub['Engagement_Rate_Monthly'].mean() * 100)
            cnts.append(len(s_sub))
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean ER (%)')

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × sentiment_label × _account. '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 for each sentiment-account combination.',
        y=-0.06)
    save_chart(fig, 'EC3_engagement_sentiment_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC4: Mean ER by Emotion × Account (Faceted)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('EC4', 'Mean Engagement Rate by Emotion × Account',
                              f'Which emotions get the best engagement per account?'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_clf[hc_clf['_account'] == acc]
        vals = []
        cnts = []
        for e in EMO_ORDER:
            e_sub = sub[sub['emo_primary'] == e]
            vals.append(e_sub['Engagement_Rate_Monthly'].mean() * 100 if len(e_sub) > 0 else 0)
            cnts.append(len(e_sub))
        bars = ax.bar(EMO_ORDER, vals, color=[EMO_COLORS[e] for e in EMO_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.6)
        add_value_labels(ax, bars, fmt='{:.3f}%', fontsize=6)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean ER (%)')
        ax.tick_params(axis='x', rotation=30, labelsize=7)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × emo_primary × _account. '
        f'Classified posts only (n={n_clf:,}). '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 for each emotion-account pair.',
        y=-0.10)
    save_chart(fig, 'EC4_engagement_emotion_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC5: Mean ER by Tone × Account (Faceted)
    # ═════════════════════════════════════════════════════════════════════════
    hc_tone = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]
    tone_display = [t.replace('_', ' ') for t in TONE_ORDER]

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('EC5', 'Mean Engagement Rate by Tone × Account',
                              f'Which rhetorical approach drives engagement per account?'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_tone[hc_tone['_account'] == acc]
        vals = []
        cnts = []
        for t in TONE_ORDER:
            t_sub = sub[sub['tone_primary'] == t]
            vals.append(t_sub['Engagement_Rate_Monthly'].mean() * 100 if len(t_sub) > 0 else 0)
            cnts.append(len(t_sub))
        bars = ax.bar(tone_display, vals, color=[TONE_COLORS[t] for t in TONE_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.6)
        add_value_labels(ax, bars, fmt='{:.3f}%', fontsize=6)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean ER (%)')
        ax.tick_params(axis='x', rotation=25, labelsize=7)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × tone_primary × _account. '
        f'Classified posts, Neutral-tone excluded. '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 for each tone-account pair.',
        y=-0.10)
    save_chart(fig, 'EC5_engagement_tone_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC6: Top 20 Highest-Engagement Posts — Sentiment/Emotion/Tone
    # ═════════════════════════════════════════════════════════════════════════
    top20 = hc.nlargest(20, 'Engagement_Rate_Monthly').copy()

    # Summarise the breakdown
    top_sent = top20['sentiment_label'].value_counts()
    top_emo  = top20['emo_primary'].value_counts()
    top_tone = top20['tone_primary'].value_counts()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(format_title('EC6', 'Top 20 Highest-Engagement Posts — Classification Breakdown',
                              f'What sentiment/emotion/tone do viral posts have?'),
                 fontsize=14, fontweight='bold', y=1.02)

    # Sentiment
    ax = axes[0]
    s_labels = [s for s in SENT_ORDER if s in top_sent.index]
    s_vals   = [top_sent.get(s, 0) for s in s_labels]
    s_colors = [SENT_COLORS[s] for s in s_labels]
    bars = ax.bar(s_labels, s_vals, color=s_colors, edgecolor='white', width=0.5)
    add_value_labels(ax, bars, fmt='{:.0f}', fontsize=10)
    ax.set_title('Caption Sentiment', fontsize=11)
    ax.set_ylabel('Number of Posts')
    ax.set_ylim(0, max(s_vals) * 1.4 if s_vals else 5)

    # Emotion
    ax = axes[1]
    e_labels = [e for e in EMO_ORDER if e in top_emo.index]
    e_vals   = [top_emo.get(e, 0) for e in e_labels]
    e_colors = [EMO_COLORS[e] for e in e_labels]
    bars = ax.bar(e_labels, e_vals, color=e_colors, edgecolor='white', width=0.5)
    add_value_labels(ax, bars, fmt='{:.0f}', fontsize=10)
    ax.set_title('Primary Emotion', fontsize=11)
    ax.tick_params(axis='x', rotation=20)

    # Tone
    ax = axes[2]
    t_labels = [t for t in TONE_ORDER if t in top_tone.index]
    t_vals   = [top_tone.get(t, 0) for t in t_labels]
    t_colors = [TONE_COLORS[t] for t in t_labels]
    bars = ax.bar([t.replace('_', ' ') for t in t_labels], t_vals,
                  color=t_colors, edgecolor='white', width=0.5)
    add_value_labels(ax, bars, fmt='{:.0f}', fontsize=10)
    ax.set_title('Rhetorical Tone', fontsize=11)
    ax.tick_params(axis='x', rotation=15)

    add_methodology(fig,
        f'Source: Top 20 posts by Engagement_Rate_Monthly (highest ER). '
        f'ER range of these 20 posts: [{top20["Engagement_Rate_Monthly"].min():.4f}, '
        f'{top20["Engagement_Rate_Monthly"].max():.4f}]. '
        f'Bars = count of each label among the top 20 posts.',
        y=-0.06)
    save_chart(fig, 'EC6_top20_posts_breakdown.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC7: Mean Likes by Sentiment × Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('EC7', 'Mean Likes by Caption Sentiment × Account',
                              f'Raw popularity metric by sentiment for each account'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]
        vals = []
        cnts = []
        for s in SENT_ORDER:
            s_sub = sub[sub['sentiment_label'] == s]
            vals.append(s_sub['likes'].mean())
            cnts.append(len(s_sub))
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:,.0f}', counts=cnts, fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean Likes')

    add_methodology(fig,
        f'Source: likes column × sentiment_label × _account. '
        f'Values = mean(likes) per sentiment per account. '
        f'Unlike ER, likes is a raw count not normalised by audience size.',
        y=-0.06)
    save_chart(fig, 'EC7_likes_by_sentiment_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC8: Engagement Rate by Sentiment × Timeline
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('EC8', 'Mean Engagement Rate by Sentiment × Timeline',
                              f'Does sentiment-engagement relationship change over time?'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    width = 0.25

    for j, sent in enumerate(SENT_ORDER):
        vals = []
        cnts = []
        for tl in TL_ORDER:
            sub = hc[(hc['_timeline'] == tl) & (hc['sentiment_label'] == sent)]
            vals.append(sub['Engagement_Rate_Monthly'].mean() * 100)
            cnts.append(len(sub))
        bars = ax.bar(x + j * width, vals, width, label=sent,
                      color=SENT_COLORS[sent], edgecolor='white', linewidth=0.5)
        add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=7)

    ax.set_xticks(x + width)
    ax.set_xticklabels([TL_LABELS[tl] for tl in TL_ORDER])
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.legend(title='Sentiment')

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × sentiment_label × _timeline. '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 per sentiment-timeline pair.',
        y=-0.04)
    save_chart(fig, 'EC8_engagement_sentiment_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC9: Does Negativity Drive Engagement? (Comparative)
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('EC9', 'Does Negativity Drive Engagement?',
                              f'Negative vs Positive ER per account — the key question'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(ACC_ORDER))
    width = 0.3

    neg_ers = []
    pos_ers = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        neg_er = sub[sub['sentiment_label'] == 'Negative']['Engagement_Rate_Monthly'].mean() * 100
        pos_er = sub[sub['sentiment_label'] == 'Positive']['Engagement_Rate_Monthly'].mean() * 100
        neg_ers.append(neg_er)
        pos_ers.append(pos_er)

    bars1 = ax.bar(x - width/2, neg_ers, width, label='Negative', color=SENT_COLORS['Negative'],
                   edgecolor='white', linewidth=0.8)
    bars2 = ax.bar(x + width/2, pos_ers, width, label='Positive', color=SENT_COLORS['Positive'],
                   edgecolor='white', linewidth=0.8)

    add_value_labels(ax, bars1, fmt='{:.4f}%', fontsize=8)
    add_value_labels(ax, bars2, fmt='{:.4f}%', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([ACC_LABELS[a] for a in ACC_ORDER])
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.legend(title='Sentiment')

    # Add winner annotations
    for i, acc in enumerate(ACC_ORDER):
        winner = 'Negative' if neg_ers[i] > pos_ers[i] else 'Positive'
        diff = abs(neg_ers[i] - pos_ers[i])
        higher = max(neg_ers[i], pos_ers[i])
        ax.text(i, higher + higher * 0.05, f'⬆ {winner}\nΔ={diff:.4f}pp',
                ha='center', fontsize=7, color='#555555', style='italic')

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × sentiment_label × _account. '
        f'Compares mean ER for Negative vs Positive posts per account. '
        f'Δ = absolute difference in percentage points. '
        f'In political communication research, the "negativity bias" hypothesis posits '
        f'that negative content generates higher engagement.',
        y=-0.06)
    save_chart(fig, 'EC9_negativity_drives_engagement.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART EC10: Engagement by Subjectivity
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('EC10', 'Mean Engagement Rate by Subjectivity',
                              f'Do fact-based or opinion-based posts get more engagement?'),
                 fontsize=14, fontweight='bold', y=1.02)

    subj_er = compute_mean_engagement(hc_clf, 'subj_label', 'Engagement_Rate_Monthly')
    vals = [subj_er.get(s, {}).get('mean', 0) * 100 for s in SUBJ_ORDER]
    cnts = [subj_er.get(s, {}).get('count', 0) for s in SUBJ_ORDER]
    bars = ax.bar(SUBJ_ORDER, vals, color=[SUBJ_COLORS[s] for s in SUBJ_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.4f}%', counts=cnts, fontsize=10)
    ax.set_ylabel('Mean Engagement Rate (%)')
    ax.set_ylim(0, max(vals) * 1.3)

    add_methodology(fig,
        f'Source: Engagement_Rate_Monthly × subj_label. Classified posts (n={n_clf:,}). '
        f'Mean ER = mean(Engagement_Rate_Monthly) × 100 per subjectivity label.',
        y=-0.04)
    save_chart(fig, 'EC10_engagement_by_subjectivity.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    sec_insights = []

    # 6.1 Overall ER by sentiment
    er_by_sent = {s: er_sent.get(s, {}).get('mean', 0) for s in SENT_ORDER}
    top_sent_er = max(SENT_ORDER, key=lambda s: er_by_sent[s])
    low_sent_er = min(SENT_ORDER, key=lambda s: er_by_sent[s])

    sec_insights.append((
        'INSIGHT 6.1',
        'Engagement Rate by Caption Sentiment — Overall',
        f'Mean Engagement Rate (Monthly) by caption sentiment:\n\n'
        + '\n'.join([f'    • {s}: {er_by_sent[s]*100:.4f}% (n={er_sent.get(s, {}).get("count", 0):,})'
                     for s in SENT_ORDER]) + '\n\n'
        f'  {top_sent_er} posts generate the highest mean engagement rate, while '
        f'{low_sent_er} generates the lowest. The difference between highest and lowest '
        f'is {(er_by_sent[top_sent_er] - er_by_sent[low_sent_er])*100:.4f} percentage points.\n\n'
        f'  This addresses a fundamental question in political communication: does '
        f'negativity drive engagement? The data shows that {top_sent_er.lower()} content '
        f'is the most engaging, {"confirming the negativity bias hypothesis" if top_sent_er == "Negative" else "challenging the negativity bias hypothesis"}.',
        f'Mean of Engagement_Rate_Monthly per sentiment_label. '
        f'Engagement_Rate_Monthly is pre-computed as total engagements ÷ estimated reach.'
    ))

    # 6.2 Multi-metric
    metric_data = {}
    for metric in metrics:
        metric_data[metric] = {s: hc[hc['sentiment_label']==s][metric].mean() for s in SENT_ORDER}

    metric_lines = []
    for metric, mlabel in zip(metrics, metric_labels):
        vals_str = ', '.join([f'{s}={metric_data[metric][s]:,.0f}' for s in SENT_ORDER])
        top_m = max(SENT_ORDER, key=lambda s: metric_data[metric][s])
        metric_lines.append(f'    • {mlabel}: {vals_str} → Highest: {top_m}')

    sec_insights.append((
        'INSIGHT 6.2',
        'Multi-Metric Engagement by Sentiment',
        f'Mean raw engagement metrics by caption sentiment:\n\n'
        + '\n'.join(metric_lines) + '\n\n'
        f'  While ER normalises for audience size, raw metrics (likes, comments, shares, views) '
        f'reveal absolute engagement volume. These may differ from ER rankings because '
        f'larger accounts naturally get more raw engagement.',
        f'Mean of likes, comments_count, shares, views per sentiment_label. '
        f'These are raw counts, not normalised for audience size.'
    ))

    # 6.3 Per-account negativity bias
    neg_bias_lines = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        neg_er_val = sub[sub['sentiment_label']=='Negative']['Engagement_Rate_Monthly'].mean() * 100
        pos_er_val = sub[sub['sentiment_label']=='Positive']['Engagement_Rate_Monthly'].mean() * 100
        diff = neg_er_val - pos_er_val
        bias = 'negativity bias' if diff > 0 else 'positivity advantage'
        neg_bias_lines.append(
            f'    • {ACC_LABELS[acc]}: Neg ER={neg_er_val:.4f}%, Pos ER={pos_er_val:.4f}%, '
            f'Δ={diff:+.4f}pp ({bias})'
        )

    sec_insights.append((
        'INSIGHT 6.3',
        'Negativity Bias Test per Account',
        f'Does negativity drive more engagement? Per-account comparison '
        f'(Negative ER − Positive ER):\n\n'
        + '\n'.join(neg_bias_lines) + '\n\n'
        f'  The negativity bias hypothesis is a key research question in political communication. '
        f'A positive Δ means negative posts get MORE engagement than positive ones; '
        f'a negative Δ means positive content is more engaging. This varies by account.',
        f'Mean(Engagement_Rate_Monthly) for Negative vs Positive posts per account. '
        f'Δ = Negative_ER − Positive_ER (both × 100).'
    ))

    # 6.4 Anger vs Hope engagement
    if 'Anger' in EMO_ORDER and 'Hope' in EMO_ORDER:
        anger_er = hc_clf[hc_clf['emo_primary']=='Anger']['Engagement_Rate_Monthly'].mean()
        hope_er  = hc_clf[hc_clf['emo_primary']=='Hope']['Engagement_Rate_Monthly'].mean()
        anger_n  = (hc_clf['emo_primary']=='Anger').sum()
        hope_n   = (hc_clf['emo_primary']=='Hope').sum()

        sec_insights.append((
            'INSIGHT 6.4',
            'Anger vs Hope — Which Emotion Gets More Engagement?',
            f'Anger posts: Mean ER = {anger_er*100:.4f}% (n={anger_n:,})\n'
            f'  Hope posts:  Mean ER = {hope_er*100:.4f}% (n={hope_n:,})\n'
            f'  Difference:  {(anger_er-hope_er)*100:+.4f}pp\n\n'
            f'  This comparison directly tests whether outrage-driven content ('
            f'anger) generates more engagement than aspirational content (hope). '
            f'{"Anger" if anger_er > hope_er else "Hope"} wins with a '
            f'{abs(anger_er-hope_er)*100:.4f}pp advantage. '
            f'This has direct implications for campaign strategy and content design.',
            f'Mean(Engagement_Rate_Monthly) for emo_primary="Anger" vs "Hope". '
            f'Classified posts only.'
        ))

    # 6.5 Inspirational vs Aggressive tone engagement
    insp_er = hc_tone[hc_tone['tone_primary']=='Inspirational']['Engagement_Rate_Monthly'].mean()
    agg_er  = hc_tone[hc_tone['tone_primary']=='Aggressive']['Engagement_Rate_Monthly'].mean()

    sec_insights.append((
        'INSIGHT 6.5',
        'Inspirational vs Aggressive Tone — Engagement Comparison',
        f'Inspirational tone: Mean ER = {insp_er*100:.4f}% (n={(hc_tone["tone_primary"]=="Inspirational").sum():,})\n'
        f'  Aggressive tone:   Mean ER = {agg_er*100:.4f}% (n={(hc_tone["tone_primary"]=="Aggressive").sum():,})\n'
        f'  Difference:        {(insp_er-agg_er)*100:+.4f}pp\n\n'
        f'  {"Inspirational" if insp_er > agg_er else "Aggressive"} content generates higher '
        f'engagement. This is a key strategic insight: even in a highly aggressive discourse '
        f'environment, the rhetorical approach that resonates most with audiences may '
        f'not be the most common one.',
        f'Mean(Engagement_Rate_Monthly) for tone_primary="Inspirational" vs "Aggressive". '
        f'Neutral-tone excluded.'
    ))

    # 6.6 Top posts profile
    top_ER_range = f'{top20["Engagement_Rate_Monthly"].min():.4f} – {top20["Engagement_Rate_Monthly"].max():.4f}'
    top_sent_dist = top20['sentiment_label'].value_counts().to_dict()
    top_acc_dist  = top20['_account'].value_counts().to_dict()

    sec_insights.append((
        'INSIGHT 6.6',
        'Profile of Top 20 Highest-Engagement Posts',
        f'The 20 posts with the highest Engagement Rate Monthly '
        f'(ER range: {top_ER_range}):\n\n'
        f'  Sentiment: {", ".join([f"{k}={v}" for k, v in top_sent_dist.items()])}\n'
        f'  Accounts:  {", ".join([f"{ACC_LABELS.get(k, k)}={v}" for k, v in top_acc_dist.items()])}\n\n'
        f'  The viral/high-engagement posts tend to cluster around particular sentiments and '
        f'accounts. This profile reveals who creates the most engaging content and what '
        f'emotional/tonal characteristics define viral political posts in AP.',
        f'Top 20 rows by Engagement_Rate_Monthly (descending). '
        f'Breakdown by sentiment_label, emo_primary, tone_primary, _account.'
    ))

    # 6.7 Engagement by subjectivity
    subj_er_data = {s: subj_er.get(s, {}).get('mean', 0) for s in SUBJ_ORDER}
    top_subj = max(SUBJ_ORDER, key=lambda s: subj_er_data[s])

    sec_insights.append((
        'INSIGHT 6.7',
        'Engagement by Subjectivity — Facts vs Opinions',
        f'Mean ER by subjectivity label:\n\n'
        + '\n'.join([f'    • {s}: {subj_er_data[s]*100:.4f}% '
                     f'(n={subj_er.get(s, {}).get("count", 0):,})'
                     for s in SUBJ_ORDER]) + '\n\n'
        f'  {top_subj} posts generate the highest engagement. '
        f'This reveals whether audiences are more responsive to factual evidence '
        f'or emotional opinions in political discourse.',
        f'Mean(Engagement_Rate_Monthly) per subj_label. Classified posts only.'
    ))

    # 6.8 Engagement by sentiment × timeline
    sec_insights.append((
        'INSIGHT 6.8',
        'Engagement-Sentiment Relationship Over Time',
        f'Mean ER by sentiment across timelines:\n\n'
        + '\n'.join([
            f'    {TL_LABELS[tl]}: ' + ', '.join([
                f'{s}={hc[(hc["_timeline"]==tl)&(hc["sentiment_label"]==s)]["Engagement_Rate_Monthly"].mean()*100:.4f}%'
                for s in SENT_ORDER
            ])
            for tl in TL_ORDER
        ]) + '\n\n'
        f'  Tracking how the engagement-sentiment relationship changes across time '
        f'reveals whether audiences become more or less receptive to particular sentiments '
        f'as the political cycle progresses.',
        f'Mean(Engagement_Rate_Monthly) per sentiment_label per _timeline × 100.'
    ))

    # 6.9 Conclusion
    sec_insights.append((
        'INSIGHT 6.9',
        'Conclusion — Engagement-Sentiment Dynamics',
        f'Key findings on engagement-sentiment correlations:\n\n'
        f'  1. {top_sent_er} posts generate the highest overall engagement.\n'
        f'  2. {"Negativity bias is confirmed" if top_sent_er == "Negative" else "The negativity bias hypothesis is not supported"} '
        f'at the overall level.\n'
        f'  3. There is significant per-account variation in the negativity bias pattern.\n'
        f'  4. {"Anger" if anger_er > hope_er else "Hope"} outperforms {"Hope" if anger_er > hope_er else "Anger"} in engagement.\n'
        f'  5. {"Inspirational" if insp_er > agg_er else "Aggressive"} tone generates higher ER than '
        f'{"Aggressive" if insp_er > agg_er else "Inspirational"}.\n'
        f'  6. {top_subj} content gets the most engagement among subjectivity categories.\n\n'
        f'  These findings have direct strategic implications: political actors who understand '
        f'which emotional register maximises engagement can optimise their content strategy.',
        f'Synthesis of insights 6.1–6.8.'
    ))

    all_sections.append(('Engagement × Sentiment Correlations', sec_insights))

    write_insight_report('06_engagement_correlations_insights.txt',
                         'ENGAGEMENT × SENTIMENT CORRELATIONS', all_sections)

    print(f"\n  Done! Generated 10 charts + 1 text report (9 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

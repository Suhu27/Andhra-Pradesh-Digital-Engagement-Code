# ============================================================
#  SCRIPT 01 — CAPTION SENTIMENT ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: sentiment_label (Positive / Neutral / Negative)
#  Source column: sentiment_label (XLM-RoBERTa multilingual classifier)
#  Supporting columns: score_positive, score_neutral, score_negative,
#                      sentiment_confidence
#  Subset: needs_review=False → 21,909 high-confidence rows
#
#  Run: python insight_01_caption_sentiment.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 01: CAPTION SENTIMENT ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)

    # ── All insights collected here ──────────────────────────────────────────
    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S1: Overall Caption Sentiment Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    dist, _ = compute_distribution(hc['sentiment_label'], SENT_ORDER)
    labels  = SENT_ORDER
    counts  = [dist[l]['count'] for l in labels]
    pcts    = [dist[l]['pct'] for l in labels]
    colors  = [SENT_COLORS[l] for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle(format_title('S1', 'Overall Caption Sentiment Distribution',
                              f'n = {n_total:,} high-confidence posts'),
                 fontsize=14, fontweight='bold', y=1.02)

    # Bar chart
    bars = ax1.bar(labels, pcts, color=colors, edgecolor='white', linewidth=0.8, width=0.6)
    add_value_labels(ax1, bars, fmt='{:.1f}%', counts=counts, fontsize=10)
    ax1.set_ylabel('Percentage of Posts (%)')
    ax1.set_xlabel('Caption Sentiment Label')
    ax1.set_ylim(0, max(pcts) * 1.25)
    ax1.set_title('Distribution (Bar)', fontsize=11, pad=10)

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        counts, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, pctdistance=0.75, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')
    ax2.set_title('Distribution (Pie)', fontsize=11, pad=10)

    add_methodology(fig,
        f'Source: sentiment_label column (XLM-RoBERTa multilingual sentiment classifier). '
        f'Subset: needs_review=False (n={n_total:,}). '
        f'Percentages = count of each label ÷ {n_total:,} × 100.',
        y=-0.04)
    save_chart(fig, 'S1_overall_caption_sentiment.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S2: Caption Sentiment by Account
    # ═════════════════════════════════════════════════════════════════════════
    grp = compute_grouped_distribution(hc, '_account', 'sentiment_label', SENT_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('S2', 'Caption Sentiment by Account',
                              f'Percentage distribution per account'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        acc_data = grp.get(acc, {})
        acc_total = acc_data.get('_total', 0)
        vals  = [acc_data.get(s, {}).get('pct', 0) for s in SENT_ORDER]
        cnts  = [acc_data.get(s, {}).get('count', 0) for s in SENT_ORDER]
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={acc_total:,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 60)
        if i == 0:
            ax.set_ylabel('Percentage of Posts (%)')
        ax.tick_params(axis='x', rotation=0)

    add_methodology(fig,
        f'Source: sentiment_label column. Subset: needs_review=False. '
        f'Percentages = count of label per account ÷ total posts for that account × 100. '
        f'Each panel sums to 100%.',
        y=-0.06)
    save_chart(fig, 'S2_caption_sentiment_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S3: Caption Sentiment by Alliance
    # ═════════════════════════════════════════════════════════════════════════
    grp_al = compute_grouped_distribution(hc, '_alliance', 'sentiment_label', SENT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(format_title('S3', 'Caption Sentiment by Alliance',
                              f'YSRCP Alliance = YSRCP + YS Jagan  |  TDP Alliance = TDP + NCBN'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        al_data  = grp_al.get(al, {})
        al_total = al_data.get('_total', 0)
        vals = [al_data.get(s, {}).get('pct', 0) for s in SENT_ORDER]
        cnts = [al_data.get(s, {}).get('count', 0) for s in SENT_ORDER]
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=10)
        ax.set_title(f'{al}\n(n={al_total:,})', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage of Posts (%)')

    add_methodology(fig,
        f'Source: sentiment_label column. Alliance = YSRCP+YS Jagan vs TDP+NCBN. '
        f'Percentages = count of label per alliance ÷ total posts for that alliance × 100.',
        y=-0.04)
    save_chart(fig, 'S3_caption_sentiment_by_alliance.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S4: Caption Sentiment by Timeline
    # ═════════════════════════════════════════════════════════════════════════
    grp_tl = compute_grouped_distribution(hc, '_timeline', 'sentiment_label', SENT_ORDER)

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('S4', 'Caption Sentiment Evolution Across Timelines',
                              f'Grouped by 3 time periods'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(TL_ORDER))
    width = 0.25
    for j, sent in enumerate(SENT_ORDER):
        vals = [grp_tl.get(tl, {}).get(sent, {}).get('pct', 0) for tl in TL_ORDER]
        cnts = [grp_tl.get(tl, {}).get(sent, {}).get('count', 0) for tl in TL_ORDER]
        bars = ax.bar(x + j * width, vals, width, label=sent,
                      color=SENT_COLORS[sent], edgecolor='white', linewidth=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=7)

    ax.set_xticks(x + width)
    tl_labels_with_n = [f'{TL_LABELS[tl]}\n(n={grp_tl.get(tl, {}).get("_total", 0):,})'
                        for tl in TL_ORDER]
    ax.set_xticklabels(tl_labels_with_n)
    ax.set_ylabel('Percentage of Posts (%)')
    ax.set_ylim(0, 55)
    ax.legend(title='Sentiment', framealpha=0.9)

    add_methodology(fig,
        f'Source: sentiment_label × _timeline columns. '
        f'Percentages = count of label per timeline ÷ total posts in that timeline × 100. '
        f'Timeline periods: Jun–Aug 2023, Sep–Dec 2023, Jan–May 2024.',
        y=-0.04)
    save_chart(fig, 'S4_caption_sentiment_by_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S5: Caption Sentiment by Platform
    # ═════════════════════════════════════════════════════════════════════════
    grp_pl = compute_grouped_distribution(hc, '_platform', 'sentiment_label', SENT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(format_title('S5', 'Caption Sentiment by Platform',
                              f'Facebook vs Instagram'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, plat in enumerate(PLAT_ORDER):
        ax = axes[i]
        pl_data  = grp_pl.get(plat, {})
        pl_total = pl_data.get('_total', 0)
        vals = [pl_data.get(s, {}).get('pct', 0) for s in SENT_ORDER]
        cnts = [pl_data.get(s, {}).get('count', 0) for s in SENT_ORDER]
        bars = ax.bar(SENT_ORDER, vals, color=[SENT_COLORS[s] for s in SENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=10)
        ax.set_title(f'{plat}\n(n={pl_total:,})', fontsize=12,
                     color=PLAT_COLORS[plat], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage of Posts (%)')

    add_methodology(fig,
        f'Source: sentiment_label × _platform columns. '
        f'Percentages = count of label per platform ÷ total posts on that platform × 100.',
        y=-0.04)
    save_chart(fig, 'S5_caption_sentiment_by_platform.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S6: Sentiment Confidence Distribution by Label
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('S6', 'Caption Sentiment Confidence Score Distribution',
                              f'How confident is the classifier for each sentiment label?'),
                 fontsize=14, fontweight='bold', y=1.02)

    for sent in SENT_ORDER:
        subset = hc[hc['sentiment_label'] == sent]['sentiment_confidence']
        ax.hist(subset, bins=30, alpha=0.6, label=f'{sent} (n={len(subset):,}, mean={subset.mean():.3f})',
                color=SENT_COLORS[sent], edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Sentiment Confidence Score')
    ax.set_ylabel('Number of Posts')
    ax.legend(title='Sentiment Label', framealpha=0.9, fontsize=9)
    ax.set_xlim(0.3, 1.0)

    add_methodology(fig,
        f'Source: sentiment_confidence column grouped by sentiment_label. '
        f'Histogram bins = 30. '
        f'Confidence score = max(score_positive, score_neutral, score_negative) '
        f'from the XLM-RoBERTa output logits. Range: [{hc["sentiment_confidence"].min():.3f}, {hc["sentiment_confidence"].max():.3f}].',
        y=-0.04)
    save_chart(fig, 'S6_sentiment_confidence_distribution.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S7: Mean Sentiment Scores by Account (3-class scores)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('S7', 'Mean Caption Sentiment Scores by Account',
                              f'Average of score_positive, score_neutral, score_negative per account'),
                 fontsize=14, fontweight='bold', y=1.02)

    score_cols = ['score_positive', 'score_neutral', 'score_negative']
    score_labels = ['Positive', 'Neutral', 'Negative']

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]
        means = [sub[c].mean() for c in score_cols]
        bars = ax.bar(score_labels, means,
                      color=[SENT_COLORS[l] for l in score_labels],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.3f}', fontsize=9)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 0.65)
        if i == 0:
            ax.set_ylabel('Mean Score')

    add_methodology(fig,
        f'Source: score_positive, score_neutral, score_negative columns. '
        f'These are the raw softmax probabilities from XLM-RoBERTa (sum to ~1.0 per row). '
        f'Values shown = mean of each score column per account.',
        y=-0.06)
    save_chart(fig, 'S7_mean_sentiment_scores_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S8: Positive-to-Negative Ratio by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('S8', 'Positive-to-Negative Ratio by Account',
                              f'Ratio > 1 = more positive posts than negative'),
                 fontsize=14, fontweight='bold', y=1.02)

    ratios = []
    accs_plot = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        n_pos = (sub['sentiment_label'] == 'Positive').sum()
        n_neg = (sub['sentiment_label'] == 'Negative').sum()
        ratio = n_pos / n_neg if n_neg > 0 else 0
        ratios.append(ratio)
        accs_plot.append(f'{ACC_LABELS[acc]}\n(n={len(sub):,})')

    bars = ax.bar(accs_plot, ratios, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.2f}', fontsize=11)
    ax.axhline(y=1.0, color='#E74C3C', linestyle='--', linewidth=1, alpha=0.7, label='Ratio = 1.0 (balanced)')
    ax.set_ylabel('Positive / Negative Ratio')
    ax.set_ylim(0, max(ratios) * 1.35)
    ax.legend(fontsize=9)

    add_methodology(fig,
        f'Source: sentiment_label column. '
        f'Ratio = count(Positive) ÷ count(Negative) per account. '
        f'Ratio > 1.0 means more positive posts than negative; < 1.0 means more negative.',
        y=-0.04)
    save_chart(fig, 'S8_positive_negative_ratio_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART S9: Sentiment by Account × Timeline (Faceted Heatmap)
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('S9', 'Caption Sentiment Trajectory per Account Across Timelines',
                              f'How each account\'s sentiment shifts over time'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]
        x = np.arange(len(TL_ORDER))
        width = 0.25

        for j, sent in enumerate(SENT_ORDER):
            vals = []
            cnts = []
            for tl in TL_ORDER:
                tl_sub = sub[sub['_timeline'] == tl]
                tl_total = len(tl_sub)
                c = (tl_sub['sentiment_label'] == sent).sum()
                vals.append(pct(c, tl_total))
                cnts.append(c)
            bars = ax.bar(x + j * width, vals, width, label=sent if i == 0 else '',
                          color=SENT_COLORS[sent], edgecolor='white', linewidth=0.3)
            add_value_labels(ax, bars, fmt='{:.0f}%', fontsize=6)

        ax.set_xticks(x + width)
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        acc_n = len(sub)
        ax.set_title(f'{ACC_LABELS[acc]} (n={acc_n:,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 65)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    axes[0].legend(title='Sentiment', fontsize=7, title_fontsize=8, loc='upper left')

    add_methodology(fig,
        f'Source: sentiment_label × _timeline × _account. '
        f'Each grouped bar = percentage of that sentiment within that account-timeline combination. '
        f'Percentages = count of label ÷ total posts for that account in that timeline × 100.',
        y=-0.08)
    save_chart(fig, 'S9_sentiment_trajectory_per_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    # ── Compute all numbers needed ──
    overall = compute_distribution(hc['sentiment_label'], SENT_ORDER)[0]
    by_acc  = compute_grouped_distribution(hc, '_account', 'sentiment_label', SENT_ORDER)
    by_al   = compute_grouped_distribution(hc, '_alliance', 'sentiment_label', SENT_ORDER)
    by_tl   = compute_grouped_distribution(hc, '_timeline', 'sentiment_label', SENT_ORDER)
    by_pl   = compute_grouped_distribution(hc, '_platform', 'sentiment_label', SENT_ORDER)

    # Positive-to-Negative ratios
    pn_ratios = {}
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        np_ = (sub['sentiment_label'] == 'Positive').sum()
        nn_ = (sub['sentiment_label'] == 'Negative').sum()
        pn_ratios[acc] = np_ / nn_ if nn_ > 0 else float('inf')

    # Mean confidence by sentiment
    conf_by_sent = {}
    for sent in SENT_ORDER:
        conf_by_sent[sent] = hc[hc['sentiment_label'] == sent]['sentiment_confidence'].mean()

    # Mean scores by account
    score_means = {}
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        score_means[acc] = {
            'pos': sub['score_positive'].mean(),
            'neu': sub['score_neutral'].mean(),
            'neg': sub['score_negative'].mean(),
        }

    # ── Build insight sections ──

    sec1_insights = []

    # Insight 1.1
    sec1_insights.append((
        'INSIGHT 1.1',
        'Overall Caption Sentiment Distribution',
        f'Out of {n_total:,} high-confidence posts analysed, Neutral sentiment leads with '
        f'{overall["Neutral"]["count"]:,} posts ({overall["Neutral"]["pct"]:.1f}%), followed by '
        f'Negative at {overall["Negative"]["count"]:,} ({overall["Negative"]["pct"]:.1f}%) and '
        f'Positive at {overall["Positive"]["count"]:,} ({overall["Positive"]["pct"]:.1f}%).\n\n'
        f'  The dominance of Neutral sentiment indicates that the majority of political social media '
        f'content in Andhra Pradesh maintains a relatively measured textual tone, though a significant '
        f'one-third of all posts tilts negative. Only one in four posts is classified as positive. '
        f'This pattern is consistent with political discourse where criticism and news-sharing '
        f'(both neutral-coded) outweigh celebratory content.',
        f'Counts of each unique value in the sentiment_label column for rows where needs_review=False. '
        f'Percentages = count of each label ÷ {n_total:,} (total high-confidence posts) × 100. '
        f'Classification: XLM-RoBERTa multilingual sentiment model with 3-class softmax output.'
    ))

    # Insight 1.2
    most_neg_acc = max(ACC_ORDER, key=lambda a: by_acc[a].get('Negative', {}).get('pct', 0))
    most_pos_acc = max(ACC_ORDER, key=lambda a: by_acc[a].get('Positive', {}).get('pct', 0))
    least_neg_acc = min(ACC_ORDER, key=lambda a: by_acc[a].get('Negative', {}).get('pct', 0))

    acc_breakdown = []
    for acc in ACC_ORDER:
        d = by_acc[acc]
        acc_breakdown.append(
            f'    • {ACC_LABELS[acc]} (n={d["_total"]:,}): '
            f'Positive={d["Positive"]["pct"]:.1f}%, '
            f'Neutral={d["Neutral"]["pct"]:.1f}%, '
            f'Negative={d["Negative"]["pct"]:.1f}%'
        )

    sec1_insights.append((
        'INSIGHT 1.2',
        'Caption Sentiment Breakdown by Account',
        f'Per-account caption sentiment distributions:\n\n'
        + '\n'.join(acc_breakdown) + '\n\n'
        f'  {ACC_LABELS[most_neg_acc]} has the highest negative percentage at '
        f'{by_acc[most_neg_acc]["Negative"]["pct"]:.1f}%, while {ACC_LABELS[most_pos_acc]} '
        f'has the highest positive percentage at {by_acc[most_pos_acc]["Positive"]["pct"]:.1f}%. '
        f'{ACC_LABELS[least_neg_acc]} has the lowest negativity at '
        f'{by_acc[least_neg_acc]["Negative"]["pct"]:.1f}%.',
        f'Counts of sentiment_label per _account, divided by total posts for that account × 100. '
        f'Source columns: sentiment_label, _account. Subset: needs_review=False.'
    ))

    # Insight 1.3 — Alliance comparison
    ysrcp_al = by_al.get('YSRCP Alliance', {})
    tdp_al   = by_al.get('TDP Alliance', {})

    sec1_insights.append((
        'INSIGHT 1.3',
        'Alliance-Level Caption Sentiment Comparison',
        f'YSRCP Alliance (YSRCP + YS Jagan, n={ysrcp_al.get("_total", 0):,}):\n'
        f'    Positive = {ysrcp_al.get("Positive", {}).get("pct", 0):.1f}% ({ysrcp_al.get("Positive", {}).get("count", 0):,}), '
        f'Neutral = {ysrcp_al.get("Neutral", {}).get("pct", 0):.1f}% ({ysrcp_al.get("Neutral", {}).get("count", 0):,}), '
        f'Negative = {ysrcp_al.get("Negative", {}).get("pct", 0):.1f}% ({ysrcp_al.get("Negative", {}).get("count", 0):,})\n\n'
        f'  TDP Alliance (TDP + NCBN, n={tdp_al.get("_total", 0):,}):\n'
        f'    Positive = {tdp_al.get("Positive", {}).get("pct", 0):.1f}% ({tdp_al.get("Positive", {}).get("count", 0):,}), '
        f'Neutral = {tdp_al.get("Neutral", {}).get("pct", 0):.1f}% ({tdp_al.get("Neutral", {}).get("count", 0):,}), '
        f'Negative = {tdp_al.get("Negative", {}).get("pct", 0):.1f}% ({tdp_al.get("Negative", {}).get("count", 0):,})\n\n'
        f'  The TDP Alliance shows a {tdp_al.get("Negative", {}).get("pct", 0) - ysrcp_al.get("Negative", {}).get("pct", 0):.1f} '
        f'percentage-point higher negativity rate compared to the YSRCP Alliance. '
        f'This is consistent with typical opposition party communication strategies that '
        f'rely more heavily on critique and negative framing against the incumbent government.',
        f'Posts grouped by alliance mapping: YSRCP + YS Jagan → YSRCP Alliance; TDP + NCBN → TDP Alliance. '
        f'Percentages = count of label per alliance ÷ total posts for that alliance × 100.'
    ))

    # Insight 1.4 — Timeline trend
    tl_trend = []
    for tl in TL_ORDER:
        d = by_tl.get(tl, {})
        tl_trend.append(
            f'    • {TL_LABELS[tl]} (n={d.get("_total", 0):,}): '
            f'Neg={d.get("Negative", {}).get("pct", 0):.1f}%, '
            f'Pos={d.get("Positive", {}).get("pct", 0):.1f}%'
        )

    first_neg = by_tl.get(TL_ORDER[0], {}).get('Negative', {}).get('pct', 0)
    last_neg  = by_tl.get(TL_ORDER[-1], {}).get('Negative', {}).get('pct', 0)
    neg_trend = 'increasing' if last_neg > first_neg else 'decreasing' if last_neg < first_neg else 'stable'

    sec1_insights.append((
        'INSIGHT 1.4',
        'Caption Sentiment Trend Across Timelines',
        f'Negativity across the three time periods:\n\n'
        + '\n'.join(tl_trend) + '\n\n'
        f'  Caption negativity shows a {neg_trend} trend from {TL_LABELS[TL_ORDER[0]]} '
        f'({first_neg:.1f}%) to {TL_LABELS[TL_ORDER[-1]]} ({last_neg:.1f}%), '
        f'a shift of {last_neg - first_neg:+.1f} percentage points. '
        f'The final period (Jan–May 2024) coincides with pre-election campaigning in AP, '
        f'and any change in negativity could reflect intensifying campaign rhetoric.',
        f'Sentiment_label counts per _timeline period, divided by total posts in that period × 100. '
        f'Timeline periods: Jun–Aug 2023, Sep–Dec 2023, Jan–May 2024.'
    ))

    # Insight 1.5 — Platform comparison
    fb_data  = by_pl.get('Facebook', {})
    ig_data  = by_pl.get('Instagram', {})

    sec1_insights.append((
        'INSIGHT 1.5',
        'Caption Sentiment by Platform — Facebook vs Instagram',
        f'Facebook (n={fb_data.get("_total", 0):,}): '
        f'Positive={fb_data.get("Positive", {}).get("pct", 0):.1f}%, '
        f'Neutral={fb_data.get("Neutral", {}).get("pct", 0):.1f}%, '
        f'Negative={fb_data.get("Negative", {}).get("pct", 0):.1f}%\n\n'
        f'  Instagram (n={ig_data.get("_total", 0):,}): '
        f'Positive={ig_data.get("Positive", {}).get("pct", 0):.1f}%, '
        f'Neutral={ig_data.get("Neutral", {}).get("pct", 0):.1f}%, '
        f'Negative={ig_data.get("Negative", {}).get("pct", 0):.1f}%\n\n'
        f'  The platform-level comparison reveals whether political parties adapt their '
        f'messaging tone to the demographic and algorithmic characteristics of each platform. '
        f'A higher negativity on one platform may indicate strategic targeting of critique-oriented '
        f'content for audiences more receptive to political debate.',
        f'Sentiment_label counts per _platform, divided by total posts on that platform × 100.'
    ))

    # Insight 1.6 — Confidence analysis
    sec1_insights.append((
        'INSIGHT 1.6',
        'Classifier Confidence Analysis by Sentiment Label',
        f'Mean classifier confidence by sentiment label:\n'
        f'    • Positive: {conf_by_sent["Positive"]:.3f}\n'
        f'    • Neutral:  {conf_by_sent["Neutral"]:.3f}\n'
        f'    • Negative: {conf_by_sent["Negative"]:.3f}\n\n'
        f'  The confidence score represents the highest softmax probability across the three '
        f'sentiment classes. A higher mean confidence indicates the classifier is more "certain" '
        f'about that particular label assignment. '
        f'{"Negative" if conf_by_sent["Negative"] >= max(conf_by_sent.values()) else "Positive" if conf_by_sent["Positive"] >= max(conf_by_sent.values()) else "Neutral"} '
        f'posts have the highest mean confidence, suggesting the linguistic signals '
        f'for that sentiment are the most distinctive in this corpus.',
        f'Mean of sentiment_confidence column grouped by sentiment_label. '
        f'Confidence = max(score_positive, score_neutral, score_negative) from the XLM-RoBERTa classifier. '
        f'Range: [{hc["sentiment_confidence"].min():.3f}, {hc["sentiment_confidence"].max():.3f}].'
    ))

    # Insight 1.7 — Positive-to-Negative ratio
    ratio_lines = []
    for acc in ACC_ORDER:
        ratio_lines.append(f'    • {ACC_LABELS[acc]}: {pn_ratios[acc]:.2f}')

    most_pos_ratio = max(ACC_ORDER, key=lambda a: pn_ratios[a])
    most_neg_ratio = min(ACC_ORDER, key=lambda a: pn_ratios[a])

    sec1_insights.append((
        'INSIGHT 1.7',
        'Positive-to-Negative Ratio per Account',
        f'Positive/Negative ratio (>1 = net positive; <1 = net negative):\n\n'
        + '\n'.join(ratio_lines) + '\n\n'
        f'  {ACC_LABELS[most_pos_ratio]} has the highest P/N ratio at {pn_ratios[most_pos_ratio]:.2f}, '
        f'meaning it posts {pn_ratios[most_pos_ratio]:.1f}× more positive content than negative. '
        f'{ACC_LABELS[most_neg_ratio]} has the lowest ratio at {pn_ratios[most_neg_ratio]:.2f}, '
        f'indicating a greater lean towards negative messaging.\n\n'
        f'  This metric is a single-number summary of each account\'s "positivity balance." '
        f'In political communication, a ratio below 1.0 suggests the account relies primarily '
        f'on attack-based messaging; above 1.0 suggests a more constructive communication style.',
        f'Ratio = count(sentiment_label="Positive") ÷ count(sentiment_label="Negative") per account. '
        f'Source: sentiment_label column, needs_review=False subset.'
    ))

    # Insight 1.8 — Raw sentiment scores by account
    score_lines = []
    for acc in ACC_ORDER:
        sm = score_means[acc]
        score_lines.append(
            f'    • {ACC_LABELS[acc]}: score_pos={sm["pos"]:.3f}, score_neu={sm["neu"]:.3f}, score_neg={sm["neg"]:.3f}'
        )

    sec1_insights.append((
        'INSIGHT 1.8',
        'Mean Sentiment Probability Scores by Account',
        f'Mean raw classification scores (softmax probabilities, sum ≈ 1.0 per post):\n\n'
        + '\n'.join(score_lines) + '\n\n'
        f'  Unlike the discrete sentiment_label (which assigns the argmax label), these mean '
        f'scores reveal the average probability "lean" of each account\'s posts. An account '
        f'with similar scores across all three classes has more ambiguous/contested content; '
        f'an account with one score much higher has more clearly polarised content.',
        f'Mean of score_positive, score_neutral, score_negative columns per _account. '
        f'These columns contain the raw 3-class softmax probabilities from XLM-RoBERTa. '
        f'They sum to approximately 1.0 per row.'
    ))

    # Insight 1.9 — Overall conclusion
    dominant_sent = max(SENT_ORDER, key=lambda s: overall[s]['pct'])
    sec1_insights.append((
        'INSIGHT 1.9',
        'Conclusion — Caption Sentiment Landscape',
        f'The Andhra Pradesh political social media landscape, as captured through {n_total:,} '
        f'high-confidence posts across four accounts and two platforms, is characterised by:\n\n'
        f'  1. A {dominant_sent.lower()}-dominant overall distribution, with {overall[dominant_sent]["pct"]:.1f}% '
        f'of posts classified as {dominant_sent}.\n'
        f'  2. Significant inter-account variation: {ACC_LABELS[most_pos_acc]} leans most positive, '
        f'while {ACC_LABELS[most_neg_acc]} leans most negative.\n'
        f'  3. The TDP Alliance exhibits higher negativity than the YSRCP Alliance, consistent '
        f'with opposition-vs-incumbent communication dynamics.\n'
        f'  4. Temporal analysis shows {neg_trend} negativity toward the pre-election period.\n'
        f'  5. Classifier confidence is highest for {max(conf_by_sent, key=conf_by_sent.get)} '
        f'posts, indicating the clearest linguistic markers for that sentiment.\n\n'
        f'  These patterns establish the baseline sentiment landscape against which deeper '
        f'analyses (emotion, tone, subjectivity) should be interpreted.',
        f'Synthesis of insights 1.1–1.8. No new calculations; this is an interpretive summary.'
    ))

    all_sections.append(('Overall Distribution & Fundamentals', sec1_insights))

    write_insight_report('01_caption_sentiment_insights.txt',
                         'CAPTION SENTIMENT ANALYSIS', all_sections)

    print(f"\n  Done! Generated 9 charts + 1 text report (9 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

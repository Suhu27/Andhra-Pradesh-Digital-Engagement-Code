# ============================================================
#  SCRIPT 02 — HASHTAG SENTIMENT ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: hashtag_dominant_label (Positive / Neutral / Negative / NoHashtags)
#  Supporting: hashtag_sentiment_score, hashtag_neg/neu/pos_count,
#              hashtag_total_count, hashtag_avg_confidence, hashtag_charged_tags
#  Subset: needs_review=False → 21,909 rows (19,864 with hashtags)
#
#  Run: python insight_02_hashtag_sentiment.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 02: HASHTAG SENTIMENT ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)

    # Subset: posts WITH hashtags (exclude NoHashtags)
    hc_ht = hc[hc['hashtag_dominant_label'] != 'NoHashtags'].copy()
    n_ht = len(hc_ht)
    n_no_ht = n_total - n_ht
    print(f"  → {n_ht:,} posts with hashtags, {n_no_ht:,} without hashtags.\n")

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H1: Overall Hashtag Sentiment Distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("  Generating charts...")

    # Including NoHashtags for context
    full_dist, _ = compute_distribution(hc['hashtag_dominant_label'],
                                        HSENT_ORDER + ['NoHashtags'])
    # Excluding NoHashtags for pure sentiment %
    ht_dist, _ = compute_distribution(hc_ht['hashtag_dominant_label'], HSENT_ORDER)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={'width_ratios': [1, 1.2]})
    fig.suptitle(format_title('H1', 'Overall Hashtag Sentiment Distribution',
                              f'n = {n_total:,} total  |  {n_ht:,} with hashtags  |  {n_no_ht:,} without'),
                 fontsize=14, fontweight='bold', y=1.02)

    # Left: All posts including NoHashtags
    all_labels = HSENT_ORDER + ['NoHashtags']
    all_counts = [full_dist[l]['count'] for l in all_labels]
    all_pcts   = [full_dist[l]['pct'] for l in all_labels]
    all_cols   = [HSENT_COLORS[l] for l in all_labels]
    bars1 = ax1.bar(all_labels, all_pcts, color=all_cols, edgecolor='white', width=0.55)
    add_value_labels(ax1, bars1, fmt='{:.1f}%', counts=all_counts, fontsize=9)
    ax1.set_title('Including NoHashtags Posts', fontsize=11)
    ax1.set_ylabel('Percentage of All Posts (%)')
    ax1.set_ylim(0, max(all_pcts) * 1.3)

    # Right: Only posts with hashtags
    ht_labels = HSENT_ORDER
    ht_counts = [ht_dist[l]['count'] for l in ht_labels]
    ht_pcts   = [ht_dist[l]['pct'] for l in ht_labels]
    ht_cols   = [HSENT_COLORS[l] for l in ht_labels]
    bars2 = ax2.bar(ht_labels, ht_pcts, color=ht_cols, edgecolor='white', width=0.5)
    add_value_labels(ax2, bars2, fmt='{:.1f}%', counts=ht_counts, fontsize=10)
    ax2.set_title(f'Only Posts With Hashtags (n={n_ht:,})', fontsize=11)
    ax2.set_ylabel('Percentage of Hashtagged Posts (%)')
    ax2.set_ylim(0, max(ht_pcts) * 1.3)

    add_methodology(fig,
        f'Source: hashtag_dominant_label column. Left panel: all {n_total:,} HC posts. '
        f'Right panel: {n_ht:,} posts where hashtag_dominant_label ≠ "NoHashtags". '
        f'Hashtag sentiment assigned via hashtag dictionary lookup; dominant_label = label with '
        f'highest count among pos/neu/neg hashtags per post.',
        y=-0.06)
    save_chart(fig, 'H1_overall_hashtag_sentiment.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H2: Hashtag Sentiment by Account
    # ═════════════════════════════════════════════════════════════════════════
    grp = compute_grouped_distribution(hc_ht, '_account', 'hashtag_dominant_label', HSENT_ORDER)

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('H2', 'Hashtag Sentiment by Account',
                              f'Percentage distribution among posts with hashtags'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        acc_data = grp.get(acc, {})
        acc_total = acc_data.get('_total', 0)
        vals = [acc_data.get(s, {}).get('pct', 0) for s in HSENT_ORDER]
        cnts = [acc_data.get(s, {}).get('count', 0) for s in HSENT_ORDER]
        bars = ax.bar(HSENT_ORDER, vals, color=[HSENT_COLORS[s] for s in HSENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=8)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={acc_total:,} with HT)', fontsize=11,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_ylim(0, 55)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    add_methodology(fig,
        f'Source: hashtag_dominant_label × _account. Excludes NoHashtags posts. '
        f'Percentages = count per label ÷ total hashtagged posts for that account × 100.',
        y=-0.06)
    save_chart(fig, 'H2_hashtag_sentiment_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H3: Caption vs Hashtag Sentiment Agreement Matrix (Heatmap)
    # ═════════════════════════════════════════════════════════════════════════
    agree_df = hc_ht[['sentiment_label', 'hashtag_dominant_label']].copy()
    ct = pd.crosstab(agree_df['sentiment_label'], agree_df['hashtag_dominant_label'],
                     normalize='all') * 100

    # Reorder
    ct = ct.reindex(index=SENT_ORDER, columns=HSENT_ORDER, fill_value=0)
    ct_counts = pd.crosstab(agree_df['sentiment_label'], agree_df['hashtag_dominant_label'])
    ct_counts = ct_counts.reindex(index=SENT_ORDER, columns=HSENT_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('H3', 'Caption vs Hashtag Sentiment Agreement Matrix',
                              f'n = {n_ht:,} posts with both caption and hashtag sentiment'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(HSENT_ORDER)))
    ax.set_xticklabels([f'HT: {s}' for s in HSENT_ORDER])
    ax.set_yticks(range(len(SENT_ORDER)))
    ax.set_yticklabels([f'Caption: {s}' for s in SENT_ORDER])
    ax.set_xlabel('Hashtag Dominant Sentiment')
    ax.set_ylabel('Caption Sentiment')

    # Annotate cells
    for r in range(len(SENT_ORDER)):
        for c in range(len(HSENT_ORDER)):
            val = ct.values[r, c]
            cnt = ct_counts.values[r, c]
            color = 'white' if val > ct.values.max() * 0.6 else 'black'
            ax.text(c, r, f'{val:.1f}%\n(n={cnt:,})', ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

    # Diagonal = agreement
    for k in range(min(len(SENT_ORDER), len(HSENT_ORDER))):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                                    fill=False, edgecolor='#2ECC71', linewidth=2.5))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage of All Hashtagged Posts (%)')

    add_methodology(fig,
        f'Source: cross-tab of sentiment_label × hashtag_dominant_label for {n_ht:,} posts '
        f'with hashtags. Cell value = count in cell ÷ {n_ht:,} × 100. '
        f'Green borders = diagonal (agreement: caption and hashtag sentiment match). '
        f'Off-diagonal cells = disagreement between caption and hashtag sentiment.',
        y=-0.04)
    save_chart(fig, 'H3_caption_hashtag_agreement_matrix.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H4: Hashtag Sentiment Score Distribution by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('H4', 'Hashtag Sentiment Score Distribution by Account',
                              f'Score range: [-1 (all negative HTs) to +1 (all positive HTs)]'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_ht[hc_ht['_account'] == acc]['hashtag_sentiment_score'].dropna()
        ax.hist(sub, bins=25, color=ACC_COLORS[acc], edgecolor='white',
                linewidth=0.5, alpha=0.85)
        mean_val = sub.mean()
        ax.axvline(mean_val, color='#E74C3C', linestyle='--', linewidth=1.5,
                   label=f'Mean={mean_val:.3f}')
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,}, mean={mean_val:.3f})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_xlabel('Hashtag Score')
        if i == 0:
            ax.set_ylabel('Number of Posts')
        ax.legend(fontsize=7)
        ax.set_xlim(-1.1, 1.1)

    add_methodology(fig,
        f'Source: hashtag_sentiment_score column (range [-1, +1]). '
        f'Score = (pos_count − neg_count) ÷ total_matched_hashtags per post. '
        f'+1 = all hashtags positive, −1 = all negative, 0 = balanced/neutral. '
        f'Red dashed line = mean score per account. Posts without hashtags excluded.',
        y=-0.08)
    save_chart(fig, 'H4_hashtag_score_distribution_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H5: Mean Hashtag Volume per Post by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('H5', 'Mean Hashtag Volume per Post by Account',
                              f'Average number of hashtags used per post'),
                 fontsize=14, fontweight='bold', y=1.02)

    ht_vols = []
    ht_labels_plot = []
    ht_ns = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        mean_ht = sub['hashtag_total_count'].mean()
        ht_vols.append(mean_ht)
        ht_labels_plot.append(f'{ACC_LABELS[acc]}\n(n={len(sub):,})')
        ht_ns.append(len(sub))

    bars = ax.bar(ht_labels_plot, ht_vols, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.2f}', fontsize=11)
    ax.set_ylabel('Mean Hashtags per Post')
    ax.set_ylim(0, max(ht_vols) * 1.3)

    add_methodology(fig,
        f'Source: hashtag_total_count column. '
        f'Value = mean(hashtag_total_count) per _account for all {n_total:,} HC posts '
        f'(including posts with 0 hashtags).',
        y=-0.04)
    save_chart(fig, 'H5_hashtag_volume_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H6: Caption-Hashtag Agreement Rate by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('H6', 'Caption-Hashtag Sentiment Agreement Rate by Account',
                              f'% of hashtagged posts where caption and hashtag sentiment match'),
                 fontsize=14, fontweight='bold', y=1.02)

    agree_rates = []
    agree_labels = []
    agree_ns = []
    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        agree = (sub['sentiment_label'] == sub['hashtag_dominant_label']).sum()
        rate = pct(agree, len(sub))
        agree_rates.append(rate)
        agree_labels.append(f'{ACC_LABELS[acc]}\n(n={len(sub):,})')
        agree_ns.append(agree)

    bars = ax.bar(agree_labels, agree_rates, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.5,
                f'{h:.1f}%\n({agree_ns[i]:,} agree)', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#333333')
    ax.set_ylabel('Agreement Rate (%)')
    ax.set_ylim(0, max(agree_rates) * 1.25)
    ax.axhline(y=33.3, color='#888888', linestyle=':', linewidth=1,
               alpha=0.7, label='Random baseline (33.3%)')
    ax.legend(fontsize=8)

    add_methodology(fig,
        f'Source: sentiment_label vs hashtag_dominant_label. '
        f'Agreement = posts where both labels are identical (e.g., both "Positive"). '
        f'Rate = agreement_count ÷ total hashtagged posts per account × 100. '
        f'Random baseline = 33.3% (3 classes, random assignment).',
        y=-0.04)
    save_chart(fig, 'H6_caption_hashtag_agreement_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H7: Hashtag Sentiment by Alliance
    # ═════════════════════════════════════════════════════════════════════════
    grp_al = compute_grouped_distribution(hc_ht, '_alliance', 'hashtag_dominant_label', HSENT_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    fig.suptitle(format_title('H7', 'Hashtag Sentiment by Alliance',
                              f'YSRCP Alliance vs TDP Alliance (hashtagged posts only)'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        al_data  = grp_al.get(al, {})
        al_total = al_data.get('_total', 0)
        vals = [al_data.get(s, {}).get('pct', 0) for s in HSENT_ORDER]
        cnts = [al_data.get(s, {}).get('count', 0) for s in HSENT_ORDER]
        bars = ax.bar(HSENT_ORDER, vals, color=[HSENT_COLORS[s] for s in HSENT_ORDER],
                      edgecolor='white', linewidth=0.8, width=0.5)
        add_value_labels(ax, bars, fmt='{:.1f}%', counts=cnts, fontsize=10)
        ax.set_title(f'{al}\n(n={al_total:,} with HT)', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_ylim(0, 50)
        if i == 0:
            ax.set_ylabel('Percentage (%)')

    add_methodology(fig,
        f'Source: hashtag_dominant_label × _alliance. NoHashtags posts excluded. '
        f'Percentages = count per label ÷ total hashtagged posts for that alliance × 100.',
        y=-0.04)
    save_chart(fig, 'H7_hashtag_sentiment_by_alliance.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART H8: Hashtag Composition Breakdown by Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)
    fig.suptitle(format_title('H8', 'Mean Hashtag Composition per Post by Account',
                              f'Average number of positive, neutral, and negative hashtags per post'),
                 fontsize=14, fontweight='bold', y=1.02)

    ht_comp_cols = ['hashtag_pos_count', 'hashtag_neu_count', 'hashtag_neg_count']
    ht_comp_labels = ['Positive HTs', 'Neutral HTs', 'Negative HTs']
    ht_comp_colors = [SENT_COLORS['Positive'], SENT_COLORS['Neutral'], SENT_COLORS['Negative']]

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc_ht[hc_ht['_account'] == acc]
        means = [sub[c].mean() for c in ht_comp_cols]
        bars = ax.bar(ht_comp_labels, means, color=ht_comp_colors,
                      edgecolor='white', linewidth=0.8, width=0.55)
        add_value_labels(ax, bars, fmt='{:.2f}', fontsize=9)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Mean Count per Post')
        ax.tick_params(axis='x', rotation=15)

    add_methodology(fig,
        f'Source: hashtag_pos_count, hashtag_neu_count, hashtag_neg_count columns. '
        f'Values = mean count of each hashtag type per post, only for posts with ≥1 hashtag. '
        f'Shows the average composition of hashtag sentiment within each post.',
        y=-0.08)
    save_chart(fig, 'H8_hashtag_composition_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    # Pre-compute
    ht_overall = compute_distribution(hc_ht['hashtag_dominant_label'], HSENT_ORDER)[0]
    by_acc_ht  = compute_grouped_distribution(hc_ht, '_account', 'hashtag_dominant_label', HSENT_ORDER)

    overall_agree = (hc_ht['sentiment_label'] == hc_ht['hashtag_dominant_label']).sum()
    overall_agree_rate = pct(overall_agree, n_ht)

    # Agreement by account
    acc_agree = {}
    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        ag = (sub['sentiment_label'] == sub['hashtag_dominant_label']).sum()
        acc_agree[acc] = {'agree': ag, 'rate': pct(ag, len(sub)), 'total': len(sub)}

    # Mean hashtag score by account
    acc_ht_score = {}
    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]['hashtag_sentiment_score'].dropna()
        acc_ht_score[acc] = sub.mean()

    # Mean hashtag volume
    acc_ht_vol = {}
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        acc_ht_vol[acc] = sub['hashtag_total_count'].mean()

    # Disagreement patterns
    disagree = hc_ht[hc_ht['sentiment_label'] != hc_ht['hashtag_dominant_label']]
    disagree_patterns = pd.crosstab(disagree['sentiment_label'],
                                     disagree['hashtag_dominant_label'])

    sec_insights = []

    # Insight 2.1
    sec_insights.append((
        'INSIGHT 2.1',
        'Overall Hashtag Sentiment Distribution',
        f'Among {n_ht:,} posts with hashtags (out of {n_total:,} total; {n_no_ht:,} have no hashtags):\n\n'
        f'  • Negative: {ht_overall["Negative"]["count"]:,} ({ht_overall["Negative"]["pct"]:.1f}%)\n'
        f'  • Neutral:  {ht_overall["Neutral"]["count"]:,} ({ht_overall["Neutral"]["pct"]:.1f}%)\n'
        f'  • Positive: {ht_overall["Positive"]["count"]:,} ({ht_overall["Positive"]["pct"]:.1f}%)\n\n'
        f'  Hashtag sentiment is more evenly distributed than caption sentiment, with the '
        f'three categories closely balanced. This contrasts with caption sentiment where '
        f'Neutral dominates. Hashtags tend to carry more explicit political signalling '
        f'(both attack-oriented and promotional), leading to a more polarised distribution.',
        f'Counts of hashtag_dominant_label (excluding "NoHashtags") ÷ {n_ht:,} × 100. '
        f'hashtag_dominant_label = the sentiment label (Pos/Neu/Neg) with the highest '
        f'count among matched hashtags in each post.'
    ))

    # Insight 2.2
    acc_ht_lines = []
    for acc in ACC_ORDER:
        d = by_acc_ht.get(acc, {})
        acc_ht_lines.append(
            f'    • {ACC_LABELS[acc]} (n={d.get("_total", 0):,}): '
            f'Pos={d.get("Positive", {}).get("pct", 0):.1f}%, '
            f'Neu={d.get("Neutral", {}).get("pct", 0):.1f}%, '
            f'Neg={d.get("Negative", {}).get("pct", 0):.1f}%'
        )

    most_pos_ht = max(ACC_ORDER, key=lambda a: by_acc_ht.get(a, {}).get('Positive', {}).get('pct', 0))
    most_neg_ht = max(ACC_ORDER, key=lambda a: by_acc_ht.get(a, {}).get('Negative', {}).get('pct', 0))

    sec_insights.append((
        'INSIGHT 2.2',
        'Hashtag Sentiment by Account',
        f'Per-account hashtag sentiment (posts with hashtags only):\n\n'
        + '\n'.join(acc_ht_lines) + '\n\n'
        f'  {ACC_LABELS[most_pos_ht]} leads in positive hashtag usage at '
        f'{by_acc_ht.get(most_pos_ht, {}).get("Positive", {}).get("pct", 0):.1f}%, while '
        f'{ACC_LABELS[most_neg_ht]} has the highest negative hashtag proportion at '
        f'{by_acc_ht.get(most_neg_ht, {}).get("Negative", {}).get("pct", 0):.1f}%.',
        f'hashtag_dominant_label counts per _account (NoHashtags excluded) ÷ total '
        f'hashtagged posts per account × 100.'
    ))

    # Insight 2.3 — Agreement rate
    sec_insights.append((
        'INSIGHT 2.3',
        'Caption-Hashtag Sentiment Agreement Rate',
        f'Overall agreement rate: {overall_agree_rate:.1f}% ({overall_agree:,} of {n_ht:,} '
        f'hashtagged posts have matching caption and hashtag sentiment).\n\n'
        f'  Per-account agreement rates:\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {acc_agree[a]["rate"]:.1f}% ({acc_agree[a]["agree"]:,}/{acc_agree[a]["total"]:,})'
                     for a in ACC_ORDER]) + '\n\n'
        f'  A random chance agreement rate with 3 classes is 33.3%, so the observed '
        f'{overall_agree_rate:.1f}% indicates {"better than random" if overall_agree_rate > 33.3 else "near-random"} '
        f'alignment between caption text sentiment and hashtag sentiment. '
        f'Partial agreement is expected because hashtags often serve different communicative '
        f'functions (branding, trending) than the caption text itself.',
        f'Agreement = posts where sentiment_label == hashtag_dominant_label. '
        f'Rate = agreement count ÷ total hashtagged posts × 100. Random baseline = 33.3% (uniform 3-class).'
    ))

    # Insight 2.4 — Disagreement patterns
    top_disagree = disagree_patterns.stack().sort_values(ascending=False).head(5)
    disagree_lines = []
    for (cap_s, ht_s), count in top_disagree.items():
        disagree_lines.append(f'    • Caption={cap_s} + Hashtag={ht_s}: {count:,} posts')

    sec_insights.append((
        'INSIGHT 2.4',
        'Caption-Hashtag Sentiment Disagreement Patterns',
        f'When caption and hashtag sentiment disagree ({len(disagree):,} posts), '
        f'the most common mismatch patterns are:\n\n'
        + '\n'.join(disagree_lines) + '\n\n'
        f'  These mismatches often occur when a neutral/positive caption is paired with '
        f'attack-oriented hashtags (negative), or when a negative critique uses '
        f'self-promotional (positive) hashtags for discoverability.',
        f'Cross-tab of sentiment_label × hashtag_dominant_label, filtered to rows where '
        f'the two labels differ. Top patterns sorted by count descending.'
    ))

    # Insight 2.5 — Hashtag volume
    vol_lines = [f'    • {ACC_LABELS[a]}: {acc_ht_vol[a]:.2f} hashtags/post' for a in ACC_ORDER]
    most_vol = max(ACC_ORDER, key=lambda a: acc_ht_vol[a])
    least_vol = min(ACC_ORDER, key=lambda a: acc_ht_vol[a])

    sec_insights.append((
        'INSIGHT 2.5',
        'Hashtag Usage Volume by Account',
        f'Mean hashtags per post (including posts with 0 hashtags):\n\n'
        + '\n'.join(vol_lines) + '\n\n'
        f'  {ACC_LABELS[most_vol]} uses the most hashtags per post ({acc_ht_vol[most_vol]:.2f}), '
        f'while {ACC_LABELS[least_vol]} uses the fewest ({acc_ht_vol[least_vol]:.2f}). '
        f'Higher hashtag volume typically indicates a more aggressive SEO/discoverability '
        f'strategy on social media, and may also reflect more complex political messaging.',
        f'Mean of hashtag_total_count per _account for all {n_total:,} HC posts. '
        f'Includes posts with 0 hashtags in the denominator.'
    ))

    # Insight 2.6 — Hashtag sentiment score by account
    score_lines = [f'    • {ACC_LABELS[a]}: {acc_ht_score[a]:.4f}' for a in ACC_ORDER]
    most_pos_score = max(ACC_ORDER, key=lambda a: acc_ht_score[a])
    most_neg_score = min(ACC_ORDER, key=lambda a: acc_ht_score[a])

    sec_insights.append((
        'INSIGHT 2.6',
        'Mean Hashtag Sentiment Score by Account',
        f'Mean hashtag_sentiment_score (range: -1.0 to +1.0):\n\n'
        + '\n'.join(score_lines) + '\n\n'
        f'  {ACC_LABELS[most_pos_score]} has the most positive mean hashtag score '
        f'({acc_ht_score[most_pos_score]:.4f}), while {ACC_LABELS[most_neg_score]} has the most '
        f'negative ({acc_ht_score[most_neg_score]:.4f}). A score near 0 indicates balanced '
        f'hashtag usage; deviation from 0 indicates a lean toward positive or negative hashtags.',
        f'Mean of hashtag_sentiment_score per _account (NoHashtags posts excluded). '
        f'Score formula: (pos_count − neg_count) ÷ total_matched_hashtags. '
        f'Range: [-1 (all neg), 0 (balanced), +1 (all pos)].'
    ))

    # Insight 2.7 — Caption vs Hashtag strategic disconnect
    cap_dist = compute_grouped_distribution(hc_ht, '_account', 'sentiment_label', SENT_ORDER)
    disconnect_lines = []
    for acc in ACC_ORDER:
        cap_neg = cap_dist.get(acc, {}).get('Negative', {}).get('pct', 0)
        ht_neg  = by_acc_ht.get(acc, {}).get('Negative', {}).get('pct', 0)
        diff = ht_neg - cap_neg
        disconnect_lines.append(
            f'    • {ACC_LABELS[acc]}: Caption Neg={cap_neg:.1f}%, '
            f'Hashtag Neg={ht_neg:.1f}%, Δ={diff:+.1f}pp'
        )

    sec_insights.append((
        'INSIGHT 2.7',
        'Caption vs Hashtag Negativity Gap per Account',
        f'Comparing negative% in captions vs hashtags (Δ = HashtagNeg − CaptionNeg):\n\n'
        + '\n'.join(disconnect_lines) + '\n\n'
        f'  A positive Δ means hashtags are more negative than captions — the account adds '
        f'negative framing via hashtags beyond what the caption text conveys. A negative Δ '
        f'means captions are more negative than hashtags — the account\'s attack messaging '
        f'is in the text rather than the tags.',
        f'Caption Neg% = count(sentiment_label="Negative") ÷ total hashtagged posts per account. '
        f'Hashtag Neg% = count(hashtag_dominant_label="Negative") ÷ same denominator. '
        f'Δ = Hashtag Neg% − Caption Neg%. Only posts with hashtags included in both calculations.'
    ))

    # Insight 2.8 — Conclusion
    sec_insights.append((
        'INSIGHT 2.8',
        'Conclusion — Hashtag Sentiment Landscape',
        f'The hashtag sentiment analysis reveals:\n\n'
        f'  1. Hashtag sentiment is more evenly distributed (Neg {ht_overall["Negative"]["pct"]:.1f}%, '
        f'Neu {ht_overall["Neutral"]["pct"]:.1f}%, Pos {ht_overall["Positive"]["pct"]:.1f}%) '
        f'compared to caption sentiment.\n'
        f'  2. Caption-hashtag agreement is {overall_agree_rate:.1f}%, indicating moderate alignment.\n'
        f'  3. {ACC_LABELS[most_pos_ht]} has the most positive hashtag strategy; '
        f'{ACC_LABELS[most_neg_ht]} the most negative.\n'
        f'  4. {ACC_LABELS[most_vol]} uses the highest volume of hashtags per post.\n'
        f'  5. There are notable caption-hashtag sentiment gaps, suggesting strategic use of '
        f'hashtags for messaging beyond what the caption text alone conveys.\n\n'
        f'  Hashtags serve as a parallel signalling mechanism that may amplify, contradict, '
        f'or extend the sentiment of the caption text.',
        f'Synthesis of insights 2.1–2.7. No new calculations.'
    ))

    all_sections.append(('Hashtag Sentiment Analysis', sec_insights))

    write_insight_report('02_hashtag_sentiment_insights.txt',
                         'HASHTAG SENTIMENT ANALYSIS', all_sections)

    print(f"\n  Done! Generated 8 charts + 1 text report (8 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

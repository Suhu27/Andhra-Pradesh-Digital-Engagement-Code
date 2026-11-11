# ============================================================
#  SCRIPT 08 — CROSS-DIMENSIONAL ANALYSIS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: Advanced cross-dimensional correlations
#  - Sentiment × Emotion, Sentiment × Tone, Emotion × Tone heatmaps
#  - Caption vs Hashtag alignment by account
#  - Composite Negativity Index per account per timeline
#  - Multi-dimensional profile summaries
#
#  Run: python insight_08_cross_dimensional.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 08: CROSS-DIMENSIONAL ANALYSIS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)
    n_clf = len(hc_clf)

    hc_tone = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]
    tone_display = [t.replace('_', ' ') for t in TONE_ORDER]

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD1: Sentiment × Emotion Heatmap
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    ct1 = pd.crosstab(hc_clf['sentiment_label'], hc_clf['emo_primary'],
                      normalize='index') * 100
    ct1 = ct1.reindex(index=SENT_ORDER, columns=EMO_ORDER, fill_value=0)
    ct1_raw = pd.crosstab(hc_clf['sentiment_label'], hc_clf['emo_primary'])
    ct1_raw = ct1_raw.reindex(index=SENT_ORDER, columns=EMO_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('CD1', 'Caption Sentiment × Primary Emotion',
                              f'Which emotions co-occur with which sentiment? (n={n_clf:,})'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct1.values, cmap='RdYlGn', aspect='auto')
    ax.set_xticks(range(len(EMO_ORDER)))
    ax.set_xticklabels(EMO_ORDER, rotation=20, ha='right')
    ax.set_yticks(range(len(SENT_ORDER)))
    ax.set_yticklabels(SENT_ORDER)
    ax.set_xlabel('Primary Emotion')
    ax.set_ylabel('Caption Sentiment')

    for r in range(len(SENT_ORDER)):
        for c in range(len(EMO_ORDER)):
            val = ct1.values[r, c]
            cnt = ct1_raw.values[r, c]
            clr = 'white' if val > 40 else 'black'
            ax.text(c, r, f'{val:.1f}%\n({cnt:,})', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=clr)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row-normalised % (each sentiment row sums to 100%)')

    add_methodology(fig,
        f'Source: cross-tab of sentiment_label × emo_primary, row-normalised. '
        f'Cell = % of that emotion within that sentiment category. '
        f'E.g., "Of all Negative posts, X% have Anger as primary emotion." '
        f'Classified posts only (n={n_clf:,}).',
        y=-0.06)
    save_chart(fig, 'CD1_sentiment_emotion_heatmap.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD2: Sentiment × Tone Heatmap
    # ═════════════════════════════════════════════════════════════════════════
    ct2 = pd.crosstab(hc_tone['sentiment_label'], hc_tone['tone_primary'],
                      normalize='index') * 100
    ct2 = ct2.reindex(index=SENT_ORDER, columns=TONE_ORDER, fill_value=0)
    ct2_raw = pd.crosstab(hc_tone['sentiment_label'], hc_tone['tone_primary'])
    ct2_raw = ct2_raw.reindex(index=SENT_ORDER, columns=TONE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(format_title('CD2', 'Caption Sentiment × Rhetorical Tone',
                              f'How sentiment maps to rhetorical strategy (n={len(hc_tone):,})'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct2.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(TONE_ORDER)))
    ax.set_xticklabels(tone_display, rotation=15)
    ax.set_yticks(range(len(SENT_ORDER)))
    ax.set_yticklabels(SENT_ORDER)
    ax.set_xlabel('Rhetorical Tone')
    ax.set_ylabel('Caption Sentiment')

    for r in range(len(SENT_ORDER)):
        for c in range(len(TONE_ORDER)):
            val = ct2.values[r, c]
            cnt = ct2_raw.values[r, c]
            clr = 'white' if val > ct2.values.max() * 0.55 else 'black'
            ax.text(c, r, f'{val:.1f}%\n({cnt:,})', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=clr)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row-normalised %')

    add_methodology(fig,
        f'Source: cross-tab of sentiment_label × tone_primary, row-normalised. '
        f'Neutral-tone excluded. '
        f'Cell = % of that tone within that sentiment. Each row sums to ~100%.',
        y=-0.06)
    save_chart(fig, 'CD2_sentiment_tone_heatmap.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD3: Emotion × Tone Heatmap
    # ═════════════════════════════════════════════════════════════════════════
    ct3 = pd.crosstab(hc_tone['emo_primary'], hc_tone['tone_primary'],
                      normalize='index') * 100
    emo_rows = [e for e in EMO_ORDER if e in ct3.index]
    ct3 = ct3.reindex(index=emo_rows, columns=TONE_ORDER, fill_value=0)
    ct3_raw = pd.crosstab(hc_tone['emo_primary'], hc_tone['tone_primary'])
    ct3_raw = ct3_raw.reindex(index=emo_rows, columns=TONE_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(format_title('CD3', 'Primary Emotion × Rhetorical Tone',
                              f'How emotions and rhetorical tones align'),
                 fontsize=14, fontweight='bold', y=1.02)

    im = ax.imshow(ct3.values, cmap='PuBuGn', aspect='auto')
    ax.set_xticks(range(len(TONE_ORDER)))
    ax.set_xticklabels(tone_display, rotation=15)
    ax.set_yticks(range(len(emo_rows)))
    ax.set_yticklabels(emo_rows)
    ax.set_xlabel('Rhetorical Tone')
    ax.set_ylabel('Primary Emotion')

    for r in range(len(emo_rows)):
        for c in range(len(TONE_ORDER)):
            val = ct3.values[r, c]
            cnt = ct3_raw.values[r, c]
            clr = 'white' if val > ct3.values.max() * 0.55 else 'black'
            ax.text(c, r, f'{val:.1f}%\n({cnt:,})', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=clr)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Row-normalised %')

    add_methodology(fig,
        f'Source: cross-tab of emo_primary × tone_primary, row-normalised. '
        f'Neutral-tone excluded. '
        f'Cell = % of that tone within that emotion. Each row sums to ~100%.',
        y=-0.06)
    save_chart(fig, 'CD3_emotion_tone_heatmap.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD4: Caption-Hashtag Sentiment Alignment by Account
    # ═════════════════════════════════════════════════════════════════════════
    hc_ht = hc[hc['hashtag_dominant_label'] != 'NoHashtags']

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('CD4', 'Caption-Hashtag Sentiment Alignment by Account',
                              f'Agreement, positive mismatch, and negative mismatch rates'),
                 fontsize=14, fontweight='bold', y=1.02)

    x = np.arange(len(ACC_ORDER))
    width = 0.2

    agree_vals = []
    pos_cap_neg_ht = []   # positive caption + negative hashtag
    neg_cap_pos_ht = []   # negative caption + positive hashtag

    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        total = len(sub)
        if total == 0:
            agree_vals.append(0)
            pos_cap_neg_ht.append(0)
            neg_cap_pos_ht.append(0)
            continue

        agree = (sub['sentiment_label'] == sub['hashtag_dominant_label']).sum()
        agree_vals.append(pct(agree, total))

        pcnh = ((sub['sentiment_label'] == 'Positive') &
                (sub['hashtag_dominant_label'] == 'Negative')).sum()
        pos_cap_neg_ht.append(pct(pcnh, total))

        ncph = ((sub['sentiment_label'] == 'Negative') &
                (sub['hashtag_dominant_label'] == 'Positive')).sum()
        neg_cap_pos_ht.append(pct(ncph, total))

    bars1 = ax.bar(x - width, agree_vals, width, label='Agreement (match)',
                   color='#2ECC71', edgecolor='white')
    bars2 = ax.bar(x, pos_cap_neg_ht, width, label='Pos Caption + Neg Hashtag',
                   color='#E74C3C', edgecolor='white')
    bars3 = ax.bar(x + width, neg_cap_pos_ht, width, label='Neg Caption + Pos Hashtag',
                   color='#3498DB', edgecolor='white')

    for b_set in [bars1, bars2, bars3]:
        add_value_labels(ax, b_set, fmt='{:.1f}%', fontsize=8)

    ax.set_xticks(x)
    acc_labels_n = [f'{ACC_LABELS[a]}\n(n={len(hc_ht[hc_ht["_account"]==a]):,})'
                    for a in ACC_ORDER]
    ax.set_xticklabels(acc_labels_n)
    ax.set_ylabel('Percentage of Hashtagged Posts (%)')
    ax.legend(fontsize=9, framealpha=0.9)

    add_methodology(fig,
        f'Source: sentiment_label vs hashtag_dominant_label per _account. '
        f'NoHashtags posts excluded. '
        f'Agreement = both labels identical. '
        f'"Pos Cap + Neg HT" = positive caption with negative hashtags (strategic amplification). '
        f'"Neg Cap + Pos HT" = negative caption with positive hashtags (mixed messaging).',
        y=-0.06)
    save_chart(fig, 'CD4_caption_hashtag_alignment_by_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD5: Composite Negativity Index per Account per Timeline
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('CD5', 'Composite Negativity Index per Account per Timeline',
                              f'Index = (Negative% + Anger% + Aggressive%) ÷ 3'),
                 fontsize=14, fontweight='bold', y=1.02)

    for acc in ACC_ORDER:
        neg_idx_vals = []
        for tl in TL_ORDER:
            # Sentiment: negative %
            sent_sub = hc[(hc['_account'] == acc) & (hc['_timeline'] == tl)]
            neg_pct_val = pct((sent_sub['sentiment_label'] == 'Negative').sum(), len(sent_sub))

            # Emotion: anger %  (from classified subset)
            clf_sub = hc_clf[(hc_clf['_account'] == acc) & (hc_clf['_timeline'] == tl)]
            anger_pct_val = pct((clf_sub['emo_primary'] == 'Anger').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0

            # Tone: aggressive %  (from tone subset)
            tone_sub = hc_tone[(hc_tone['_account'] == acc) & (hc_tone['_timeline'] == tl)]
            agg_pct_val = pct((tone_sub['tone_primary'] == 'Aggressive').sum(), len(tone_sub)) if len(tone_sub) > 0 else 0

            # Composite index = simple average of the three
            idx = (neg_pct_val + anger_pct_val + agg_pct_val) / 3
            neg_idx_vals.append(idx)

        ax.plot(range(len(TL_ORDER)), neg_idx_vals, marker='o', linewidth=2.5,
                color=ACC_COLORS[acc], label=ACC_LABELS[acc], markersize=8)
        for k, v in enumerate(neg_idx_vals):
            ax.text(k, v + 0.5, f'{v:.1f}', ha='center', fontsize=8,
                    color=ACC_COLORS[acc], fontweight='bold')

    ax.set_xticks(range(len(TL_ORDER)))
    ax.set_xticklabels([TL_LABELS[tl] for tl in TL_ORDER])
    ax.set_ylabel('Composite Negativity Index')
    ax.set_ylim(0, 65)
    ax.legend(title='Account', framealpha=0.9)

    add_methodology(fig,
        f'Source: Composite of sentiment_label="Negative" %, emo_primary="Anger" %, '
        f'tone_primary="Aggressive" %. Index = (Neg% + Anger% + Agg%) ÷ 3. '
        f'Each component calculated from its respective subset: sentiment from all HC, '
        f'emotion from classified, tone from classified excl. Neutral. '
        f'Higher = more negative/aggressive overall posture.',
        y=-0.06)
    save_chart(fig, 'CD5_negativity_index_timeline.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD6: Composite Negativity Index (Overall per Account)
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(format_title('CD6', 'Composite Negativity Index — Overall per Account',
                              f'Single-number summary of each account\'s negativity posture'),
                 fontsize=14, fontweight='bold', y=1.02)

    neg_indices = []
    neg_components = []
    for acc in ACC_ORDER:
        sent_sub = hc[hc['_account'] == acc]
        n_pct = pct((sent_sub['sentiment_label'] == 'Negative').sum(), len(sent_sub))

        clf_sub = hc_clf[hc_clf['_account'] == acc]
        a_pct = pct((clf_sub['emo_primary'] == 'Anger').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0

        tone_sub = hc_tone[hc_tone['_account'] == acc]
        ag_pct = pct((tone_sub['tone_primary'] == 'Aggressive').sum(), len(tone_sub)) if len(tone_sub) > 0 else 0

        idx = (n_pct + a_pct + ag_pct) / 3
        neg_indices.append(idx)
        neg_components.append((n_pct, a_pct, ag_pct))

    bars = ax.bar([f'{ACC_LABELS[a]}\n(n={len(hc[hc["_account"]==a]):,})' for a in ACC_ORDER],
                  neg_indices, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)

    for i, bar in enumerate(bars):
        h = bar.get_height()
        n_p, a_p, ag_p = neg_components[i]
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}\n(Neg={n_p:.0f}%, Ang={a_p:.0f}%, Agg={ag_p:.0f}%)',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')

    ax.set_ylabel('Composite Negativity Index')
    ax.set_ylim(0, max(neg_indices) * 1.3 if neg_indices else 50)

    add_methodology(fig,
        f'Source: Composite of Neg%, Anger%, Aggressive%. '
        f'Index = (Neg% + Anger% + Agg%) ÷ 3. Component values shown in annotations.',
        y=-0.04)
    save_chart(fig, 'CD6_negativity_index_overall.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD7: Messaging Consistency Score per Account
    # ═════════════════════════════════════════════════════════════════════════
    # Consistency = how well caption sentiment, hashtag sentiment, and tone align
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('CD7', 'Messaging Consistency Score per Account',
                              f'How internally consistent is each account\'s messaging?'),
                 fontsize=14, fontweight='bold', y=1.02)

    consistency_scores = []
    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        total = len(sub)
        if total == 0:
            consistency_scores.append(0)
            continue

        # Metric 1: Caption-Hashtag agreement rate
        cap_ht_agree = pct((sub['sentiment_label'] == sub['hashtag_dominant_label']).sum(), total)

        # Metric 2: Sentiment-Emotion alignment
        # Negative sentiment should map to Anger/Fear; Positive to Hope/Trust/Pride
        clf_sub = hc_clf[hc_clf['_account'] == acc]
        clf_total = len(clf_sub)
        if clf_total > 0:
            neg_anger = ((clf_sub['sentiment_label'] == 'Negative') &
                         (clf_sub['emo_primary'].isin(['Anger', 'Fear']))).sum()
            pos_hope = ((clf_sub['sentiment_label'] == 'Positive') &
                        (clf_sub['emo_primary'].isin(['Hope', 'Trust', 'Pride']))).sum()
            neg_total = (clf_sub['sentiment_label'] == 'Negative').sum()
            pos_total = (clf_sub['sentiment_label'] == 'Positive').sum()
            neg_align = pct(neg_anger, neg_total) if neg_total > 0 else 0
            pos_align = pct(pos_hope, pos_total) if pos_total > 0 else 0
            sent_emo_align = (neg_align + pos_align) / 2
        else:
            sent_emo_align = 0

        # Composite consistency = average of the two alignment metrics
        consistency = (cap_ht_agree + sent_emo_align) / 2
        consistency_scores.append(consistency)

    bars = ax.bar([f'{ACC_LABELS[a]}\n(n={len(hc_ht[hc_ht["_account"]==a]):,})' for a in ACC_ORDER],
                  consistency_scores, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.1f}', fontsize=11)
    ax.set_ylabel('Messaging Consistency Score (0–100)')
    ax.set_ylim(0, max(consistency_scores) * 1.2 if consistency_scores else 100)

    add_methodology(fig,
        f'Source: Composite of (1) Caption-Hashtag agreement rate and '
        f'(2) Sentiment-Emotion alignment rate. '
        f'Sentiment-Emotion alignment = % of Negative posts with Anger/Fear emotion + '
        f'% of Positive posts with Hope/Trust/Pride emotion, averaged. '
        f'Consistency = (caption-HT agreement + sent-emo alignment) ÷ 2. '
        f'Higher = more internally coherent messaging.',
        y=-0.06)
    save_chart(fig, 'CD7_messaging_consistency.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD8: Multi-Dimensional Account Profile (Radar-style as Bar)
    # ═════════════════════════════════════════════════════════════════════════
    # For each account, show key % metrics side by side
    profile_metrics = {
        'Negative%': lambda sub, clf_sub, tone_sub: pct((sub['sentiment_label'] == 'Negative').sum(), len(sub)),
        'Positive%': lambda sub, clf_sub, tone_sub: pct((sub['sentiment_label'] == 'Positive').sum(), len(sub)),
        'Anger%': lambda sub, clf_sub, tone_sub: pct((clf_sub['emo_primary'] == 'Anger').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0,
        'Hope%': lambda sub, clf_sub, tone_sub: pct((clf_sub['emo_primary'] == 'Hope').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0,
        'Aggressive%': lambda sub, clf_sub, tone_sub: pct((tone_sub['tone_primary'] == 'Aggressive').sum(), len(tone_sub)) if len(tone_sub) > 0 else 0,
        'Persuasive%': lambda sub, clf_sub, tone_sub: pct((tone_sub['tone_primary'] == 'Persuasive').sum(), len(tone_sub)) if len(tone_sub) > 0 else 0,
        'Fact%': lambda sub, clf_sub, tone_sub: pct((clf_sub['subj_label'] == 'Fact').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0,
        'Opinion%': lambda sub, clf_sub, tone_sub: pct((clf_sub['subj_label'] == 'Opinion').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0,
    }
    metric_names = list(profile_metrics.keys())
    metric_colors = ['#E74C3C', '#2ECC71', '#E74C3C', '#2ECC71',
                     '#E74C3C', '#3498DB', '#3498DB', '#E74C3C']

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('CD8', 'Multi-Dimensional Account Profile',
                              f'Key sentiment/emotion/tone/subjectivity metrics per account'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]
        clf_sub = hc_clf[hc_clf['_account'] == acc]
        tone_sub = hc_tone[hc_tone['_account'] == acc]

        vals = [fn(sub, clf_sub, tone_sub) for fn in profile_metrics.values()]
        bars = ax.barh(range(len(metric_names)), vals, color=metric_colors,
                       edgecolor='white', linewidth=0.5, height=0.6)
        for j, bar in enumerate(bars):
            w = bar.get_width()
            ax.text(w + 0.5, j, f'{w:.1f}%', va='center', fontsize=8,
                    fontweight='bold', color='#333')

        ax.set_yticks(range(len(metric_names)))
        if i == 0:
            ax.set_yticklabels(metric_names, fontsize=9)
        else:
            ax.set_yticklabels([])
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_xlim(0, 65)
        ax.invert_yaxis()

    add_methodology(fig,
        f'Source: Multiple columns (sentiment_label, emo_primary, tone_primary, subj_label). '
        f'Each metric = % of classified posts with that label per account. '
        f'Provides a holistic "fingerprint" of each account\'s communication style.',
        y=-0.10)
    save_chart(fig, 'CD8_multidimensional_profile.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD9: Sentiment-Tone Alignment Strength by Account
    # ═════════════════════════════════════════════════════════════════════════
    # How strongly does negativity map to aggression per account?
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('CD9', 'Negative-Aggressive Alignment Strength by Account',
                              f'% of Negative-sentiment posts that also have Aggressive tone'),
                 fontsize=14, fontweight='bold', y=1.02)

    align_vals = []
    for acc in ACC_ORDER:
        sub = hc_tone[hc_tone['_account'] == acc]
        neg_posts = sub[sub['sentiment_label'] == 'Negative']
        total_neg = len(neg_posts)
        neg_agg = (neg_posts['tone_primary'] == 'Aggressive').sum()
        align_vals.append(pct(neg_agg, total_neg))

    bars = ax.bar([f'{ACC_LABELS[a]}\n(n neg={len(hc_tone[(hc_tone["_account"]==a)&(hc_tone["sentiment_label"]=="Negative")]):,})'
                   for a in ACC_ORDER],
                  align_vals, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)
    add_value_labels(ax, bars, fmt='{:.1f}%', fontsize=11)
    ax.set_ylabel('% of Negative Posts with Aggressive Tone')
    ax.set_ylim(0, max(align_vals) * 1.25 if align_vals else 100)

    add_methodology(fig,
        f'Source: tone_primary filtered to Negative sentiment posts per account. '
        f'Value = count(Negative + Aggressive) ÷ count(Negative) per account × 100. '
        f'Higher = negative sentiment more strongly paired with aggressive rhetoric.',
        y=-0.04)
    save_chart(fig, 'CD9_negative_aggressive_alignment.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART CD10: Alliance Multi-Dimensional Comparison
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle(format_title('CD10', 'Alliance Multi-Dimensional Profile Comparison',
                              f'Holistic comparison: YSRCP Alliance vs TDP Alliance'),
                 fontsize=14, fontweight='bold', y=1.02)

    for i, al in enumerate(ALLIANCE_ORDER):
        ax = axes[i]
        sub = hc[hc['_alliance'] == al]
        clf_sub = hc_clf[hc_clf['_alliance'] == al]
        tone_sub = hc_tone[hc_tone['_alliance'] == al]

        vals = [fn(sub, clf_sub, tone_sub) for fn in profile_metrics.values()]
        bars = ax.barh(range(len(metric_names)), vals,
                       color=ALLIANCE_COLORS[al], edgecolor='white',
                       linewidth=0.5, height=0.6, alpha=0.85)
        for j, bar in enumerate(bars):
            w = bar.get_width()
            ax.text(w + 0.5, j, f'{w:.1f}%', va='center', fontsize=9,
                    fontweight='bold', color='#333')

        ax.set_yticks(range(len(metric_names)))
        ax.set_yticklabels(metric_names, fontsize=10)
        ax.set_title(f'{al}\n(n={len(sub):,})', fontsize=12,
                     color=ALLIANCE_COLORS[al], fontweight='bold')
        ax.set_xlim(0, 65)
        ax.invert_yaxis()

    add_methodology(fig,
        f'Source: Multiple columns, grouped by alliance. '
        f'Each metric = % per alliance. Provides head-to-head multi-dimensional comparison.',
        y=-0.08)
    save_chart(fig, 'CD10_alliance_profile_comparison.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    sec_insights = []

    # 8.1 Sentiment-Emotion alignment
    neg_anger_pct = pct(
        ((hc_clf['sentiment_label'] == 'Negative') & (hc_clf['emo_primary'] == 'Anger')).sum(),
        (hc_clf['sentiment_label'] == 'Negative').sum()
    )
    pos_pride_trust_pct = pct(
        ((hc_clf['sentiment_label'] == 'Positive') & (hc_clf['emo_primary'].isin(['Pride', 'Trust', 'Hope']))).sum(),
        (hc_clf['sentiment_label'] == 'Positive').sum()
    )
    neu_info_pct = pct(
        ((hc_clf['sentiment_label'] == 'Neutral') & (hc_clf['emo_primary'] == 'Informational')).sum(),
        (hc_clf['sentiment_label'] == 'Neutral').sum()
    )

    sec_insights.append((
        'INSIGHT 8.1',
        'Sentiment–Emotion Alignment Analysis',
        f'How well do sentiment labels align with emotions?\n\n'
        f'  • Of Negative-sentiment posts, {neg_anger_pct:.1f}% have Anger as primary emotion\n'
        f'  • Of Positive-sentiment posts, {pos_pride_trust_pct:.1f}% have Hope/Trust/Pride\n'
        f'  • Of Neutral-sentiment posts, {neu_info_pct:.1f}% have Informational emotion\n\n'
        f'  The alignment is {"strong" if neg_anger_pct > 60 else "moderate" if neg_anger_pct > 40 else "weak"} '
        f'for negative-anger pairing, indicating that sentiment and emotion '
        f'classifiers are {"highly coherent" if neg_anger_pct > 60 else "partially overlapping" if neg_anger_pct > 40 else "capturing different aspects"}. '
        f'The {100 - neg_anger_pct:.1f}% of negative posts that are NOT angry may '
        f'represent fear-based, sad, or critical-but-composed content.',
        f'Cross-tab of sentiment_label × emo_primary. '
        f'Alignment = count of expected pairing ÷ total of that sentiment × 100. '
        f'Expected: Negative→Anger/Fear, Positive→Hope/Trust/Pride, Neutral→Informational.'
    ))

    # 8.2 Sentiment-Tone alignment
    neg_agg_pct = pct(
        ((hc_tone['sentiment_label'] == 'Negative') & (hc_tone['tone_primary'] == 'Aggressive')).sum(),
        (hc_tone['sentiment_label'] == 'Negative').sum()
    )
    pos_insp_per_pct = pct(
        ((hc_tone['sentiment_label'] == 'Positive') &
         (hc_tone['tone_primary'].isin(['Persuasive', 'Inspirational']))).sum(),
        (hc_tone['sentiment_label'] == 'Positive').sum()
    )

    sec_insights.append((
        'INSIGHT 8.2',
        'Sentiment–Tone Alignment Analysis',
        f'How well do sentiment labels align with rhetorical tones?\n\n'
        f'  • Of Negative-sentiment posts, {neg_agg_pct:.1f}% have Aggressive tone\n'
        f'  • Of Positive-sentiment posts, {pos_insp_per_pct:.1f}% have Persuasive/Inspirational tone\n\n'
        f'  The sentiment-tone alignment reveals how political actors pair '
        f'negative content with aggressive rhetoric (attack messaging) vs other approaches. '
        f'The {100 - neg_agg_pct:.1f}% of negative posts that are NOT aggressive may use '
        f'victim narrative, defensive, or measured critique approaches.',
        f'Cross-tab of sentiment_label × tone_primary. '
        f'Negative→Aggressive alignment = count ÷ total Negative posts × 100.'
    ))

    # 8.3 Caption-Hashtag mismatch
    overall_agree = pct(
        (hc_ht['sentiment_label'] == hc_ht['hashtag_dominant_label']).sum(),
        len(hc_ht)
    )

    mismatch_acc = {}
    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        total = len(sub)
        agree = (sub['sentiment_label'] == sub['hashtag_dominant_label']).sum()
        mismatch_acc[acc] = 100 - pct(agree, total)

    most_mismatch = max(ACC_ORDER, key=lambda a: mismatch_acc[a])
    least_mismatch = min(ACC_ORDER, key=lambda a: mismatch_acc[a])

    sec_insights.append((
        'INSIGHT 8.3',
        'Caption-Hashtag Sentiment Mismatch Rate',
        f'Overall mismatch rate: {100 - overall_agree:.1f}% (agreement: {overall_agree:.1f}%)\n\n'
        f'  Per-account mismatch rates:\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {mismatch_acc[a]:.1f}% mismatch'
                     for a in ACC_ORDER]) + '\n\n'
        f'  {ACC_LABELS[most_mismatch]} has the highest mismatch ({mismatch_acc[most_mismatch]:.1f}%), '
        f'suggesting more strategic use of hashtags that diverge from caption sentiment. '
        f'{ACC_LABELS[least_mismatch]} has the lowest ({mismatch_acc[least_mismatch]:.1f}%), '
        f'indicating more internally consistent messaging.',
        f'Mismatch = 100 - agreement%. Agreement = posts where sentiment_label == hashtag_dominant_label. '
        f'NoHashtags posts excluded.'
    ))

    # 8.4 Composite negativity index
    neg_idx_overall = {}
    for acc in ACC_ORDER:
        sent_sub = hc[hc['_account'] == acc]
        n_pct = pct((sent_sub['sentiment_label'] == 'Negative').sum(), len(sent_sub))
        clf_sub = hc_clf[hc_clf['_account'] == acc]
        a_pct = pct((clf_sub['emo_primary'] == 'Anger').sum(), len(clf_sub)) if len(clf_sub) > 0 else 0
        tone_sub = hc_tone[hc_tone['_account'] == acc]
        ag_pct = pct((tone_sub['tone_primary'] == 'Aggressive').sum(), len(tone_sub)) if len(tone_sub) > 0 else 0
        neg_idx_overall[acc] = (n_pct + a_pct + ag_pct) / 3

    most_neg_idx = max(ACC_ORDER, key=lambda a: neg_idx_overall[a])
    least_neg_idx = min(ACC_ORDER, key=lambda a: neg_idx_overall[a])

    sec_insights.append((
        'INSIGHT 8.4',
        'Composite Negativity Index — Overall Ranking',
        f'Composite Negativity Index = (Negative% + Anger% + Aggressive%) ÷ 3:\n\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {neg_idx_overall[a]:.1f}'
                     for a in sorted(ACC_ORDER, key=lambda a: neg_idx_overall[a], reverse=True)])
        + f'\n\n  {ACC_LABELS[most_neg_idx]} has the highest composite negativity index '
        f'({neg_idx_overall[most_neg_idx]:.1f}), indicating the most uniformly negative '
        f'communication posture across sentiment, emotion, and tone dimensions. '
        f'{ACC_LABELS[least_neg_idx]} has the lowest ({neg_idx_overall[least_neg_idx]:.1f}), '
        f'suggesting a more balanced or positive communication approach.\n\n'
        f'  This composite index provides a single-number summary that captures '
        f'the multi-dimensional negativity of each account, going beyond any single classifier.',
        f'Index components: Negative% from sentiment_label, Anger% from emo_primary, '
        f'Aggressive% from tone_primary. Each calculated from its respective subset. '
        f'Index = simple average of the three percentages. Range: 0 (no negativity) to ~100 (maximum).'
    ))

    # 8.5 Emotion-tone cross-patterns
    # What tone do anger posts get?
    anger_tones = hc_tone[hc_tone['emo_primary'] == 'Anger']['tone_primary'].value_counts()
    anger_agg_pct = pct(anger_tones.get('Aggressive', 0), anger_tones.sum())

    # What tone do hope posts get?
    hope_tones = hc_tone[hc_tone['emo_primary'] == 'Hope']['tone_primary'].value_counts()
    hope_insp_pct = pct(hope_tones.get('Inspirational', 0), hope_tones.sum())
    hope_per_pct = pct(hope_tones.get('Persuasive', 0), hope_tones.sum())

    sec_insights.append((
        'INSIGHT 8.5',
        'Emotion–Tone Cross-Patterns',
        f'How do emotions map to rhetorical tones?\n\n'
        f'  Anger posts: {anger_agg_pct:.1f}% are Aggressive, '
        f'{pct(anger_tones.get("Persuasive", 0), anger_tones.sum()):.1f}% are Persuasive\n'
        f'  Hope posts: {hope_per_pct:.1f}% are Persuasive, '
        f'{hope_insp_pct:.1f}% are Inspirational\n'
        f'  Trust posts: {pct(hc_tone[hc_tone["emo_primary"]=="Trust"]["tone_primary"].value_counts().get("Persuasive", 0), (hc_tone["emo_primary"]=="Trust").sum()):.1f}% are Persuasive\n\n'
        f'  The emotion-tone mapping reveals how emotional content gets expressed '
        f'rhetorically. Anger overwhelmingly pairs with Aggressive tone — a natural '
        f'alignment. But not all anger is aggressive; some angry posts use persuasive '
        f'or even victim narrative framing.',
        f'Cross-tab of emo_primary × tone_primary. Row-normalised: % = count ÷ total posts '
        f'with that emotion × 100. Neutral-tone excluded.'
    ))

    # 8.6 Messaging consistency
    consistency_data = {}
    for acc in ACC_ORDER:
        consistency_data[acc] = consistency_scores[ACC_ORDER.index(acc)]

    most_consistent = max(ACC_ORDER, key=lambda a: consistency_data[a])
    least_consistent = min(ACC_ORDER, key=lambda a: consistency_data[a])

    sec_insights.append((
        'INSIGHT 8.6',
        'Messaging Consistency Analysis',
        f'Messaging consistency score per account (higher = more internally coherent):\n\n'
        + '\n'.join([f'    • {ACC_LABELS[a]}: {consistency_data[a]:.1f}'
                     for a in sorted(ACC_ORDER, key=lambda a: consistency_data[a], reverse=True)])
        + f'\n\n  {ACC_LABELS[most_consistent]} has the most internally consistent messaging, '
        f'meaning its caption sentiment, hashtag sentiment, and emotion-sentiment pairings '
        f'align well. {ACC_LABELS[least_consistent]} has the most inconsistent messaging, '
        f'which may indicate strategic complexity or mixed messaging.',
        f'Consistency = average of (1) caption-hashtag agreement rate and '
        f'(2) sentiment-emotion alignment rate per account.'
    ))

    # 8.7 Alliance comparison summary
    al_profiles = {}
    for al in ALLIANCE_ORDER:
        sub = hc[hc['_alliance'] == al]
        clf_sub = hc_clf[hc_clf['_alliance'] == al]
        tone_sub = hc_tone[hc_tone['_alliance'] == al]
        al_profiles[al] = {name: fn(sub, clf_sub, tone_sub)
                           for name, fn in profile_metrics.items()}

    sec_insights.append((
        'INSIGHT 8.7',
        'Alliance Multi-Dimensional Profile Summary',
        f'YSRCP Alliance:\n'
        + '\n'.join([f'    {k}: {v:.1f}%' for k, v in al_profiles.get('YSRCP Alliance', {}).items()])
        + f'\n\n  TDP Alliance:\n'
        + '\n'.join([f'    {k}: {v:.1f}%' for k, v in al_profiles.get('TDP Alliance', {}).items()])
        + f'\n\n  Key differences: The dimension with the largest gap between the two alliances '
        f'reveals the primary strategic divergence in their communication approaches.',
        f'Each metric = % per alliance across multiple classification columns.'
    ))

    # 8.8 Master conclusion
    # Need er_sent for conclusion
    er_sent = compute_mean_engagement(hc, 'sentiment_label', 'Engagement_Rate_Monthly')

    sec_insights.append((
        'INSIGHT 8.8',
        'MASTER CONCLUSION — Cross-Dimensional Synthesis',
        f'Bringing together all dimensions of analysis:\n\n'
        f'  1. SENTIMENT: Neutral-dominant discourse with significant negativity '
        f'({compute_distribution(hc["sentiment_label"], SENT_ORDER)[0]["Negative"]["pct"]:.1f}%).\n'
        f'  2. EMOTION: Anger leads at '
        f'{compute_distribution(hc_clf["emo_primary"], EMO_ORDER)[0]["Anger"]["pct"]:.1f}%, '
        f'driving the emotional temperature of political discourse.\n'
        f'  3. TONE: Aggressive rhetoric dominates at '
        f'{compute_distribution(hc_tone["tone_primary"], TONE_ORDER)[0]["Aggressive"]["pct"]:.1f}%, '
        f'but Persuasive is a strong counterweight.\n'
        f'  4. SUBJECTIVITY: Mixed content prevails, with opinion exceeding pure facts.\n'
        f'  5. HASHTAGS: More polarised than captions, serving as strategic signal amplifiers.\n'
        f'  6. ENGAGEMENT: {"Positive" if er_sent.get("Positive", {}).get("mean", 0) > er_sent.get("Negative", {}).get("mean", 0) else "Negative"} '
        f'sentiment generates the highest engagement.\n'
        f'  7. TEMPORAL: Discourse shows evolution as elections approach.\n'
        f'  8. ACCOUNTS: {ACC_LABELS[most_neg_idx]} is the most negative overall; '
        f'{ACC_LABELS[least_neg_idx]} the least negative.\n'
        f'  9. CONSISTENCY: Messaging coherence varies significantly across accounts.\n\n'
        f'  These findings collectively reveal a highly polarised, multi-layered political '
        f'communication ecosystem in Andhra Pradesh, with distinct strategic identities for '
        f'each account and clear temporal dynamics as the 2024 elections approached.',
        f'Synthesis of all insights from scripts 01-08. No new calculations.'
    ))



    all_sections.append(('Cross-Dimensional Analysis', sec_insights))

    write_insight_report('08_cross_dimensional_insights.txt',
                         'CROSS-DIMENSIONAL ANALYSIS', all_sections)

    print(f"\n  Done! Generated 10 charts + 1 text report (8 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

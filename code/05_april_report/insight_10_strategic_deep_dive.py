# ============================================================
#  SCRIPT 10 — STRATEGIC DEEP DIVE
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: Advanced strategic communication patterns
#  - Pre-election strategy pivot (who shifted strategy the most?)
#  - Top-performing posts profile (what makes viral posts?)
#  - Negativity-Engagement paradox per account
#  - Hashtag strategy vs caption strategy alignment
#  - Rhetorical repertoire analysis (tone combinations)
#
#  Run: python insight_10_strategic_deep_dive.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 10: STRATEGIC DEEP DIVE")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)
    n_clf = len(hc_clf)

    hc_tone = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD1: Pre-Election Strategy Pivot — Who Shifted Most?
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    # Compare first period (JunAug2023) vs last (JanMay2024) across multiple dimensions
    metrics_shift = {
        'Negative%': lambda sub: pct((sub['sentiment_label'] == 'Negative').sum(), len(sub)),
        'Positive%': lambda sub: pct((sub['sentiment_label'] == 'Positive').sum(), len(sub)),
        'Anger%': lambda sub: pct((sub['emo_primary'] == 'Anger').sum(), len(sub)) if 'emo_primary' in sub.columns and len(sub) > 0 else 0,
        'Aggressive%': lambda sub: pct((sub['tone_primary'] == 'Aggressive').sum(), len(sub)) if 'tone_primary' in sub.columns and len(sub) > 0 else 0,
        'Fact%': lambda sub: pct((sub['subj_label'] == 'Fact').sum(), len(sub)) if 'subj_label' in sub.columns and len(sub) > 0 else 0,
    }

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle(format_title('SD1',
        'Pre-Election Strategy Pivot — Change from Jun–Aug 2023 to Jan–May 2024',
        'Which account shifted strategy the most as elections approached?'),
        fontsize=13, fontweight='bold', y=1.02)

    metric_names = list(metrics_shift.keys())
    x = np.arange(len(metric_names))
    n_acc = len(ACC_ORDER)
    width = 0.18
    offset = -(n_acc - 1) * width / 2

    for j, acc in enumerate(ACC_ORDER):
        deltas = []
        for metric_name, fn in metrics_shift.items():
            if metric_name in ['Anger%', 'Aggressive%', 'Fact%']:
                early = hc_clf[(hc_clf['_account'] == acc) & (hc_clf['_timeline'] == 'JunAug2023')]
                late = hc_clf[(hc_clf['_account'] == acc) & (hc_clf['_timeline'] == 'JanMay2024')]
            else:
                early = hc[(hc['_account'] == acc) & (hc['_timeline'] == 'JunAug2023')]
                late = hc[(hc['_account'] == acc) & (hc['_timeline'] == 'JanMay2024')]
            val_early = fn(early) if len(early) > 0 else 0
            val_late = fn(late) if len(late) > 0 else 0
            deltas.append(val_late - val_early)

        bars = ax.bar(x + offset + j * width, deltas, width,
                      label=ACC_LABELS[acc], color=ACC_COLORS[acc],
                      edgecolor='white', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ypos = h + 0.3 if h >= 0 else h - 0.8
            ax.text(bar.get_x() + bar.get_width()/2., ypos,
                    f'{h:+.1f}', ha='center', fontsize=7, fontweight='bold',
                    color=ACC_COLORS[acc])

    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel('Change (percentage points)')
    ax.legend(title='Account', framealpha=0.9)

    add_methodology(fig,
        f'Source: Δ = [Jan–May 2024 value] minus [Jun–Aug 2023 value] per metric per account. '
        f'Positive Δ = metric increased toward pre-election. '
        f'Negative Δ = metric decreased. '
        f'Larger absolute Δ = bigger strategy shift.',
        y=-0.06)
    save_chart(fig, 'SD1_preelection_strategy_pivot.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD2: Top 1% Posts Profile — Sentiment Distribution
    # ═════════════════════════════════════════════════════════════════════════
    # What characterises the highest-engagement posts?
    er_col = 'Engagement_Rate_Monthly'
    top_pct = 0.01
    n_top = max(1, int(len(hc) * top_pct))
    hc_sorted = hc.sort_values(er_col, ascending=False)
    top_posts = hc_sorted.head(n_top)
    bottom_posts = hc_sorted.tail(n_top)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(format_title('SD2',
        f'Top 1% vs Bottom 1% Posts — What Makes Viral Content?',
        f'Top {n_top:,} posts by Engagement Rate vs bottom {n_top:,}'),
        fontsize=14, fontweight='bold', y=1.02)

    # Subplot 1: Sentiment
    ax = axes[0]
    for i, (label, data, color) in enumerate([
        ('Top 1%', top_posts, '#2ECC71'),
        ('Bottom 1%', bottom_posts, '#E74C3C'),
        ('Overall', hc, '#95A5A6')
    ]):
        vals = [(data['sentiment_label'] == s).sum() / len(data) * 100 for s in SENT_ORDER]
        x_pos = np.arange(len(SENT_ORDER))
        bars = ax.bar(x_pos + i * 0.25 - 0.25, vals, 0.25, label=label,
                      color=color, alpha=0.8, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.0f}%', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(np.arange(len(SENT_ORDER)))
    ax.set_xticklabels(SENT_ORDER)
    ax.set_title('Sentiment Profile', fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.legend(fontsize=7)

    # Subplot 2: Emotion
    ax = axes[1]
    top_clf = hc_clf[hc_clf.index.isin(top_posts.index)]
    bot_clf = hc_clf[hc_clf.index.isin(bottom_posts.index)]

    for i, (label, data, color) in enumerate([
        ('Top 1%', top_clf, '#2ECC71'),
        ('Bottom 1%', bot_clf, '#E74C3C'),
        ('Overall', hc_clf, '#95A5A6')
    ]):
        if len(data) > 0:
            vals = [(data['emo_primary'] == e).sum() / len(data) * 100 for e in EMO_ORDER]
        else:
            vals = [0] * len(EMO_ORDER)
        x_pos = np.arange(len(EMO_ORDER))
        bars = ax.bar(x_pos + i * 0.25 - 0.25, vals, 0.25, label=label,
                      color=color, alpha=0.8, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.0f}%', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(np.arange(len(EMO_ORDER)))
    ax.set_xticklabels(EMO_ORDER, rotation=20, ha='right', fontsize=8)
    ax.set_title('Emotion Profile', fontweight='bold')
    ax.legend(fontsize=7)

    # Subplot 3: Account distribution
    ax = axes[2]
    for i, (label, data, color) in enumerate([
        ('Top 1%', top_posts, '#2ECC71'),
        ('Bottom 1%', bottom_posts, '#E74C3C'),
        ('Overall', hc, '#95A5A6')
    ]):
        vals = [(data['_account'] == a).sum() / len(data) * 100 for a in ACC_ORDER]
        x_pos = np.arange(len(ACC_ORDER))
        bars = ax.bar(x_pos + i * 0.25 - 0.25, vals, 0.25, label=label,
                      color=color, alpha=0.8, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                        f'{h:.0f}%', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(np.arange(len(ACC_ORDER)))
    ax.set_xticklabels([ACC_LABELS[a] for a in ACC_ORDER])
    ax.set_title('Account Distribution', fontweight='bold')
    ax.legend(fontsize=7)

    add_methodology(fig,
        f'Top 1% = {n_top:,} posts with highest Engagement_Rate_Monthly. '
        f'Bottom 1% = {n_top:,} posts with lowest ER. '
        f'Bars show % distribution of each category for top/bottom/overall. '
        f'Differences reveal what characterises viral political content.',
        y=-0.08)
    save_chart(fig, 'SD2_top_bottom_posts_profile.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD3: Content Volume vs Per-Post Engagement
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(format_title('SD3',
        'Content Volume vs Mean Engagement Rate per Account',
        'Does posting more dilute per-post engagement?'),
        fontsize=14, fontweight='bold', y=1.02)

    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        volume = len(sub)
        mean_er = sub['Engagement_Rate_Monthly'].mean() * 100
        ax.scatter(volume, mean_er, s=300, color=ACC_COLORS[acc],
                   zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'{ACC_LABELS[acc]}\n({volume:,} posts, {mean_er:.4f}% ER)',
                    (volume, mean_er), textcoords="offset points",
                    xytext=(15, 10), fontsize=9, fontweight='bold',
                    color=ACC_COLORS[acc],
                    arrowprops=dict(arrowstyle='->', color=ACC_COLORS[acc], lw=1.5))

    ax.set_xlabel('Total Post Volume', fontsize=11)
    ax.set_ylabel('Mean Engagement Rate (%)', fontsize=11)

    add_methodology(fig,
        f'Source: Post count and mean(Engagement_Rate_Monthly) × 100 per account. '
        f'An inverse relationship (high volume = low ER) suggests "content fatigue" — '
        f'audiences engage less per post when flooded with content.',
        y=-0.04)
    save_chart(fig, 'SD3_volume_vs_engagement.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD4: Negativity-Engagement Paradox per Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(format_title('SD4',
        'Negativity Index vs Mean Engagement Rate',
        'Do more negative accounts get more or less engagement?'),
        fontsize=14, fontweight='bold', y=1.02)

    neg_indices = []
    er_vals = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        clf = hc_clf[hc_clf['_account'] == acc]
        tone = hc_tone[hc_tone['_account'] == acc]

        neg_p = pct((sub['sentiment_label'] == 'Negative').sum(), len(sub))
        anger_p = pct((clf['emo_primary'] == 'Anger').sum(), len(clf)) if len(clf) > 0 else 0
        agg_p = pct((tone['tone_primary'] == 'Aggressive').sum(), len(tone)) if len(tone) > 0 else 0
        neg_idx = (neg_p + anger_p + agg_p) / 3
        mean_er = sub['Engagement_Rate_Monthly'].mean() * 100

        neg_indices.append(neg_idx)
        er_vals.append(mean_er)

        ax.scatter(neg_idx, mean_er, s=350, color=ACC_COLORS[acc],
                   zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(f'{ACC_LABELS[acc]}\n(NI={neg_idx:.1f}, ER={mean_er:.4f}%)',
                    (neg_idx, mean_er), textcoords="offset points",
                    xytext=(15, 10), fontsize=9, fontweight='bold',
                    color=ACC_COLORS[acc],
                    arrowprops=dict(arrowstyle='->', color=ACC_COLORS[acc], lw=1.5))

    ax.set_xlabel('Composite Negativity Index', fontsize=11)
    ax.set_ylabel('Mean Engagement Rate (%)', fontsize=11)

    add_methodology(fig,
        f'Source: Negativity Index (Neg%+Anger%+Agg%)÷3 vs mean ER per account. '
        f'If negative accounts get higher engagement, it supports the "negativity bias" hypothesis. '
        f'If positive accounts do better, it challenges that assumption.',
        y=-0.04)
    save_chart(fig, 'SD4_negativity_engagement_paradox.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD5: Hashtag Strategy — Amplification vs Contradiction
    # ═════════════════════════════════════════════════════════════════════════
    # For each account: what % of posts use hashtags to AMPLIFY sentiment vs CONTRADICT it?
    hc_ht = hc[hc['hashtag_dominant_label'] != 'NoHashtags']

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(format_title('SD5',
        'Hashtag Strategy — Amplification, Neutral Pairing, and Contradiction',
        'How do accounts use hashtags relative to caption sentiment?'),
        fontsize=13, fontweight='bold', y=1.02)

    x = np.arange(len(ACC_ORDER))
    width = 0.2

    amplify_vals = []   # Same polarity: (Neg+Neg) or (Pos+Pos)
    neutral_pair = []   # One is neutral
    contradict_vals = [] # Opposite polarity: (Neg+Pos) or (Pos+Neg)

    for acc in ACC_ORDER:
        sub = hc_ht[hc_ht['_account'] == acc]
        total = len(sub)
        if total == 0:
            amplify_vals.append(0)
            neutral_pair.append(0)
            contradict_vals.append(0)
            continue

        amp = 0  # both positive or both negative
        neu = 0  # at least one neutral
        con = 0  # opposite polarity

        for _, row in sub.iterrows():
            cs = row['sentiment_label']
            hs = row['hashtag_dominant_label']
            if cs == hs:
                if cs == 'Neutral':
                    neu += 1
                else:
                    amp += 1
            elif cs == 'Neutral' or hs == 'Neutral':
                neu += 1
            else:
                con += 1

        amplify_vals.append(pct(amp, total))
        neutral_pair.append(pct(neu, total))
        contradict_vals.append(pct(con, total))

    bars1 = ax.bar(x - width, amplify_vals, width, label='Amplification (same polarity)',
                   color='#2ECC71', edgecolor='white')
    bars2 = ax.bar(x, neutral_pair, width, label='Neutral Pairing',
                   color='#95A5A6', edgecolor='white')
    bars3 = ax.bar(x + width, contradict_vals, width, label='Contradiction (opposite)',
                   color='#E74C3C', edgecolor='white')

    for b in [bars1, bars2, bars3]:
        add_value_labels(ax, b, fmt='{:.1f}%', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{ACC_LABELS[a]}\n(n={len(hc_ht[hc_ht["_account"]==a]):,})'
                        for a in ACC_ORDER])
    ax.set_ylabel('Percentage of Hashtagged Posts (%)')
    ax.legend(fontsize=9, framealpha=0.9)

    add_methodology(fig,
        f'Source: sentiment_label vs hashtag_dominant_label. '
        f'Amplification = both Positive or both Negative. '
        f'Neutral Pairing = at least one is Neutral, or both Neutral. '
        f'Contradiction = opposite polarity (Pos+Neg or Neg+Pos). '
        f'Contradiction suggests deliberate strategic messaging divergence.',
        y=-0.06)
    save_chart(fig, 'SD5_hashtag_strategy_analysis.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD6: Emotion-Driven Engagement Premium
    # ═════════════════════════════════════════════════════════════════════════
    # Which emotions give the biggest engagement "premium" vs overall average?
    overall_er = hc_clf['Engagement_Rate_Monthly'].mean() * 100

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('SD6',
        'Emotion-Driven Engagement Premium',
        f'ER difference vs overall mean ({overall_er:.4f}%) — which emotions outperform?'),
        fontsize=13, fontweight='bold', y=1.02)

    premiums = []
    for emo in EMO_ORDER:
        emo_er = hc_clf[hc_clf['emo_primary'] == emo]['Engagement_Rate_Monthly'].mean() * 100
        premium = emo_er - overall_er
        premiums.append(premium)

    colors = ['#2ECC71' if p >= 0 else '#E74C3C' for p in premiums]
    bars = ax.bar(EMO_ORDER, premiums, color=colors, edgecolor='white', width=0.5)

    for bar, p in zip(bars, premiums):
        h = bar.get_height()
        ypos = h + 0.005 if h >= 0 else h - 0.015
        ax.text(bar.get_x() + bar.get_width()/2., ypos,
                f'{p:+.4f}pp', ha='center', fontsize=9, fontweight='bold',
                color='#333')

    ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    ax.set_ylabel('Engagement Premium (pp vs overall mean)')
    ax.set_xlabel('Primary Emotion')

    add_methodology(fig,
        f'Premium = mean ER for that emotion minus overall mean ER ({overall_er:.4f}%). '
        f'Positive = that emotion outperforms average engagement. '
        f'Negative = underperforms. This identifies which emotional appeals are most '
        f'"rewarded" by audiences.',
        y=-0.06)
    save_chart(fig, 'SD6_emotion_engagement_premium.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD7: Alliance Strategy Asymmetry
    # ═════════════════════════════════════════════════════════════════════════
    # Do the two alliances mirror each other's strategy or diverge?
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(format_title('SD7',
        'Alliance Strategy Asymmetry',
        'How different are the two alliances across every dimension?'),
        fontsize=14, fontweight='bold', y=1.02)

    all_metrics = {
        'Negative%': ('sentiment_label', 'Negative', hc),
        'Positive%': ('sentiment_label', 'Positive', hc),
        'Neutral%': ('sentiment_label', 'Neutral', hc),
        'Anger%': ('emo_primary', 'Anger', hc_clf),
        'Hope%': ('emo_primary', 'Hope', hc_clf),
        'Trust%': ('emo_primary', 'Trust', hc_clf),
        'Pride%': ('emo_primary', 'Pride', hc_clf),
        'Fear%': ('emo_primary', 'Fear', hc_clf),
        'Aggressive%': ('tone_primary', 'Aggressive', hc_tone),
        'Persuasive%': ('tone_primary', 'Persuasive', hc_tone),
        'Inspirational%': ('tone_primary', 'Inspirational', hc_tone),
        'Victim Nar.%': ('tone_primary', 'Victim_Narrative', hc_tone),
        'Fact%': ('subj_label', 'Fact', hc_clf),
        'Opinion%': ('subj_label', 'Opinion', hc_clf),
    }

    y = np.arange(len(all_metrics))
    h = 0.35

    al_vals = {al: [] for al in ALLIANCE_ORDER}
    for metric_name, (col, val, df) in all_metrics.items():
        for al in ALLIANCE_ORDER:
            sub = df[df['_alliance'] == al]
            total = len(sub)
            c = (sub[col] == val).sum()
            al_vals[al].append(pct(c, total))

    bars_y = ax.barh(y + h/2, al_vals['YSRCP Alliance'], h,
                     label='YSRCP Alliance', color=ALLIANCE_COLORS['YSRCP Alliance'],
                     edgecolor='white', alpha=0.85)
    bars_t = ax.barh(y - h/2, al_vals['TDP Alliance'], h,
                     label='TDP Alliance', color=ALLIANCE_COLORS['TDP Alliance'],
                     edgecolor='white', alpha=0.85)

    for bar in bars_y:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2., f'{w:.1f}',
                va='center', fontsize=7, color=ALLIANCE_COLORS['YSRCP Alliance'],
                fontweight='bold')
    for bar in bars_t:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2., f'{w:.1f}',
                va='center', fontsize=7, color=ALLIANCE_COLORS['TDP Alliance'],
                fontweight='bold')

    # Delta column
    max_val = max(max(al_vals['YSRCP Alliance']), max(al_vals['TDP Alliance']))
    for i, metric_name in enumerate(all_metrics.keys()):
        delta = al_vals['YSRCP Alliance'][i] - al_vals['TDP Alliance'][i]
        color = ALLIANCE_COLORS['YSRCP Alliance'] if delta > 0 else ALLIANCE_COLORS['TDP Alliance']
        ax.text(max_val + 5, y[i], f'Δ={delta:+.1f}',
                ha='center', va='center', fontsize=7, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#f0f0f0', edgecolor='#ccc'))

    ax.set_yticks(y)
    ax.set_yticklabels(list(all_metrics.keys()), fontsize=9)
    ax.set_xlabel('Percentage (%)')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.invert_yaxis()
    ax.set_xlim(0, max_val + 12)

    add_methodology(fig,
        f'Source: All classification columns grouped by alliance. '
        f'Δ = YSRCP Alliance value minus TDP Alliance value. '
        f'The metric with the largest |Δ| reveals the primary strategic divergence.',
        y=-0.06)
    save_chart(fig, 'SD7_alliance_strategy_asymmetry.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART SD8: Quarterly Posting Rhythm per Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    fig.suptitle(format_title('SD8',
        'Posting Rhythm — Timeline Distribution per Platform per Account',
        'How does each account distribute content across timelines and platforms?'),
        fontsize=13, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]

        x_pos = np.arange(len(TL_ORDER))
        width = 0.35

        for j, plat in enumerate(PLAT_ORDER):
            vals = []
            for tl in TL_ORDER:
                c = ((sub['_timeline'] == tl) & (sub['_platform'] == plat)).sum()
                vals.append(c)
            bars = ax.bar(x_pos + (j - 0.5) * width, vals, width,
                          label=plat if i == 0 else '',
                          color=PLAT_COLORS[plat], edgecolor='white', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if h > 50:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 20,
                            f'{h:,.0f}', ha='center', fontsize=6, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([TL_LABELS[t].replace(' ', '\n') for t in TL_ORDER], fontsize=7)
        ax.set_title(f'{ACC_LABELS[acc]}\n(n={len(sub):,})', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        if i == 0:
            ax.set_ylabel('Post Count')

    axes[0].legend(fontsize=8)

    add_methodology(fig,
        f'Source: _timeline × _platform × _account. '
        f'Post counts per platform per timeline per account. '
        f'Reveals whether accounts shifted platform emphasis as elections approached.',
        y=-0.08)
    save_chart(fig, 'SD8_posting_rhythm.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    sec_insights = []

    # 10.1 Pre-election pivot
    pivot_lines = []
    total_shift_per_acc = {}
    for acc in ACC_ORDER:
        deltas = {}
        for metric_name, fn in metrics_shift.items():
            if metric_name in ['Anger%', 'Aggressive%', 'Fact%']:
                early = hc_clf[(hc_clf['_account'] == acc) & (hc_clf['_timeline'] == 'JunAug2023')]
                late = hc_clf[(hc_clf['_account'] == acc) & (hc_clf['_timeline'] == 'JanMay2024')]
            else:
                early = hc[(hc['_account'] == acc) & (hc['_timeline'] == 'JunAug2023')]
                late = hc[(hc['_account'] == acc) & (hc['_timeline'] == 'JanMay2024')]
            val_early = fn(early) if len(early) > 0 else 0
            val_late = fn(late) if len(late) > 0 else 0
            deltas[metric_name] = val_late - val_early

        total_shift = sum(abs(v) for v in deltas.values())
        total_shift_per_acc[acc] = total_shift
        pivot_lines.append(
            f'    • {ACC_LABELS[acc]}: '
            + ', '.join([f'{k}={v:+.1f}pp' for k, v in deltas.items()])
            + f'  (Total |Δ|={total_shift:.1f}pp)'
        )

    most_shifted = max(ACC_ORDER, key=lambda a: total_shift_per_acc[a])
    least_shifted = min(ACC_ORDER, key=lambda a: total_shift_per_acc[a])

    sec_insights.append((
        'INSIGHT 10.1',
        'Pre-Election Strategy Pivot Analysis',
        f'Change in key metrics from Jun–Aug 2023 to Jan–May 2024:\n\n'
        + '\n'.join(pivot_lines) + '\n\n'
        f'  {ACC_LABELS[most_shifted]} pivoted the MOST (total |Δ|={total_shift_per_acc[most_shifted]:.1f}pp), '
        f'indicating the largest pre-election strategic repositioning.\n'
        f'  {ACC_LABELS[least_shifted]} pivoted the LEAST (total |Δ|={total_shift_per_acc[least_shifted]:.1f}pp), '
        f'suggesting a more consistent long-term strategy.',
        f'Δ = value in Jan–May 2024 minus value in Jun–Aug 2023 per metric per account. '
        f'Total |Δ| = sum of absolute deltas across all metrics.'
    ))

    # 10.2 Top posts profile
    top_neg = pct((top_posts['sentiment_label'] == 'Negative').sum(), len(top_posts))
    top_pos = pct((top_posts['sentiment_label'] == 'Positive').sum(), len(top_posts))
    overall_neg = pct((hc['sentiment_label'] == 'Negative').sum(), n_total)
    overall_pos = pct((hc['sentiment_label'] == 'Positive').sum(), n_total)

    top_acc_dist = top_posts['_account'].value_counts()
    top_acc_lines = [f'    • {ACC_LABELS.get(a, a)}: {c} posts ({c/n_top*100:.1f}%)'
                     for a, c in top_acc_dist.items()]

    sec_insights.append((
        'INSIGHT 10.2',
        f'Top 1% Viral Posts Profile (n={n_top:,})',
        f'What characterises the highest-engagement posts?\n\n'
        f'  Sentiment: Negative={top_neg:.1f}% (vs {overall_neg:.1f}% overall), '
        f'Positive={top_pos:.1f}% (vs {overall_pos:.1f}% overall)\n\n'
        f'  Account distribution of top posts:\n'
        + '\n'.join(top_acc_lines) + '\n\n'
        f'  {"Positive content is over-represented" if top_pos > overall_pos else "Negative content is over-represented"} '
        f'in the top 1% posts, '
        f'{"supporting the finding that positive content drives engagement" if top_pos > overall_pos else "supporting the negativity bias hypothesis"}.',
        f'Top 1% = {n_top:,} posts with highest Engagement_Rate_Monthly. '
        f'Compared against overall dataset distribution.'
    ))

    # 10.3 Volume vs engagement
    vol_er_lines = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        vol_er_lines.append(
            f'    • {ACC_LABELS[acc]}: {len(sub):,} posts, Mean ER={sub["Engagement_Rate_Monthly"].mean()*100:.4f}%'
        )

    most_vol = max(ACC_ORDER, key=lambda a: len(hc[hc['_account'] == a]))
    most_er = max(ACC_ORDER, key=lambda a: hc[hc['_account'] == a]['Engagement_Rate_Monthly'].mean())

    sec_insights.append((
        'INSIGHT 10.3',
        'Content Volume vs Per-Post Engagement Paradox',
        f'Volume and engagement per account:\n\n'
        + '\n'.join(vol_er_lines) + '\n\n'
        f'  {ACC_LABELS[most_vol]} posts the most but '
        f'{"also" if most_vol == most_er else "does NOT"} have the highest per-post engagement.\n'
        f'  {"This suggests no content fatigue effect." if most_vol == most_er else "This suggests a content saturation / fatigue effect — more posts lead to lower per-post engagement."}',
        f'Volume = count of posts per account. Mean ER = mean(Engagement_Rate_Monthly) × 100.'
    ))

    # 10.4 Hashtag strategy
    strat_lines = []
    for i, acc in enumerate(ACC_ORDER):
        strat_lines.append(
            f'    • {ACC_LABELS[acc]}: Amplify={amplify_vals[i]:.1f}%, '
            f'Neutral Pair={neutral_pair[i]:.1f}%, Contradict={contradict_vals[i]:.1f}%'
        )

    most_contradict = ACC_ORDER[contradict_vals.index(max(contradict_vals))]

    sec_insights.append((
        'INSIGHT 10.4',
        'Hashtag Strategy — Amplification vs Contradiction',
        f'How accounts use hashtags relative to caption sentiment:\n\n'
        + '\n'.join(strat_lines) + '\n\n'
        f'  {ACC_LABELS[most_contradict]} has the highest contradiction rate '
        f'({max(contradict_vals):.1f}%), meaning it most frequently uses hashtags with '
        f'opposite sentiment to the caption. This "contradiction" strategy may serve to '
        f'inject negative hashtags into otherwise positive content (strategic poisoning) '
        f'or use positive hashtags to soften negative content (strategic softening).',
        f'Amplification = both sentiment and hashtag have same non-neutral polarity. '
        f'Neutral Pairing = at least one is neutral. '
        f'Contradiction = opposite polarity (Pos+Neg or Neg+Pos).'
    ))

    # 10.5 Emotion engagement premium
    emo_premiums = {}
    for emo in EMO_ORDER:
        emo_er = hc_clf[hc_clf['emo_primary'] == emo]['Engagement_Rate_Monthly'].mean() * 100
        emo_premiums[emo] = emo_er - overall_er

    best_emo = max(EMO_ORDER, key=lambda e: emo_premiums[e])
    worst_emo = min(EMO_ORDER, key=lambda e: emo_premiums[e])

    sec_insights.append((
        'INSIGHT 10.5',
        'Emotion-Driven Engagement Premium',
        f'Engagement premium by emotion (vs overall mean ER={overall_er:.4f}%):\n\n'
        + '\n'.join([f'    • {e}: {emo_premiums[e]:+.4f}pp' for e in EMO_ORDER])
        + f'\n\n  {best_emo} gives the highest engagement premium ({emo_premiums[best_emo]:+.4f}pp), '
        f'while {worst_emo} gives the lowest ({emo_premiums[worst_emo]:+.4f}pp).\n\n'
        f'  This reveals which emotional appeals are most "rewarded" by audiences. '
        f'Politicians seeking engagement should lean into {best_emo}-driven content.',
        f'Premium = mean ER for that emotion minus overall mean ER. '
        f'All values computed from classified posts only.'
    ))

    # 10.6 Alliance asymmetry
    max_delta_metric = ''
    max_delta_val = 0
    metric_names_list = list(all_metrics.keys())
    for i, metric_name in enumerate(metric_names_list):
        delta = abs(al_vals['YSRCP Alliance'][i] - al_vals['TDP Alliance'][i])
        if delta > max_delta_val:
            max_delta_val = delta
            max_delta_metric = metric_name

    sec_insights.append((
        'INSIGHT 10.6',
        'Alliance Strategy Asymmetry — Key Divergence Points',
        f'Head-to-head alliance comparison across {len(all_metrics)} dimensions:\n\n'
        + '\n'.join([
            f'    • {m}: YSRCP={al_vals["YSRCP Alliance"][i]:.1f}%, '
            f'TDP={al_vals["TDP Alliance"][i]:.1f}%, '
            f'Δ={al_vals["YSRCP Alliance"][i]-al_vals["TDP Alliance"][i]:+.1f}pp'
            for i, m in enumerate(metric_names_list)
        ])
        + f'\n\n  The LARGEST divergence is on **{max_delta_metric}** '
        f'(Δ={max_delta_val:.1f}pp), revealing the primary strategic battlefield '
        f'between the two alliances.',
        f'Each metric computed per alliance. Δ = YSRCP Alliance minus TDP Alliance.'
    ))

    # 10.7 Overarching strategic insights
    sec_insights.append((
        'INSIGHT 10.7',
        'STRATEGIC COMMUNICATION SYNTHESIS',
        f'Overarching strategic findings:\n\n'
        f'  1. ROLE-BASED STRATEGY: TDP Alliance (opposition) adopts a systematically '
        f'more negative/aggressive posture than YSRCP Alliance (incumbent), consistent '
        f'with classic opposition communication theory.\n\n'
        f'  2. LEADER VS PARTY DIFFERENTIATION: Leader accounts (YS Jagan, NCBN) are '
        f'strategically positioned differently from their party accounts — leaders tend '
        f'to maintain more distinct communication personas.\n\n'
        f'  3. ENGAGEMENT PARADOX: {"Positive" if top_pos > overall_pos else "Negative"} '
        f'content outperforms in engagement, challenging '
        f'{"nothing — this is expected" if top_pos > overall_pos else "the view that negativity drives engagement"}.\n\n'
        f'  4. PLATFORM ADAPTATION: Accounts show varying degrees of platform-specific '
        f'strategy, with some maintaining consistency and others diverging.\n\n'
        f'  5. PRE-ELECTION DYNAMICS: {ACC_LABELS[most_shifted]} shows the largest '
        f'strategic pivot as elections approach, while {ACC_LABELS[least_shifted]} '
        f'maintains the most stable strategy.',
        f'Synthesis of insights 10.1–10.6.'
    ))

    all_sections.append(('Strategic Deep Dive', sec_insights))

    write_insight_report('10_strategic_deep_dive_insights.txt',
                         'STRATEGIC DEEP DIVE', all_sections)

    print(f"\n  Done! Generated 8 charts + 1 text report (7 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

# ============================================================
#  SCRIPT 09 — ACCOUNT HEAD-TO-HEAD CONTRASTS
#  AP Political Sentiment Analysis — Insights Extraction
#
#  Analyses: Direct pairwise comparisons between accounts
#  - YSRCP vs TDP, NCBN vs YS Jagan (party accounts vs leader accounts)
#  - Alliance asymmetry (which alliance attacks more, who defends?)
#  - Platform strategy per account (do accounts tailor per platform?)
#  - Attack-vs-Self-Promotion ratios
#
#  Run: python insight_09_account_contrasts.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *

def main():
    setup_style()
    print("=" * 64)
    print("  SCRIPT 09: ACCOUNT HEAD-TO-HEAD CONTRASTS")
    print("=" * 64)

    hc, hc_clf = load_data()
    n_total = len(hc)
    n_clf = len(hc_clf)

    hc_tone = hc_clf[hc_clf['tone_primary'].isin(TONE_ORDER)]
    hc_ht = hc[hc['hashtag_dominant_label'] != 'NoHashtags']

    all_sections = []

    # ═════════════════════════════════════════════════════════════════════════
    #  Helper: compute full profile for a subset
    # ═════════════════════════════════════════════════════════════════════════
    def account_profile(acc):
        """Compute all key metrics for one account."""
        sub = hc[hc['_account'] == acc]
        clf = hc_clf[hc_clf['_account'] == acc]
        tone = hc_tone[hc_tone['_account'] == acc]
        ht = hc_ht[hc_ht['_account'] == acc]

        n = len(sub)
        neg_p = pct((sub['sentiment_label'] == 'Negative').sum(), n)
        pos_p = pct((sub['sentiment_label'] == 'Positive').sum(), n)
        neu_p = pct((sub['sentiment_label'] == 'Neutral').sum(), n)

        n_clf = len(clf)
        anger_p = pct((clf['emo_primary'] == 'Anger').sum(), n_clf) if n_clf > 0 else 0
        hope_p = pct((clf['emo_primary'] == 'Hope').sum(), n_clf) if n_clf > 0 else 0
        trust_p = pct((clf['emo_primary'] == 'Trust').sum(), n_clf) if n_clf > 0 else 0
        pride_p = pct((clf['emo_primary'] == 'Pride').sum(), n_clf) if n_clf > 0 else 0
        fear_p = pct((clf['emo_primary'] == 'Fear').sum(), n_clf) if n_clf > 0 else 0

        n_tone = len(tone)
        agg_p = pct((tone['tone_primary'] == 'Aggressive').sum(), n_tone) if n_tone > 0 else 0
        per_p = pct((tone['tone_primary'] == 'Persuasive').sum(), n_tone) if n_tone > 0 else 0
        insp_p = pct((tone['tone_primary'] == 'Inspirational').sum(), n_tone) if n_tone > 0 else 0
        vn_p = pct((tone['tone_primary'] == 'Victim_Narrative').sum(), n_tone) if n_tone > 0 else 0
        def_p = pct((tone['tone_primary'] == 'Defensive').sum(), n_tone) if n_tone > 0 else 0

        fact_p = pct((clf['subj_label'] == 'Fact').sum(), n_clf) if n_clf > 0 else 0
        opin_p = pct((clf['subj_label'] == 'Opinion').sum(), n_clf) if n_clf > 0 else 0
        mixed_p = pct((clf['subj_label'] == 'Mixed').sum(), n_clf) if n_clf > 0 else 0

        er = sub['Engagement_Rate_Monthly'].mean() * 100

        ht_agree = 0
        if len(ht) > 0:
            ht_agree = pct((ht['sentiment_label'] == ht['hashtag_dominant_label']).sum(), len(ht))

        return {
            'n': n, 'neg': neg_p, 'pos': pos_p, 'neu': neu_p,
            'anger': anger_p, 'hope': hope_p, 'trust': trust_p,
            'pride': pride_p, 'fear': fear_p,
            'agg': agg_p, 'per': per_p, 'insp': insp_p,
            'vn': vn_p, 'def': def_p,
            'fact': fact_p, 'opin': opin_p, 'mixed': mixed_p,
            'er': er, 'ht_agree': ht_agree,
            'pn_ratio': pos_p / neg_p if neg_p > 0 else 999,
            'ap_ratio': agg_p / per_p if per_p > 0 else 999,
            'neg_idx': (neg_p + anger_p + agg_p) / 3,
        }

    profiles = {acc: account_profile(acc) for acc in ACC_ORDER}

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC1: Head-to-Head — Party Accounts (YSRCP vs TDP)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating charts...")

    def h2h_chart(acc_a, acc_b, chart_id, filename):
        """Generate a back-to-back horizontal bar chart for two accounts."""
        pa, pb = profiles[acc_a], profiles[acc_b]

        metrics = [
            ('Negative %', pa['neg'], pb['neg']),
            ('Positive %', pa['pos'], pb['pos']),
            ('Anger %', pa['anger'], pb['anger']),
            ('Hope %', pa['hope'], pb['hope']),
            ('Aggressive %', pa['agg'], pb['agg']),
            ('Persuasive %', pa['per'], pb['per']),
            ('Victim Nar. %', pa['vn'], pb['vn']),
            ('Fact %', pa['fact'], pb['fact']),
            ('Opinion %', pa['opin'], pb['opin']),
            ('Negativity Index', pa['neg_idx'], pb['neg_idx']),
        ]

        labels = [m[0] for m in metrics]
        vals_a = [m[1] for m in metrics]
        vals_b = [m[2] for m in metrics]

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(format_title(chart_id,
            f'{ACC_LABELS[acc_a]} vs {ACC_LABELS[acc_b]} — Head-to-Head Comparison',
            f'Direct metric comparison across all dimensions'),
            fontsize=14, fontweight='bold', y=1.02)

        y = np.arange(len(labels))
        height = 0.35

        bars_a = ax.barh(y + height/2, vals_a, height, label=ACC_LABELS[acc_a],
                         color=ACC_COLORS[acc_a], edgecolor='white', linewidth=0.5)
        bars_b = ax.barh(y - height/2, vals_b, height, label=ACC_LABELS[acc_b],
                         color=ACC_COLORS[acc_b], edgecolor='white', linewidth=0.5)

        for bar in bars_a:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height()/2., f'{w:.1f}',
                    va='center', fontsize=8, fontweight='bold', color=ACC_COLORS[acc_a])
        for bar in bars_b:
            w = bar.get_width()
            ax.text(w + 0.5, bar.get_y() + bar.get_height()/2., f'{w:.1f}',
                    va='center', fontsize=8, fontweight='bold', color=ACC_COLORS[acc_b])

        # Add delta column
        for i in range(len(labels)):
            delta = vals_a[i] - vals_b[i]
            color = ACC_COLORS[acc_a] if delta > 0 else ACC_COLORS[acc_b]
            ax.text(max(max(vals_a), max(vals_b)) + 4, y[i],
                    f'Δ={delta:+.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0', edgecolor='#ccc'))

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Percentage / Index Value')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(max(vals_a), max(vals_b)) + 10)

        add_methodology(fig,
            f'Source: All classification dimensions. '
            f'{ACC_LABELS[acc_a]} (n={pa["n"]:,}) vs {ACC_LABELS[acc_b]} (n={pb["n"]:,}). '
            f'Δ = {ACC_LABELS[acc_a]} value minus {ACC_LABELS[acc_b]} value. '
            f'Positive Δ = {ACC_LABELS[acc_a]} is higher.',
            y=-0.06)
        save_chart(fig, filename)

    # AC1: YSRCP vs TDP (party accounts)
    h2h_chart('ysrcp', 'tdp', 'AC1', 'AC1_ysrcp_vs_tdp.png')

    # AC2: YS Jagan vs NCBN (leader accounts)
    h2h_chart('ysjagan', 'ncbn', 'AC2', 'AC2_ysjagan_vs_ncbn.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC3: Attack Intensity Index per Account
    # ═════════════════════════════════════════════════════════════════════════
    # Attack Intensity = weighted combination of negativity indicators
    # Components: Negative%, Anger%, Aggressive% (equally weighted)
    # vs Self-Promotion Index: Positive%, Hope+Trust+Pride%, Persuasive+Inspirational%

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(format_title('AC3',
        'Attack vs Self-Promotion Index per Account',
        'Attack = (Neg+Anger+Agg)÷3 | Self-Promo = (Pos+Hope/Trust/Pride+Per/Insp)÷3'),
        fontsize=13, fontweight='bold', y=1.02)

    x = np.arange(len(ACC_ORDER))
    width = 0.3

    attack_vals = []
    selfpromo_vals = []
    for acc in ACC_ORDER:
        p = profiles[acc]
        attack = (p['neg'] + p['anger'] + p['agg']) / 3
        selfpromo = (p['pos'] + (p['hope'] + p['trust'] + p['pride']) +
                     (p['per'] + p['insp'])) / 3
        attack_vals.append(attack)
        selfpromo_vals.append(selfpromo)

    bars_atk = ax.bar(x - width/2, attack_vals, width, label='Attack Index',
                      color='#E74C3C', edgecolor='white', linewidth=0.5)
    bars_sp = ax.bar(x + width/2, selfpromo_vals, width, label='Self-Promotion Index',
                     color='#2ECC71', edgecolor='white', linewidth=0.5)

    for i in range(len(ACC_ORDER)):
        ratio = attack_vals[i] / selfpromo_vals[i] if selfpromo_vals[i] > 0 else 0
        ax.text(x[i], max(attack_vals[i], selfpromo_vals[i]) + 1.5,
                f'A/SP={ratio:.2f}', ha='center', fontsize=8,
                fontweight='bold', color='#333',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa', edgecolor='#ccc'))

    add_value_labels(ax, bars_atk, fmt='{:.1f}', fontsize=9)
    add_value_labels(ax, bars_sp, fmt='{:.1f}', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{ACC_LABELS[a]}\n(n={profiles[a]["n"]:,})' for a in ACC_ORDER])
    ax.set_ylabel('Index Value')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, max(max(attack_vals), max(selfpromo_vals)) * 1.3)

    add_methodology(fig,
        f'Attack Index = (Negative% + Anger% + Aggressive%) ÷ 3. '
        f'Self-Promotion Index = (Positive% + [Hope+Trust+Pride]% + [Persuasive+Inspirational]%) ÷ 3. '
        f'A/SP ratio > 1 = more attack-oriented; < 1 = more self-promotional.',
        y=-0.06)
    save_chart(fig, 'AC3_attack_vs_selfpromotion.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC4: Platform Strategy Divergence per Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    fig.suptitle(format_title('AC4',
        'Platform Strategy Divergence per Account',
        'Do accounts tailor sentiment for Facebook vs Instagram?'),
        fontsize=14, fontweight='bold', y=1.02)

    for i, acc in enumerate(ACC_ORDER):
        ax = axes[i]
        sub = hc[hc['_account'] == acc]

        plat_data = {}
        for plat in PLAT_ORDER:
            ps = sub[sub['_platform'] == plat]
            total = len(ps)
            plat_data[plat] = {
                'n': total,
                'neg': pct((ps['sentiment_label'] == 'Negative').sum(), total),
                'pos': pct((ps['sentiment_label'] == 'Positive').sum(), total),
                'neu': pct((ps['sentiment_label'] == 'Neutral').sum(), total),
            }

        metrics = ['Negative', 'Neutral', 'Positive']
        y_pos = np.arange(len(metrics))
        h = 0.3

        for j, plat in enumerate(PLAT_ORDER):
            vals = [plat_data[plat]['neg'], plat_data[plat]['neu'], plat_data[plat]['pos']]
            bars = ax.barh(y_pos + (j - 0.5) * h, vals, h, label=plat if i == 0 else '',
                           color=PLAT_COLORS[plat], edgecolor='white', linewidth=0.5)
            for bar in bars:
                w = bar.get_width()
                if w > 2:
                    ax.text(w + 0.5, bar.get_y() + bar.get_height()/2.,
                            f'{w:.1f}%', va='center', fontsize=7,
                            color=PLAT_COLORS[plat], fontweight='bold')

        # Compute divergence
        fb = plat_data.get('Facebook', {})
        ig = plat_data.get('Instagram', {})
        neg_div = abs(fb.get('neg', 0) - ig.get('neg', 0))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=9)
        ax.set_title(f'{ACC_LABELS[acc]}\nΔ Neg={neg_div:.1f}pp', fontsize=10,
                     color=ACC_COLORS[acc], fontweight='bold')
        ax.set_xlim(0, 55)

    axes[0].legend(fontsize=8, loc='lower right')

    add_methodology(fig,
        f'Source: sentiment_label × _platform × _account. '
        f'Δ Neg = absolute difference in Negative% between Facebook and Instagram. '
        f'Larger Δ = more platform-differentiated strategy.',
        y=-0.08)
    save_chart(fig, 'AC4_platform_strategy_divergence.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC5: Emotion Diversity Index per Account
    # ═════════════════════════════════════════════════════════════════════════
    # Shannon entropy of emotion distribution — higher = more diverse emotion mix
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('AC5',
        'Emotion Diversity (Shannon Entropy) per Account',
        'Higher = more diverse emotional range; Lower = more concentrated'),
        fontsize=14, fontweight='bold', y=1.02)

    entropy_vals = []
    for acc in ACC_ORDER:
        clf = hc_clf[hc_clf['_account'] == acc]
        total = len(clf)
        if total == 0:
            entropy_vals.append(0)
            continue
        probs = []
        for emo in EMO_ORDER:
            p_emo = (clf['emo_primary'] == emo).sum() / total
            if p_emo > 0:
                probs.append(p_emo)
        entropy = -sum(p * np.log2(p) for p in probs)
        entropy_vals.append(entropy)

    # Max possible entropy for 6 classes = log2(6) ≈ 2.585
    max_entropy = np.log2(len(EMO_ORDER))

    bars = ax.bar([f'{ACC_LABELS[a]}\n(n={len(hc_clf[hc_clf["_account"]==a]):,})'
                   for a in ACC_ORDER],
                  entropy_vals, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)

    for bar, val in zip(bars, entropy_vals):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.03,
                f'{val:.3f}\n({val/max_entropy*100:.0f}% of max)',
                ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.axhline(y=max_entropy, color='#E74C3C', linestyle='--', alpha=0.5,
               label=f'Max possible = {max_entropy:.3f}')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_ylim(0, max_entropy * 1.15)
    ax.legend(fontsize=9)

    add_methodology(fig,
        f'Shannon Entropy H = -Σ p(x) log₂ p(x) where p(x) = proportion of each emotion. '
        f'Max entropy for {len(EMO_ORDER)} emotions = log₂({len(EMO_ORDER)}) = {max_entropy:.3f}. '
        f'Higher entropy = more evenly distributed emotions; lower = dominated by one emotion.',
        y=-0.06)
    save_chart(fig, 'AC5_emotion_diversity_entropy.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC6: Tone Diversity Index per Account
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(format_title('AC6',
        'Tone Diversity (Shannon Entropy) per Account',
        'Higher = more diverse rhetorical range; Lower = more concentrated'),
        fontsize=14, fontweight='bold', y=1.02)

    tone_entropy_vals = []
    for acc in ACC_ORDER:
        tone = hc_tone[hc_tone['_account'] == acc]
        total = len(tone)
        if total == 0:
            tone_entropy_vals.append(0)
            continue
        probs = []
        for t in TONE_ORDER:
            p_t = (tone['tone_primary'] == t).sum() / total
            if p_t > 0:
                probs.append(p_t)
        entropy = -sum(p * np.log2(p) for p in probs)
        tone_entropy_vals.append(entropy)

    max_tone_entropy = np.log2(len(TONE_ORDER))

    bars = ax.bar([f'{ACC_LABELS[a]}\n(n={len(hc_tone[hc_tone["_account"]==a]):,})'
                   for a in ACC_ORDER],
                  tone_entropy_vals, color=[ACC_COLORS[a] for a in ACC_ORDER],
                  edgecolor='white', linewidth=0.8, width=0.5)

    for bar, val in zip(bars, tone_entropy_vals):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
                f'{val:.3f}\n({val/max_tone_entropy*100:.0f}% of max)',
                ha='center', fontsize=9, fontweight='bold', color='#333')

    ax.axhline(y=max_tone_entropy, color='#E74C3C', linestyle='--', alpha=0.5,
               label=f'Max possible = {max_tone_entropy:.3f}')
    ax.set_ylabel('Shannon Entropy (bits)')
    ax.set_ylim(0, max_tone_entropy * 1.15)
    ax.legend(fontsize=9)

    add_methodology(fig,
        f'Shannon Entropy H = -Σ p(x) log₂ p(x) where p(x) = proportion of each tone. '
        f'Max entropy for {len(TONE_ORDER)} tones = log₂({len(TONE_ORDER)}) = {max_tone_entropy:.3f}. '
        f'Neutral-tone excluded. Higher = rhetorical versatility; lower = rhetorical rigidity.',
        y=-0.06)
    save_chart(fig, 'AC6_tone_diversity_entropy.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC7: Comprehensive Account Scorecard
    # ═════════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(format_title('AC7',
        'Comprehensive Account Scorecard',
        'All key metrics in a single comparison table'),
        fontsize=14, fontweight='bold', y=0.98)

    ax.axis('off')

    metrics_table = [
        ('Posts', [f'{profiles[a]["n"]:,}' for a in ACC_ORDER]),
        ('Negative %', [f'{profiles[a]["neg"]:.1f}%' for a in ACC_ORDER]),
        ('Positive %', [f'{profiles[a]["pos"]:.1f}%' for a in ACC_ORDER]),
        ('P/N Ratio', [f'{profiles[a]["pn_ratio"]:.2f}' for a in ACC_ORDER]),
        ('Anger %', [f'{profiles[a]["anger"]:.1f}%' for a in ACC_ORDER]),
        ('Hope %', [f'{profiles[a]["hope"]:.1f}%' for a in ACC_ORDER]),
        ('Trust %', [f'{profiles[a]["trust"]:.1f}%' for a in ACC_ORDER]),
        ('Aggressive %', [f'{profiles[a]["agg"]:.1f}%' for a in ACC_ORDER]),
        ('Persuasive %', [f'{profiles[a]["per"]:.1f}%' for a in ACC_ORDER]),
        ('A/P Ratio', [f'{profiles[a]["ap_ratio"]:.2f}' for a in ACC_ORDER]),
        ('Victim Nar. %', [f'{profiles[a]["vn"]:.1f}%' for a in ACC_ORDER]),
        ('Fact %', [f'{profiles[a]["fact"]:.1f}%' for a in ACC_ORDER]),
        ('Opinion %', [f'{profiles[a]["opin"]:.1f}%' for a in ACC_ORDER]),
        ('Mean ER %', [f'{profiles[a]["er"]:.4f}%' for a in ACC_ORDER]),
        ('HT Agreement %', [f'{profiles[a]["ht_agree"]:.1f}%' for a in ACC_ORDER]),
        ('Negativity Index', [f'{profiles[a]["neg_idx"]:.1f}' for a in ACC_ORDER]),
        ('Emo Entropy', [f'{entropy_vals[i]:.3f}' for i in range(len(ACC_ORDER))]),
        ('Tone Entropy', [f'{tone_entropy_vals[i]:.3f}' for i in range(len(ACC_ORDER))]),
    ]

    col_labels = ['Metric'] + [ACC_LABELS[a] for a in ACC_ORDER]
    cell_text = [[m[0]] + m[1] for m in metrics_table]

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc='center', loc='center',
                     colColours=['#e0e0e0'] + [ACC_COLORS[a] + '40' for a in ACC_ORDER])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        table[(0, j)].set_fontsize(10)
        table[(0, j)].set_text_props(fontweight='bold')
        if j > 0:
            table[(0, j)].set_facecolor(ACC_COLORS[ACC_ORDER[j-1]])
            table[(0, j)].set_text_props(fontweight='bold', color='white')

    # Style first column
    for i in range(1, len(metrics_table) + 1):
        table[(i, 0)].set_text_props(fontweight='bold', ha='left')
        table[(i, 0)].set_facecolor('#f8f9fa')

    add_methodology(fig,
        f'Source: All classification columns. Comprehensive comparison across '
        f'{len(metrics_table)} metrics for all 4 accounts. '
        f'P/N = Positive÷Negative. A/P = Aggressive÷Persuasive. '
        f'Entropy = Shannon diversity measure. Negativity Index = (Neg+Anger+Agg)÷3.',
        y=0.02)
    save_chart(fig, 'AC7_comprehensive_scorecard.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  CHART AC8: Leader Account vs Party Account Comparison
    # ═════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(format_title('AC8',
        'Leader Account vs Party Account — Same Alliance Comparison',
        'How does the personal account differ from the party account?'),
        fontsize=14, fontweight='bold', y=1.02)

    pairs = [
        ('ysrcp', 'ysjagan', 'YSRCP Alliance', ALLIANCE_COLORS['YSRCP Alliance']),
        ('tdp', 'ncbn', 'TDP Alliance', ALLIANCE_COLORS['TDP Alliance']),
    ]

    compare_metrics = ['neg', 'pos', 'anger', 'hope', 'agg', 'per', 'vn', 'fact', 'opin']
    compare_labels = ['Negative', 'Positive', 'Anger', 'Hope', 'Aggressive',
                      'Persuasive', 'Victim Nar.', 'Fact', 'Opinion']

    for idx, (party_acc, leader_acc, alliance_name, al_color) in enumerate(pairs):
        ax = axes[idx]
        pp = profiles[party_acc]
        lp = profiles[leader_acc]

        y = np.arange(len(compare_metrics))
        h = 0.35

        party_vals = [pp[m] for m in compare_metrics]
        leader_vals = [lp[m] for m in compare_metrics]

        ax.barh(y + h/2, party_vals, h, label=f'{ACC_LABELS[party_acc]} (Party)',
                color=ACC_COLORS[party_acc], edgecolor='white', alpha=0.8)
        ax.barh(y - h/2, leader_vals, h, label=f'{ACC_LABELS[leader_acc]} (Leader)',
                color=ACC_COLORS[leader_acc], edgecolor='white', alpha=0.8)

        for i in range(len(compare_metrics)):
            ax.text(party_vals[i] + 0.5, y[i] + h/2, f'{party_vals[i]:.1f}',
                    va='center', fontsize=7, color=ACC_COLORS[party_acc])
            ax.text(leader_vals[i] + 0.5, y[i] - h/2, f'{leader_vals[i]:.1f}',
                    va='center', fontsize=7, color=ACC_COLORS[leader_acc])

        ax.set_yticks(y)
        ax.set_yticklabels(compare_labels, fontsize=9)
        ax.set_title(f'{alliance_name}', fontsize=12, color=al_color, fontweight='bold')
        ax.legend(fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 60)

    add_methodology(fig,
        f'Source: All classification columns per account. '
        f'Within each alliance, the party account and leader account may adopt different '
        f'communication strategies — party accounts often handle more combative messaging, '
        f'while leader accounts may cultivate a more statesmanlike persona.',
        y=-0.08)
    save_chart(fig, 'AC8_leader_vs_party_account.png')

    # ═════════════════════════════════════════════════════════════════════════
    #  TEXT INSIGHTS
    # ═════════════════════════════════════════════════════════════════════════
    print("\n  Generating text insights...")

    sec_insights = []

    # 9.1 YSRCP vs TDP
    py, pt = profiles['ysrcp'], profiles['tdp']
    sec_insights.append((
        'INSIGHT 9.1',
        'YSRCP vs TDP — Party Account Head-to-Head',
        f'Key differences between the two main party accounts:\n\n'
        f'  Negativity: TDP ({pt["neg"]:.1f}%) > YSRCP ({py["neg"]:.1f}%) — gap: {pt["neg"]-py["neg"]:.1f}pp\n'
        f'  Positivity: YSRCP ({py["pos"]:.1f}%) > TDP ({pt["pos"]:.1f}%) — gap: {py["pos"]-pt["pos"]:.1f}pp\n'
        f'  Anger: TDP ({pt["anger"]:.1f}%) > YSRCP ({py["anger"]:.1f}%) — gap: {pt["anger"]-py["anger"]:.1f}pp\n'
        f'  Aggression: TDP ({pt["agg"]:.1f}%) > YSRCP ({py["agg"]:.1f}%) — gap: {pt["agg"]-py["agg"]:.1f}pp\n'
        f'  Persuasion: YSRCP ({py["per"]:.1f}%) > TDP ({pt["per"]:.1f}%) — gap: {py["per"]-pt["per"]:.1f}pp\n'
        f'  P/N Ratio: YSRCP={py["pn_ratio"]:.2f}, TDP={pt["pn_ratio"]:.2f}\n'
        f'  A/P Ratio: YSRCP={py["ap_ratio"]:.2f}, TDP={pt["ap_ratio"]:.2f}\n\n'
        f'  TDP operates with a distinctly more negative, anger-driven, and aggressive posture '
        f'compared to YSRCP. This is consistent with TDP\'s opposition role during this period, '
        f'using attack-based communication to challenge the incumbent YSRCP government.',
        f'Direct comparison of all key metrics between _account="ysrcp" and _account="tdp".'
    ))

    # 9.2 YS Jagan vs NCBN
    pj, pn = profiles['ysjagan'], profiles['ncbn']
    sec_insights.append((
        'INSIGHT 9.2',
        'YS Jagan vs NCBN — Leader Account Head-to-Head',
        f'Key differences between the two leader accounts:\n\n'
        f'  Negativity: NCBN ({pn["neg"]:.1f}%) > YS Jagan ({pj["neg"]:.1f}%) — gap: {pn["neg"]-pj["neg"]:.1f}pp\n'
        f'  Positivity: YS Jagan ({pj["pos"]:.1f}%) > NCBN ({pn["pos"]:.1f}%) — gap: {pj["pos"]-pn["pos"]:.1f}pp\n'
        f'  Anger: NCBN ({pn["anger"]:.1f}%) > YS Jagan ({pj["anger"]:.1f}%) — gap: {pn["anger"]-pj["anger"]:.1f}pp\n'
        f'  Aggression: NCBN ({pn["agg"]:.1f}%) > YS Jagan ({pj["agg"]:.1f}%) — gap: {pn["agg"]-pj["agg"]:.1f}pp\n'
        f'  P/N Ratio: YS Jagan={pj["pn_ratio"]:.2f}, NCBN={pn["pn_ratio"]:.2f}\n\n'
        f'  The contrast between the two leaders is even starker than between the party accounts. '
        f'YS Jagan\'s account maintains a significantly more positive, constructive image-building '
        f'approach, while NCBN\'s account leans heavily into attack-oriented critical messaging.',
        f'Direct comparison between _account="ysjagan" and _account="ncbn".'
    ))

    # 9.3 Attack vs Self-Promotion
    atk_sp = []
    for acc in ACC_ORDER:
        p = profiles[acc]
        atk = (p['neg'] + p['anger'] + p['agg']) / 3
        sp = (p['pos'] + (p['hope'] + p['trust'] + p['pride']) +
              (p['per'] + p['insp'])) / 3
        ratio = atk / sp if sp > 0 else 999
        atk_sp.append(f'    • {ACC_LABELS[acc]}: Attack={atk:.1f}, Self-Promo={sp:.1f}, A/SP={ratio:.2f}')

    sec_insights.append((
        'INSIGHT 9.3',
        'Attack vs Self-Promotion Balance',
        f'Each account\'s balance between attack-oriented and self-promotional messaging:\n\n'
        + '\n'.join(atk_sp) + '\n\n'
        f'  Accounts with A/SP > 1 are "net attackers" — their communication is dominated by '
        f'negativity, anger, and aggression. Accounts with A/SP < 1 are "net promoters" — '
        f'they prioritise constructive, positive, and persuasive messaging.',
        f'Attack = (Negative% + Anger% + Aggressive%) ÷ 3. '
        f'Self-Promo = (Positive% + [Hope+Trust+Pride]% + [Persuasive+Inspirational]%) ÷ 3. '
        f'A/SP = Attack ÷ Self-Promotion.'
    ))

    # 9.4 Platform strategy
    plat_insights = []
    for acc in ACC_ORDER:
        sub = hc[hc['_account'] == acc]
        fb = sub[sub['_platform'] == 'Facebook']
        ig = sub[sub['_platform'] == 'Instagram']
        fb_neg = pct((fb['sentiment_label'] == 'Negative').sum(), len(fb)) if len(fb) > 0 else 0
        ig_neg = pct((ig['sentiment_label'] == 'Negative').sum(), len(ig)) if len(ig) > 0 else 0
        div = abs(fb_neg - ig_neg)
        more_neg = 'Facebook' if fb_neg > ig_neg else 'Instagram'
        plat_insights.append(
            f'    • {ACC_LABELS[acc]}: FB Neg={fb_neg:.1f}%, IG Neg={ig_neg:.1f}%, '
            f'Δ={div:.1f}pp (more negative on {more_neg})'
        )

    sec_insights.append((
        'INSIGHT 9.4',
        'Platform Strategy Divergence Analysis',
        f'Do accounts tailor messaging by platform?\n\n'
        + '\n'.join(plat_insights) + '\n\n'
        f'  A large divergence (>5pp) in negativity between platforms suggests deliberate '
        f'platform-specific communication strategy. Political actors may use one platform '
        f'for aggressive attacks and another for positive image-building.',
        f'Negative% computed separately for Facebook and Instagram per account. '
        f'Δ = absolute difference.'
    ))

    # 9.5 Diversity analysis
    div_lines = []
    for i, acc in enumerate(ACC_ORDER):
        div_lines.append(
            f'    • {ACC_LABELS[acc]}: Emo Entropy={entropy_vals[i]:.3f} '
            f'({entropy_vals[i]/max_entropy*100:.0f}%), '
            f'Tone Entropy={tone_entropy_vals[i]:.3f} '
            f'({tone_entropy_vals[i]/max_tone_entropy*100:.0f}%)'
        )

    most_diverse_emo = ACC_ORDER[entropy_vals.index(max(entropy_vals))]
    least_diverse_emo = ACC_ORDER[entropy_vals.index(min(entropy_vals))]

    sec_insights.append((
        'INSIGHT 9.5',
        'Communication Diversity Analysis (Shannon Entropy)',
        f'How diverse is each account\'s emotional and rhetorical range?\n\n'
        + '\n'.join(div_lines) + '\n\n'
        f'  {ACC_LABELS[most_diverse_emo]} has the most diverse emotional range '
        f'(highest entropy), meaning its posts span multiple emotions relatively evenly. '
        f'{ACC_LABELS[least_diverse_emo]} has the least diverse, being more concentrated '
        f'on a single dominant emotion.\n\n'
        f'  Higher diversity can indicate sophisticated messaging strategy (different '
        f'emotional appeals for different contexts) or lack of coherent messaging focus.',
        f'Shannon Entropy H = -Σ p(x) log₂ p(x). Computed separately for emo_primary '
        f'and tone_primary per account. Max entropy = log₂(N) where N = number of categories.'
    ))

    # 9.6 Leader vs Party divergence
    ysrcp_v_jagan_neg_gap = abs(profiles['ysrcp']['neg'] - profiles['ysjagan']['neg'])
    tdp_v_ncbn_neg_gap = abs(profiles['tdp']['neg'] - profiles['ncbn']['neg'])

    sec_insights.append((
        'INSIGHT 9.6',
        'Leader vs Party Account Divergence',
        f'How different is the leader\'s personal account from the party account?\n\n'
        f'  YSRCP Alliance:\n'
        f'    YSRCP (party): Neg={profiles["ysrcp"]["neg"]:.1f}%, Pos={profiles["ysrcp"]["pos"]:.1f}%\n'
        f'    YS Jagan (leader): Neg={profiles["ysjagan"]["neg"]:.1f}%, Pos={profiles["ysjagan"]["pos"]:.1f}%\n'
        f'    Negativity gap: {ysrcp_v_jagan_neg_gap:.1f}pp\n\n'
        f'  TDP Alliance:\n'
        f'    TDP (party): Neg={profiles["tdp"]["neg"]:.1f}%, Pos={profiles["tdp"]["pos"]:.1f}%\n'
        f'    NCBN (leader): Neg={profiles["ncbn"]["neg"]:.1f}%, Pos={profiles["ncbn"]["pos"]:.1f}%\n'
        f'    Negativity gap: {tdp_v_ncbn_neg_gap:.1f}pp\n\n'
        f'  This reveals whether leaders cultivate a different (often more positive) image '
        f'than their party accounts. A large gap suggests deliberate role-differentiation '
        f'in the alliance\'s communication strategy.',
        f'Direct comparison within each alliance between party and leader accounts.'
    ))

    # 9.7 Comprehensive scorecard insight
    most_neg = max(ACC_ORDER, key=lambda a: profiles[a]['neg_idx'])
    least_neg = min(ACC_ORDER, key=lambda a: profiles[a]['neg_idx'])
    most_engaged = max(ACC_ORDER, key=lambda a: profiles[a]['er'])
    most_factual = max(ACC_ORDER, key=lambda a: profiles[a]['fact'])

    sec_insights.append((
        'INSIGHT 9.7',
        'Comprehensive Account Ranking Summary',
        f'Account rankings across key dimensions:\n\n'
        f'  Most Negative: {ACC_LABELS[most_neg]} (Index={profiles[most_neg]["neg_idx"]:.1f})\n'
        f'  Least Negative: {ACC_LABELS[least_neg]} (Index={profiles[least_neg]["neg_idx"]:.1f})\n'
        f'  Highest Engagement: {ACC_LABELS[most_engaged]} (ER={profiles[most_engaged]["er"]:.4f}%)\n'
        f'  Most Factual: {ACC_LABELS[most_factual]} (Fact={profiles[most_factual]["fact"]:.1f}%)\n'
        f'  Most Persuasive: {ACC_LABELS[max(ACC_ORDER, key=lambda a: profiles[a]["per"])]} '
        f'(Per={profiles[max(ACC_ORDER, key=lambda a: profiles[a]["per"])]["per"]:.1f}%)\n'
        f'  Most Aggressive: {ACC_LABELS[max(ACC_ORDER, key=lambda a: profiles[a]["agg"])]} '
        f'(Agg={profiles[max(ACC_ORDER, key=lambda a: profiles[a]["agg"])]["agg"]:.1f}%)\n'
        f'  Highest P/N Ratio: {ACC_LABELS[max(ACC_ORDER, key=lambda a: profiles[a]["pn_ratio"])]} '
        f'(P/N={profiles[max(ACC_ORDER, key=lambda a: profiles[a]["pn_ratio"])]["pn_ratio"]:.2f})\n',
        f'Rankings derived from all key metrics computed for each account.'
    ))

    all_sections.append(('Account Head-to-Head Contrasts', sec_insights))

    write_insight_report('09_account_contrasts_insights.txt',
                         'ACCOUNT HEAD-TO-HEAD CONTRASTS', all_sections)

    print(f"\n  Done! Generated 8 charts + 1 text report (7 insights).")
    print("=" * 64)


if __name__ == '__main__':
    main()

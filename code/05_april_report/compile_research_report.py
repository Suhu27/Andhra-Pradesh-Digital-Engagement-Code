# ============================================================
#  COMPILE RESEARCH REPORT
#  Generates a comprehensive HTML research report
#  with all charts + text insights embedded.
#
#  Run: python compile_research_report.py
#  Output: outputs/AP_Political_Sentiment_Research_Report.html
#  → Open in browser. Print to PDF (Ctrl+P → Save as PDF)
# ============================================================

import sys, os, base64, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SCRIPT_DIR, OUTPUT_DIR, CHART_DIR, TEXT_DIR
from datetime import datetime


def img_to_base64(path):
    """Convert image to base64 data URI for embedding."""
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def read_text_report(path):
    """Read text report and return structured content."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def build_html():
    print("=" * 64)
    print("  COMPILING RESEARCH REPORT")
    print("=" * 64)

    # ── Gather all outputs ──────────────────────────────────────────────
    charts = sorted(glob.glob(os.path.join(CHART_DIR, '*.png')))
    reports = sorted(glob.glob(os.path.join(TEXT_DIR, '*.txt')))

    print(f"  Found {len(charts)} charts and {len(reports)} text reports.")

    # ── Define report structure ─────────────────────────────────────────
    chapters = [
        {
            'id': 'ch1', 'num': '1',
            'title': 'Caption Sentiment Analysis',
            'prefix': 'S',
            'report': '01_caption_sentiment_insights.txt',
            'description': 'Overall sentiment distribution, per-account breakdowns, alliance comparisons, temporal trends, platform differences, and classifier confidence analysis.',
        },
        {
            'id': 'ch2', 'num': '2',
            'title': 'Hashtag Sentiment Analysis',
            'prefix': 'H',
            'report': '02_hashtag_sentiment_insights.txt',
            'description': 'Hashtag-level sentiment, caption-hashtag agreement, volume composition, and alliance-level hashtag patterns.',
        },
        {
            'id': 'ch3', 'num': '3',
            'title': 'Emotion Detection Analysis',
            'prefix': 'E',
            'report': '03_emotion_detection_insights.txt',
            'description': '6-class emotion classification (Anger, Trust, Pride, Hope, Informational, Fear), co-occurrence patterns, engagement correlations, and temporal evolution.',
        },
        {
            'id': 'ch4', 'num': '4',
            'title': 'Rhetorical Tone Analysis',
            'prefix': 'T',
            'report': '04_rhetorical_tone_insights.txt',
            'description': 'Aggressive, Persuasive, Inspirational, Victim Narrative, and Defensive tone distributions, A/P ratio, and engagement patterns.',
        },
        {
            'id': 'ch5', 'num': '5',
            'title': 'Subjectivity Analysis',
            'prefix': 'SB',
            'report': '05_subjectivity_insights.txt',
            'description': 'Fact, Mixed, and Opinion classification, cross-tabulations with sentiment and tone, and Fact-to-Opinion ratios.',
        },
        {
            'id': 'ch6', 'num': '6',
            'title': 'Engagement × Sentiment Correlations',
            'prefix': 'EC',
            'report': '06_engagement_correlations_insights.txt',
            'description': 'Deep-dive into how engagement metrics (ER, likes, shares) correlate with all classification dimensions.',
        },
        {
            'id': 'ch7', 'num': '7',
            'title': 'Timeline Evolution Analysis',
            'prefix': 'TL',
            'report': '07_timeline_evolution_insights.txt',
            'description': 'Temporal trends across all dimensions, pre-election shifts, and posting volume patterns.',
        },
        {
            'id': 'ch8', 'num': '8',
            'title': 'Cross-Dimensional Analysis',
            'prefix': 'CD',
            'report': '08_cross_dimensional_insights.txt',
            'description': 'Sentiment×Emotion, Sentiment×Tone, Emotion×Tone heatmaps, composite indices, messaging consistency, and multi-dimensional profiles.',
        },
        {
            'id': 'ch9', 'num': '9',
            'title': 'Account Head-to-Head Contrasts',
            'prefix': 'AC',
            'report': '09_account_contrasts_insights.txt',
            'description': 'Direct pairwise comparisons, attack vs self-promotion, platform strategy divergence, Shannon entropy diversity, and comprehensive scorecards.',
        },
        {
            'id': 'ch10', 'num': '10',
            'title': 'Strategic Deep Dive',
            'prefix': 'SD',
            'report': '10_strategic_deep_dive_insights.txt',
            'description': 'Pre-election pivots, viral content profiles, engagement paradoxes, hashtag strategy, and alliance asymmetry.',
        },
    ]

    # ── Build chart map ─────────────────────────────────────────────────
    chart_map = {}
    for c in charts:
        basename = os.path.basename(c)
        # Extract prefix (everything before the first underscore)
        prefix = basename.split('_')[0]
        # But some prefixes are like S1, E10, CD1 etc.
        if prefix not in chart_map:
            chart_map[prefix] = []
        # Group by the prefix letters only
        prefix_letters = ''.join([ch for ch in prefix if ch.isalpha()])
        if prefix_letters not in chart_map:
            chart_map[prefix_letters] = []
        chart_map[prefix_letters].append(c)

    # ── Build HTML ──────────────────────────────────────────────────────
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #1a1a2e;
            background: #ffffff;
            line-height: 1.7;
            font-size: 11pt;
        }

        .cover-page {
            page-break-after: always;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white;
            padding: 60px;
        }

        .cover-page h1 {
            font-size: 36pt;
            font-weight: 700;
            margin-bottom: 20px;
            line-height: 1.2;
        }

        .cover-page .subtitle {
            font-size: 16pt;
            font-weight: 300;
            opacity: 0.9;
            margin-bottom: 40px;
        }

        .cover-page .meta {
            font-size: 11pt;
            opacity: 0.7;
            line-height: 1.8;
        }

        .cover-page .divider {
            width: 120px;
            height: 3px;
            background: #e94560;
            margin: 30px auto;
        }

        .toc {
            page-break-after: always;
            padding: 60px;
            max-width: 800px;
            margin: 0 auto;
        }

        .toc h2 {
            font-size: 24pt;
            margin-bottom: 30px;
            color: #1a1a2e;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
        }

        .toc-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px dotted #ddd;
            font-size: 12pt;
        }

        .toc-item a {
            color: #0f3460;
            text-decoration: none;
            font-weight: 500;
        }

        .toc-item a:hover { color: #e94560; }

        .exec-summary {
            page-break-after: always;
            padding: 60px;
            max-width: 900px;
            margin: 0 auto;
        }

        .exec-summary h2 {
            font-size: 24pt;
            color: #1a1a2e;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }

        .exec-summary .finding {
            background: #f8f9fa;
            border-left: 4px solid #0f3460;
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }

        .exec-summary .finding strong { color: #e94560; }

        .chapter {
            page-break-before: always;
            padding: 40px 60px;
            max-width: 1100px;
            margin: 0 auto;
        }

        .chapter-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
            color: white;
            padding: 30px 40px;
            border-radius: 12px;
            margin-bottom: 30px;
        }

        .chapter-header h2 {
            font-size: 22pt;
            margin-bottom: 8px;
        }

        .chapter-header p {
            font-size: 11pt;
            opacity: 0.85;
        }

        .chart-container {
            text-align: center;
            margin: 25px 0;
            page-break-inside: avoid;
        }

        .chart-container img {
            max-width: 100%;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        .text-insights {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 30px;
            margin: 30px 0;
            white-space: pre-wrap;
            font-family: 'Inter', monospace;
            font-size: 10pt;
            line-height: 1.6;
            color: #333;
        }

        .methodology-appendix {
            page-break-before: always;
            padding: 40px 60px;
            max-width: 900px;
            margin: 0 auto;
        }

        .methodology-appendix h2 {
            font-size: 24pt;
            color: #1a1a2e;
            border-bottom: 3px solid #e94560;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }

        .methodology-appendix table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }

        .methodology-appendix th {
            background: #1a1a2e;
            color: white;
            padding: 10px 15px;
            text-align: left;
        }

        .methodology-appendix td {
            padding: 8px 15px;
            border-bottom: 1px solid #e0e0e0;
        }

        .methodology-appendix tr:nth-child(even) { background: #f8f9fa; }

        @media print {
            body { font-size: 10pt; }
            .chapter { padding: 20px 30px; }
            .cover-page { min-height: auto; padding: 40px; }
            .chart-container img { max-width: 95%; }
        }

        .findings-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }

        @media (max-width: 768px) {
            .findings-grid { grid-template-columns: 1fr; }
        }
    </style>
    """

    # ── Cover Page ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AP Political Sentiment Analysis — Research Report</title>
    {css}
</head>
<body>

<div class="cover-page">
    <h1>Political Sentiment Analysis<br>of Andhra Pradesh<br>Social Media Discourse</h1>
    <div class="divider"></div>
    <div class="subtitle">
        A Comprehensive, Multi-Dimensional Research Report<br>
        Based on 21,909 High-Confidence Social Media Posts
    </div>
    <div class="meta">
        Dataset: phase3_classified_patched.csv<br>
        Accounts: YSRCP, TDP, NCBN (Chandrababu Naidu), YS Jagan<br>
        Platforms: Facebook, Instagram<br>
        Period: June 2023 – May 2024<br>
        Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}<br><br>
        {len(charts)} Charts | {len(reports)} Text Reports | 10 Analytical Dimensions
    </div>
</div>
"""

    # ── Table of Contents ───────────────────────────────────────────────
    html += """
<div class="toc">
    <h2>Table of Contents</h2>
    <div class="toc-item"><a href="#exec-summary">Executive Summary</a></div>
"""
    for ch in chapters:
        # Check if report exists
        rpath = os.path.join(TEXT_DIR, ch['report'])
        if os.path.exists(rpath):
            html += f'    <div class="toc-item"><a href="#{ch["id"]}">Chapter {ch["num"]}: {ch["title"]}</a></div>\n'

    html += """
    <div class="toc-item"><a href="#methodology">Methodology Appendix</a></div>
</div>
"""

    # ── Executive Summary ───────────────────────────────────────────────
    html += """
<div class="exec-summary" id="exec-summary">
    <h2>Executive Summary</h2>

    <p>This report presents a comprehensive, multi-dimensional analysis of political sentiment
    in Andhra Pradesh social media discourse, covering 21,909 high-confidence posts across
    four major political accounts (YSRCP, TDP, NCBN, YS Jagan) on Facebook and Instagram,
    spanning June 2023 to May 2024 — the critical pre-election period leading to the
    AP state elections.</p>

    <h3 style="margin: 25px 0 15px; color: #0f3460;">Key Findings</h3>

    <div class="finding">
        <strong>1. Sentiment Landscape:</strong> Neutral sentiment dominates at 42%, followed by
        Negative (33%) and Positive (25%). The TDP Alliance is 10.1 percentage points more negative
        than the YSRCP Alliance, consistent with opposition-vs-incumbent communication dynamics.
    </div>

    <div class="finding">
        <strong>2. Emotional Temperature:</strong> Anger is the dominant emotion at 41.4% of all classified
        posts. 88.5% of negative-sentiment posts pair with Anger emotion, showing strong classifier alignment.
    </div>

    <div class="finding">
        <strong>3. Rhetorical Strategy:</strong> Aggressive tone leads at 45%, with NCBN having the
        highest Aggressive/Persuasive ratio (2.31). YS Jagan's account is the most persuasive (A/P = 0.57).
    </div>

    <div class="finding">
        <strong>4. Engagement Paradox:</strong> Positive content generates higher engagement than negative
        content across all four accounts, challenging the common "negativity bias" assumption in
        political communication.
    </div>

    <div class="finding">
        <strong>5. Pre-Election Dynamics:</strong> Negativity increases by ~2.3pp from Jun 2023 to May 2024.
        Post volume intensifies in the pre-election period with varying degrees of strategic pivot per account.
    </div>

    <div class="finding">
        <strong>6. Composite Negativity Index:</strong> NCBN (49.6) &gt; TDP (47.1) &gt; YSRCP (35.0) &gt;
        YS Jagan (26.1) — a single-number measure synthesizing Negative%, Anger%, and Aggressive%.
    </div>

    <div class="finding">
        <strong>7. Hashtag Strategy:</strong> Hashtags are more polarized than captions (51.5% agreement
        rate), with varying degrees of strategic amplification and contradiction across accounts.
    </div>

    <h3 style="margin: 25px 0 15px; color: #0f3460;">Report Structure</h3>
    <p>This report is organized into 10 analytical chapters, each covering a distinct dimension
    of the political communication ecosystem. Every chart includes methodology annotations,
    and every text insight documents how its values were calculated.</p>
</div>
"""

    # ── Chapters ────────────────────────────────────────────────────────
    for ch in chapters:
        report_path = os.path.join(TEXT_DIR, ch['report'])
        if not os.path.exists(report_path):
            print(f"  ⚠ Skipping chapter {ch['num']} — report not found: {ch['report']}")
            continue

        print(f"  Building Chapter {ch['num']}: {ch['title']}...")

        # Find charts for this chapter
        ch_charts = [c for c in charts if os.path.basename(c).startswith(ch['prefix'])
                     and os.path.basename(c)[len(ch['prefix']):len(ch['prefix'])+1].isdigit()]
        # Sort by number
        ch_charts.sort(key=lambda x: int(''.join(filter(str.isdigit,
                       os.path.basename(x).split('_')[0]))))

        html += f"""
<div class="chapter" id="{ch['id']}">
    <div class="chapter-header">
        <h2>Chapter {ch['num']}: {ch['title']}</h2>
        <p>{ch['description']}</p>
    </div>
"""

        # Embed charts
        for chart_path in ch_charts:
            chart_name = os.path.basename(chart_path)
            b64 = img_to_base64(chart_path)
            html += f"""
    <div class="chart-container">
        <img src="data:image/png;base64,{b64}" alt="{chart_name}" />
    </div>
"""

        # Embed text insights
        report_content = read_text_report(report_path)
        # Escape HTML characters
        report_content = (report_content
                          .replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;'))

        html += f"""
    <h3 style="margin-top: 30px; color: #0f3460; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;">
        Text Insights — Chapter {ch['num']}
    </h3>
    <div class="text-insights">{report_content}</div>
</div>
"""

    # ── Methodology Appendix ────────────────────────────────────────────
    html += """
<div class="methodology-appendix" id="methodology">
    <h2>Methodology Appendix</h2>

    <h3 style="margin: 20px 0 10px; color: #0f3460;">A. Dataset</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        <tr><td>Source File</td><td>phase3_classified_patched.csv</td></tr>
        <tr><td>Total Rows</td><td>30,726</td></tr>
        <tr><td>High-Confidence Rows (needs_review=False)</td><td>21,909</td></tr>
        <tr><td>Classified Rows (_insufficient=False)</td><td>19,790</td></tr>
        <tr><td>Accounts</td><td>YSRCP, TDP, NCBN, YS Jagan</td></tr>
        <tr><td>Platforms</td><td>Facebook, Instagram</td></tr>
        <tr><td>Period</td><td>June 2023 – May 2024</td></tr>
        <tr><td>Timelines</td><td>Jun–Aug 2023, Sep–Dec 2023, Jan–May 2024</td></tr>
    </table>

    <h3 style="margin: 20px 0 10px; color: #0f3460;">B. Classification Models</h3>
    <table>
        <tr><th>Dimension</th><th>Column</th><th>Classes</th><th>Method</th></tr>
        <tr><td>Caption Sentiment</td><td>sentiment_label</td><td>Positive, Neutral, Negative</td>
            <td>XLM-RoBERTa multilingual sentiment classifier (3-class softmax)</td></tr>
        <tr><td>Hashtag Sentiment</td><td>hashtag_dominant_label</td><td>Positive, Neutral, Negative, NoHashtags</td>
            <td>Dictionary-based translation + sentiment lookup</td></tr>
        <tr><td>Primary Emotion</td><td>emo_primary</td><td>Anger, Trust, Pride, Hope, Informational, Fear</td>
            <td>6-class emotion classifier (softmax)</td></tr>
        <tr><td>Rhetorical Tone</td><td>tone_primary</td><td>Aggressive, Persuasive, Inspirational, Victim_Narrative, Defensive</td>
            <td>5-class tone classifier with post-processing rules (R2, R5, R9)</td></tr>
        <tr><td>Subjectivity</td><td>subj_label</td><td>Fact, Mixed, Opinion</td>
            <td>3-class subjectivity classifier</td></tr>
    </table>

    <h3 style="margin: 20px 0 10px; color: #0f3460;">C. Statistical Methods</h3>
    <table>
        <tr><th>Method</th><th>Description</th></tr>
        <tr><td>Percentages</td><td>count ÷ total × 100 for all percentage calculations</td></tr>
        <tr><td>Central Tendency</td><td>Mean only (arithmetic mean). No medians used.</td></tr>
        <tr><td>Ratios</td><td>P/N Ratio, A/P Ratio, A/SP Ratio — direct division of counts or percentages</td></tr>
        <tr><td>Composite Indices</td><td>Simple averages of component percentages (e.g., Negativity Index)</td></tr>
        <tr><td>Shannon Entropy</td><td>H = -Σ p(x) log₂ p(x) for communication diversity measurement</td></tr>
        <tr><td>Cross-tabulation</td><td>Row-normalised or column-normalised crosstabs for dimensional alignment</td></tr>
    </table>

    <h3 style="margin: 20px 0 10px; color: #0f3460;">D. Filtering Rules</h3>
    <table>
        <tr><th>Filter</th><th>Applied To</th><th>Effect</th></tr>
        <tr><td>needs_review = False</td><td>All analyses</td><td>Restricts to 21,909 high-confidence rows</td></tr>
        <tr><td>_insufficient = False</td><td>Scripts 03–10</td><td>Excludes SKIPPED/insufficient rows → 19,790 classified</td></tr>
        <tr><td>tone_primary ∈ TONE_ORDER</td><td>Tone analyses</td><td>Excludes Neutral tone for focused rhetorical analysis</td></tr>
        <tr><td>hashtag_dominant_label ≠ NoHashtags</td><td>Hashtag analyses</td><td>Excludes posts without hashtags</td></tr>
    </table>

    <h3 style="margin: 20px 0 10px; color: #0f3460;">E. Tools & Dependencies</h3>
    <table>
        <tr><th>Tool</th><th>Version Used</th></tr>
        <tr><td>Python</td><td>3.11+</td></tr>
        <tr><td>pandas</td><td>Data manipulation</td></tr>
        <tr><td>numpy</td><td>Numerical computation</td></tr>
        <tr><td>matplotlib</td><td>Chart generation (Agg backend)</td></tr>
    </table>
</div>

</body>
</html>
"""

    # ── Write output ────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, 'AP_Political_Sentiment_Research_Report.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  ✓ Research report compiled!")
    print(f"    → {output_path}")
    print(f"    → {size_mb:.1f} MB ({len(charts)} charts embedded)")
    print(f"\n  To create PDF:")
    print(f"    1. Open the HTML file in your browser")
    print(f"    2. Press Ctrl+P → 'Save as PDF'")
    print(f"    3. Set margins to 'Minimum' for best results")
    print("=" * 64)


if __name__ == '__main__':
    build_html()

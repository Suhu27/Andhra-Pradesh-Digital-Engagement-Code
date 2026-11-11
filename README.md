# The Politics of Digital Engagement: analysis code

Code and selected figures for **“The Politics of Digital Engagement: Political Communication and Campaign Narratives in the 2024 Andhra Pradesh Assembly Election”** (submitted manuscript, November 2026).

The study analyzes **30,726** Facebook and Instagram posts from four official Andhra Pradesh political accounts. A 15-theme hashtag codebook assigns at least one theme to **27,388** posts, producing **54,142** theme-level records when multi-label posts are expanded. The published paper examines posting activity, themes, and engagement. **It does not measure emotion or sentiment.**

## What is here

| Location | Purpose |
| --- | --- |
| [`src/Hastags Thematic.xlsx`](src/Hastags%20Thematic.xlsx) | Codebook used for hashtag-to-theme mapping. The original filename is retained because the mapping script expects it. |
| [`src/hashtagrevamp.py`](src/hashtagrevamp.py) | Original final-stage thematic mapper. It reads original captions, case-folds hashtags, and applies the last matching codebook theme when a hashtag appears in more than one column. |
| [`src/Droppingnonpolitical.py`](src/Droppingnonpolitical.py) | Preserved post-label filtering/reinstatement script. It keeps a post labelled non-political if it has a theme, then drops remaining non-political posts. |
| [`src/engagmenetratemonthly.py`](src/engagmenetratemonthly.py) | Preserved monthly-follower engagement-rate calculation. Its filename is retained for traceability. |
| [`src/data_loader.py`](src/data_loader.py) and `src/phase*.py` | The final research-paper chart pipeline; see the figure map below. |
| [`reproductions/table3_panel_a.py`](reproductions/table3_panel_a.py) | A **new reconstruction** of Table 3 Panel A from the methods stated in the submitted manuscript. Its 15 theme counts and percentage estimates match the published table. |
| [`figures/`](figures/) | Fifteen source chart files selected for Figures 1–15. The manuscript crops or splits some multi-panel charts during page layout. |

The Instagram and Facebook collection samples are already published separately in [Instagram Post Engagement Metrics Scraper](https://github.com/Suhu27/Instagram-Post-Engagement-Metrics-Scraper) and [Facebook Posts Engagement Metric Scraper](https://github.com/Suhu27/Facebook-Posts-Engagement-Metric-Scraper). This repository focuses on the analysis behind the submitted manuscript.

## Figure map

| Manuscript figure | Source chart | Generating script |
| --- | --- | --- |
| 1 | `A3_Monthly_Post_Volume_Timeline.png` | `phase2.py` |
| 2 | `A1_Total_Posts_by_Platform.png` | `phase2.py` |
| 3 | `B1_Mean_Median_ER_per_Account.png` | `phase3.5.py` |
| 4 | `B3_Monthly_ER_Timeline.png` | `phase3.5.py` |
| 5 | `B7_ER_per_Period.png` | `phase3.5.py` |
| 6 | `C1_Overall_Theme_Frequency.png` | `phase4.py` |
| 7 | `D1_Mean_ER_per_Theme.png` | `phase5.py` |
| 8 | `D4_Top3_ER_Themes_per_Account.png` | `phase5.py` |
| 9 | `C11_Mean_ER_per_Theme_per_Account.png` | `phase4.5.py` |
| 10 | `C13_Theme_ER_per_Period.png` | `phase4.5.py` |
| 11 | `E5_Incumbent_vs_Challenger_ER.png` | `phase6.py` |
| 12 | `C3_REVISED_Thematic_Composition.png` | `phase4.5.py` |
| 13 | `C9_Per_Account_Theme_Breakdown.png` | `phase4.5.py` |
| 14 | `C6_Theme_Temporal_Shifts.png` | `phase4.py` |
| 15 | `C4_Cross_Party_Theme_Ownership.png` | `phase4.py` |

See [provenance notes](docs/PROVENANCE.md) for the evidence behind this selection and its limits.

## Reproduce the analysis

Python 3.10+ is recommended. Install dependencies with `pip install -r requirements.txt`. The source workbooks contain post-level data and are deliberately **not published**. To rerun the original chart scripts, place the 24 `*_HashtagClassified.xlsx` account/platform/period workbooks beside `src/data_loader.py`, then run from `src/`:

```bash
python phase2.py
python phase3.5.py
python phase4.py
python phase4.5.py
python phase5.py
python phase6.py
```

The scripts write chart families to `src/Analysis_Output/`. Their original run also creates companion Word reports and additional exploratory charts; the `figures/` folder contains only the selected source charts used in the manuscript. These original scripts predate this repository and expect the stated workbook names and columns.

To reproduce **Table 3 Panel A** from the same workbooks:

```bash
python reproductions/table3_panel_a.py --data-dir src --output results/table3_panel_a.csv
```

The thematic mapper and engagement calculator overwrite or rename workbooks in place. Work on copies if rerunning those preparation steps. The `Droppingnonpolitical.py` script is a preserved post-label filter, not the complete original political-label assignment engine.

## Scope and data availability

This repository contains the final manuscript-relevant source code, its theme codebook, selected figure files, and a clearly marked Table 3 Panel A reconstruction. It excludes earlier classifier experiments, the separate emotion/sentiment report pipeline, raw posts and comments, authentication/session files, and the 24 post-level workbooks. The original producer of some political labels and the exact historical code for Table 3 Panel B were not established in the preserved files; no script here is presented as that missing source.


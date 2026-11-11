# Submitted manuscript: figure-to-code map

The source image for each of the manuscript's 15 figures is in [`outputs/manuscript_figures/`](../outputs/manuscript_figures/). All original plots read the 24 workbooks in [`data/01_themed_posts/`](../data/01_themed_posts/) through `data_loader.py`. The paper layout crops or splits some source charts.

| Figure | Saved source chart | Script |
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

The manuscript also contains three tables. [`table3_panel_a_reconstruction.py`](../code/04_manuscript_analysis/table3_panel_a_reconstruction.py) reproduces Panel A of Table 3 from the reported method; the saved output is [`table3_panel_a_reconstruction.csv`](../outputs/manuscript_tables/table3_panel_a_reconstruction.csv). It is new reconstruction code, not the missing historical script. Panel B is outside the verified reproduction.

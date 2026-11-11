# Provenance and selection

## Manuscript checked

The reference is the 35-page submitted manuscript titled **“The Politics of Digital Engagement: Political Communication and Campaign Narratives in the 2024 Andhra Pradesh Assembly Election”**, local file `Manuscript.pdf` (modified 9 November 2026). It is distinct from the earlier 36-page “Emotion and Timing in Digital Campaigning” draft and the separate April sentiment report.

The manuscript itself says it does not measure emotion or sentiment. It reports 30,726 retained posts, 27,388 theme-coded posts, 54,142 expanded theme records, fifteen figures, and three tables. All fifteen selected chart files were found in the February research-paper `Analysis_Output` family. The original `data_loader.py`, `phase2.py`, `phase3.5.py`, `phase4.py`, `phase4.5.py`, `phase5.py`, and `phase6.py` files were copied without edits from `Desktop/clasfied and final datasets`. Their filenames and chart output targets link them to the figure map in the README. Some figures are split or cropped in the PDF layout; the complete source charts are preserved here.

`hashtagrevamp.py` and `Hastags Thematic.xlsx` came from that same source folder. The mapper uses `caption_raw` before `caption_en`, extracts hashtags without translation, and overwrites the workbook's theme columns. That matches the submitted manuscript's stated procedure. The source workbooks have 24 partitions and the exact thematic counts reported in the paper. `Droppingnonpolitical.py` records the reinstatement rule for non-political labelled posts with a theme, followed by removal of remaining non-political posts. `engagmenetratemonthly.py` contains the monthly follower denominators and calculation used by the source-workbook engagement fields.

## Table 3

The historical regression script for Table 3 was not found. The older `phase7.py` creates an exploratory OLS chart with label-encoded categories; it is **not** the submitted manuscript's fixed-effects Table 3 and was excluded. `reproductions/table3_panel_a.py` is new code derived from the manuscript's Table 3 note: one row per classified post, log monthly engagement rate, fifteen simultaneous theme indicators, account-by-platform and month fixed effects, video/reel indicator, hashtag count, and standard errors clustered by account/platform/month. A local run against the preserved 24 workbooks matched all 15 published theme counts and percentage estimates to one decimal place. This establishes a reproducible reconstruction of Panel A, not historical authorship of the missing code. Panel B is not claimed to be reproduced.

## Publication boundary

The 24 workbooks, raw captions/comments, authentication artifacts, and source datasets are omitted. The fifteen PNGs are aggregate charts already included in the manuscript. No credentials were found in the selected scripts. The collection code already published in two separate scraper repositories is not duplicated here.

## Later Colab file checked

The separately added `emotionclass.py` is a Colab export for the Phase 3 emotion, tone, and subjectivity pipeline. It includes the previously missing post-processing code that writes `phase3_classified_patched.csv`. Against the saved 30,726-row files, its four added flags matched the final file on every row; the other 51 columns were identical. This resolves that *separate pipeline's* patch-producer gap. It is excluded here because the submitted digital-engagement manuscript explicitly does not analyze emotion or sentiment, and its figures use the earlier thematic research workbooks.


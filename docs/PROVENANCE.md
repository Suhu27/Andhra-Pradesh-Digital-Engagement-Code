# Provenance and selection

## Research strands

The reference for the manuscript charts is the 35-page submitted paper, **“The Politics of Digital Engagement: Political Communication and Campaign Narratives in the 2024 Andhra Pradesh Assembly Election”** (local copy dated 9 November 2026). It is distinct from an earlier “Emotion and Timing in Digital Campaigning” draft and the separate April sentiment report. The submitted paper uses posting, thematic, and engagement analyses and states that it does not measure emotion or sentiment.

## Source-to-repository map

| Repository stage | Preserved source | Why selected |
| --- | --- | --- |
| `code/01_prepare_and_theme` | `deeptranslatest.py`, `emojicleaner.py` from the staged internship package; thematic/filter scripts and engagement calculator from the February analysis source | Documents translation, cleanup, theme mapping, post filtering, and monthly engagement calculation leading to the 24 thematic workbooks. |
| `data/01_themed_posts` | `Desktop/clasfied and final datasets` | The 24 `*_HashtagClassified.xlsx` workbooks are the exact input family used by the paper analysis scripts and the later sentiment work. The theme codebook comes from the same source. |
| `code/02_sentiment_and_hashtags` | `Desktop/Sentiment Analysis` | Preserved caption classifier, final dictionary-building/patch chain, and code that writes 24 `*_FullAnalysis.xlsx` files. |
| `data/02_caption_sentiment` | `Desktop/Sentiment Analysis/sentiment_output` | The 24 saved caption-sentiment workbooks consumed by the next stage. |
| `data/03_full_analysis` | `Desktop/Sentiment Analysis/full_analysis` and the final dictionary in `Desktop/Sentiment Analysis` | The 24 saved caption/hashtag full-analysis workbooks, cross-dataset summary, and final hashtag-sentiment dictionary. |
| `code/03_emotion_tone_subjectivity` | `Desktop/Sentiment Analysis/full_analysis` and the user-added `emotionclass.py` Colab export | Final multi-seed validator, production bulk classifier, verification/hybrid samples, and the exact post-hoc patch cell. |
| `data/04_final_classification` | `Desktop/Sentiment Analysis/Insightsonthesentiment/phase3_classified_patched.csv` | The final 30,726-row input used by all ten later analysis modules. Gzip is lossless; the underlying CSV is unchanged. |
| `code/04_manuscript_analysis` | `Desktop/clasfied and final datasets` | Original `data_loader.py` and phase scripts matching the 15 source figures. |
| `outputs/manuscript_figures` | February `Analysis_Output` chart family | The 15 source PNGs used in the submitted manuscript. Some are cropped or split in the PDF layout. |
| `code/05_april_report` and `outputs/april_report` | `Desktop/Sentiment Analysis/Insightsonthesentiment` | Shared config, ten analysis modules, compiler, all 90 final charts, ten text outputs, and the complete HTML report. |

## `emotionclass.py` check

The added Colab export includes code that writes `phase3_classified_patched.csv`. Its patch cell was extracted as `phase3_patch.py` without altering its four rules. The preserved `phase3_classified_all.csv` and patched CSV each have 30,726 rows. The patched CSV has four extra columns, with 76, 47, 640, and 52 true values respectively; every flag agrees with the extracted rules, and the other 51 columns agree with the original output. This establishes the producer for the later patch and supersedes older local notes that called it missing.

## Manuscript regression limit

The original historical script for Table 3 was not found. `table3_panel_a_reconstruction.py` was written from the manuscript methods: log monthly engagement rate, 15 simultaneous theme indicators, account-by-platform and month fixed effects, video/reel control, hashtag count, and clustered standard errors. A run against the 24 preserved workbooks matched all 15 Panel A counts and rounded estimates. It is labelled **reconstruction** throughout. The old `phase7.py` makes a different exploratory OLS chart and was excluded. Table 3 Panel B is not claimed to be reproduced.

## Selection boundary

The two existing scraper repositories cover collection, so this repository links to them rather than duplicating their source. The exact historical script that produced every political/non-political label could not be established; multiple classifier experiments and account-specific variants survive, but their relationship to the February final workbooks is incomplete. Those experimental variants are omitted rather than presented as confirmed producers. Also omitted: duplicate folder copies, earlier or superseded report charts, temporary model checkpoints, validation drafts, local browser/session credentials, and the earlier unsubmitted manuscript draft. The included workbooks and final CSV are full post-level files, including captions and available comment fields, following the user's public-release choice.

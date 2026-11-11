# Andhra Pradesh political communication research | internship portfolio

An organized record of the code, research datasets, and final outputs from my IIIT Hyderabad internship on the 2024 Andhra Pradesh Assembly election. The work covers **30,726 Facebook and Instagram posts** from four political accounts across June 2023–May 2024.

This repository connects the published collection examples to the later preparation, classification, analysis, and reporting stages. The submitted manuscript is **“The Politics of Digital Engagement: Political Communication and Campaign Narratives in the 2024 Andhra Pradesh Assembly Election.”**

## Start here: the research workflow

| Order | Work | Code | Data and final outputs |
| --- | --- | --- | --- |
| 0 | Collect posts and engagement | [Instagram scraper](https://github.com/Suhu27/Instagram-Post-Engagement-Metrics-Scraper) · [Facebook scraper](https://github.com/Suhu27/Facebook-Posts-Engagement-Metric-Scraper) | Collection examples stay in those two repositories. |
| 1 | Translate, clean, identify themes, calculate engagement | [`code/01_prepare_and_theme/`](code/01_prepare_and_theme/) | [24 theme-coded workbooks and theme codebook](data/01_themed_posts/) |
| 2 | Classify caption and hashtag sentiment | [`code/02_sentiment_and_hashtags/`](code/02_sentiment_and_hashtags/) | [24 caption sentiment workbooks](data/02_caption_sentiment/) · [24 full-analysis workbooks, final dictionary and summary](data/03_full_analysis/) |
| 3 | Validate and classify emotion, rhetoric, and subjectivity | [`code/03_emotion_tone_subjectivity/`](code/03_emotion_tone_subjectivity/) | [Final 30,726-row classified dataset](data/04_final_classification/phase3_classified_patched.csv.gz) |
| 4 | Analyze the submitted thematic paper | [`code/04_manuscript_analysis/`](code/04_manuscript_analysis/) | [15 manuscript figure source charts](outputs/manuscript_figures/) · [Table 3 Panel A reconstruction](outputs/manuscript_tables/table3_panel_a_reconstruction.csv) |
| 5 | Produce the later sentiment research report | [`code/05_april_report/`](code/05_april_report/) | [90 charts, 10 text analyses and self-contained HTML report](outputs/april_report/) |

The numbered folders describe the **research process**, not one script that automatically runs every stage. Several original scripts have local file paths or expect files beside them. [Reproduction notes](docs/RUNBOOK.md) explain their inputs and a safe way to rerun them.

## What the work demonstrates

- **Research data engineering:** collection across Facebook and Instagram, Telugu/English translation and text cleanup, 24 account × platform × period datasets, and monthly-follower engagement rates.
- **Thematic coding:** a 15-theme hashtag codebook. **27,388 posts** received at least one theme; exploding multiple themes yields **54,142 theme rows**.
- **Model-based classification:** caption and hashtag sentiment; multi-seed validation; guarded full-corpus emotion, rhetorical-tone, and subjectivity classification; a final post-processing step with four review flags.
- **Analysis and communication:** the exact 15 source charts selected for the submitted paper, a documented reconstruction of Table 3 Panel A, and a later 90-chart/10-section sentiment report.

## The submitted manuscript and the later report

The submitted manuscript examines posting volume, themes, and engagement. **It does not report sentiment or emotion measurements.** Its figure-to-script mapping is in [the manuscript figure guide](docs/MANUSCRIPT_FIGURES.md).

The April sentiment report is a **separate, later internship output**. It uses the same 30,726-post corpus after additional caption/hashtag sentiment and emotion/tone/subjectivity classification. Its charts and complete HTML report are under [`outputs/april_report/`](outputs/april_report/). The HTML file embeds the charts, so it can be opened on its own.

## Final dataset and Excel files

The three workbook folders contain 24 account × platform × period files at each saved stage. They include the full post-level fields, as authorized for public release. The final classification CSV is losslessly compressed to [`phase3_classified_patched.csv.gz`](data/04_final_classification/phase3_classified_patched.csv.gz); decompress it to recover the original CSV. It contains **30,726 rows and 55 columns**.

```python
import gzip, shutil
with gzip.open("data/04_final_classification/phase3_classified_patched.csv.gz", "rb") as source:
    with open("phase3_classified_patched.csv", "wb") as output:
        shutil.copyfileobj(source, output)
```

[`FILE_MANIFEST.csv`](FILE_MANIFEST.csv) lists every included file with its byte count and SHA-256 hash. The full data are published here so the final outputs can be inspected and the analysis rerun.

## About `emotionclass.py`

The added [Colab export](code/03_emotion_tone_subjectivity/emotionclass_colab_export.py) contains the production classification logic **and** the post-processing cell that writes `phase3_classified_patched.csv`. Its four added flags were checked against the saved final file: grief/solidarity (**76**), inverted defensive context (**47**), YS Jagan small-sample warning (**640**), and inspirational sarcasm risk (**52**). Every value matched, and the original 51 columns were unchanged. The [standalone patch script](code/03_emotion_tone_subjectivity/phase3_patch.py) extracts that final cell for an easier code sample.

## Scope and provenance

This repository includes the final datasets and outputs relevant to the two finished research strands, plus the saved code most directly connected to those outputs. It omits abandoned model versions, temporary checkpoints, raw authentication/session files, and duplicate copies. The exact historical producer of every political/non-political label was not established from the preserved files. The Table 3 Panel A script is a **documented reconstruction** from the submitted method and reproduces all 15 reported counts and rounded estimates; it is not represented as the missing original regression script. Panel B is not claimed to be reproduced. Details are in [provenance](docs/PROVENANCE.md).

## Run and environment

Python 3.10+ is recommended. Install the [listed dependencies](requirements.txt), then follow the [runbook](docs/RUNBOOK.md). GPU access is recommended for the large DeBERTa classification stage. The archived scripts retain their original filenames and locations where a path was part of their provenance; some require adjusting input/output paths for a new computer. Preparation scripts that overwrite workbooks should be run on copies.

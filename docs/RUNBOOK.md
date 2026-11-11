# Reproducing the saved outputs

## Environment

Use Python 3.10+ and `pip install -r requirements.txt`. The source scripts are preserved for traceability, so inspect their path configuration before running them. The DeBERTa bulk classifier was designed for a Kaggle GPU environment and takes much longer than the chart scripts.

## 1. Collection and themed workbooks

The [Instagram](https://github.com/Suhu27/Instagram-Post-Engagement-Metrics-Scraper) and [Facebook](https://github.com/Suhu27/Facebook-Posts-Engagement-Metric-Scraper) repos show the collection code. The 24 finished theme-coded input workbooks are in `data/01_themed_posts`. `code/01_prepare_and_theme` preserves the translation, cleanup, final theme mapping, non-political reinstatement/filter step, and monthly engagement calculation. Some of these scripts **overwrite workbooks in place**, so rerun them only on a copied set. The original complete producer of all political labels is not established.

## 2. Caption and hashtag sentiment

`sentiment_bulk.py` consumes the 24 themed workbooks and writes the 24 files under `data/02_caption_sentiment`. The Phase 1 scripts build and revise the hashtag-sentiment dictionary, ending with `Master_Hashtag_Dictionary_Final.csv`. `phase2_map_hashtag_sentiment.py` combines that dictionary with the sentiment workbooks and writes the 24 `*_FullAnalysis.xlsx` files plus `_PHASE2_SUMMARY.xlsx`. Its original configuration contains Windows absolute paths; point `DICT_FILE`, `DATASETS_FOLDER`, and `OUTPUT_FOLDER` at local copies when rerunning.

## 3. Emotion, tone, subjectivity, and final patch

`phase3_validate_large.py` records the multi-seed quality checks. `phase3_bulk_classify.py` reads 24 `*_FullAnalysis.xlsx` workbooks and writes `phase3_classified_all.csv`; its `INPUT_DIR` and `OUTPUT_DIR` default to Kaggle paths. The accompanying Colab export preserves extra audit and review cells. `phase3_patch.py`, extracted from that export, reads the full CSV and writes `phase3_classified_patched.csv` with four analytic flags. The published final file is compressed as `data/04_final_classification/phase3_classified_patched.csv.gz`. Decompress it with the Python snippet in the main README.

## 4. Submitted manuscript figures and Table 3 Panel A

The original paper scripts assume the 24 `*_HashtagClassified.xlsx` files are beside `data_loader.py`. For a clean rerun, create a temporary folder, **copy** the 24 workbooks and the seven files from `code/04_manuscript_analysis` into it, and run the phase scripts from that temporary folder in this order:

```text
phase2.py → phase3.5.py → phase4.py → phase4.5.py → phase5.py → phase6.py
```

They write chart families under `Analysis_Output`; the 15 images used by the manuscript are already saved in `outputs/manuscript_figures`. The archived scripts also create other charts and reports, but the selection here is limited to the manuscript figures.

For the independent Table 3 Panel A reconstruction:

```bash
python code/04_manuscript_analysis/table3_panel_a_reconstruction.py --data-dir data/01_themed_posts --output table3_panel_a_reproduced.csv
```

Compare the output with `outputs/manuscript_tables/table3_panel_a_reconstruction.csv`.

## 5. Later sentiment report

Decompress `phase3_classified_patched.csv.gz` and place the resulting CSV beside `code/05_april_report/config.py`, or edit `CSV_PATH` in that config. Run `insight_01_caption_sentiment.py` through `insight_10_strategic_deep_dive.py`, then `compile_research_report.py`. The scripts create `outputs/charts`, `outputs/text`, and the self-contained report in the analysis code folder. The published saved outputs are under `outputs/april_report`.

These steps explain how the preserved scripts connect to their files. Running the classifiers from scratch requires downloading their stated model checkpoints and may yield differences as external models or translation services change.

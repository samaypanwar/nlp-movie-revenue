# Movie overviews and box-office revenue

Do plot overviews improve prediction on later releases after accounting for release timing, genre, runtime, franchise status, and production companies?

## Rebuild the data

From the repository root, run `poetry install` with Python 3.14. Put the Kaggle [movie metadata CSV](https://www.kaggle.com/datasets/sibamsamanta07/movies-dataset-45k-films-with-budget-and-revenue) in `~/datasets/nlp-movie-revenue/raw/movies_metadata.csv`. Set `MOVIE_DATA_DIR` to use a different external directory. Download the CPI snapshot once, then rebuild any time with the same inputs.

```sh
data_dir="${MOVIE_DATA_DIR:-$HOME/datasets/nlp-movie-revenue}"
mkdir -p "$data_dir/raw"
curl -fL 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS&cosd=1913-01-01&coed=2020-12-31' \
  -o "$data_dir/raw/cpi_monthly.csv"
poetry run python notebooks/corpus.py
```

The cleaner prints row counts and SHA-256 hashes. The project movie snapshot hashes to `37ec322786b136a24604564e55160be75f8937546f11cb1a990076971fa895fd` and the CPI snapshot to `3247f33f63146035709eb0e19474ec2eec33c849c975b00228b8c5aa1fd84d5d`. Use `--raw`, `--cpi`, and `--output` for explicit paths. The derived CSV is written to `derived/movie_corpus.csv` under the external data directory. Raw and derived data stay out of Git.

The CSV has one English-original-language overview per movie ID. It drops empty and known placeholder text, collapses equivalent rows, excludes conflicting IDs, and marks unavailable financial values as missing. Genre, company IDs and country codes are JSON arrays. The CSV also includes release year and month, runtime, collection ID, and both recorded and CPI-adjusted amounts. Empty arrays indicate no entries listed. Collection IDs are for grouping and deriving a membership flag.

## Run the notebooks

Use the Poetry interpreter and set the notebook working directory to `notebooks/`. Register the kernel with `poetry run python -m ipykernel install --user --name nlp-movie-revenue` if needed.

1. **Corpus and EDA** (`01_corpus_and_eda.ipynb`) audits the raw data, calls the CLI's cleaning function, and explores release coverage, genre, and revenue. Figures are exported as 300 DPI PNGs under `outputs/`.
2. **Model comparison** (`02_model_comparison.ipynb`) interprets five NMF topics and measures how much plot adds to the full metadata controls across expanding chronological folds. It reports paired log MAE with intervals, control ablations, genre and company-support slices, and film examples. The error-bar figure is written to `outputs/paired_plot_errors.png`. Enable the optional encoder for the primary comparison.

For the optional contextual representation, install with `poetry run python -m pip install 'sentence-transformers==5.2.3'`, then set `MOVIE_SENTENCE_EMBEDDINGS=1` in the kernel environment. The notebook pins the all-MiniLM-L6-v2 checkpoint revision and uses its tokenizer. The first run downloads the checkpoint.

## Revenue and scope

The raw file contains 45,466 rows. The current cleaner retains 32,105 overviews and 6,348 films with positive recorded revenue and valid dates. The CSV has no per-film currency, geographic revenue scope, or receipt timing. Report log recorded revenue as the primary target and the CPI-adjusted model as a sensitivity under a comparable nominal USD assumption. The adjustment multiplies by annual 2016 CPI-U divided by annual release-year CPI-U from [FRED CPIAUCNS](https://fred.stlouisfed.org/series/CPIAUCNS). It approximates receipt timing with release year. Treat the experiment as retrospective and conditional on the dataset's revenue coverage.

## Remaining milestones

Milestones 1 and 2 are implemented in the model notebook. Across 2,660 evaluation films with listed companies, adding the frozen embedding increased log MAE by 0.025 (95% paired interval 0.005 to 0.045). In 2014–2016 the change was 0.006 (−0.032 to 0.044). Omitting company indicators increased pooled log MAE by 0.166 (0.127 to 0.204), the largest control ablation. The genre-specific time trends and budget sensitivity remain for milestone 3.

1. **Measure the incremental value of plot.** Use the existing expanding release windows. Compare one controls-only model with the same model plus plot on identical films. Include release year and month, runtime, multi-label genre, collection membership as a franchise proxy, and production-company indicators. Fit rare-company pooling and other transforms on earlier films only. Use the frozen embedding for the primary comparison and keep TF-IDF and five-topic NMF as references. Report paired changes in log MAE by release window and overall, with 95% paired bootstrap intervals that resample films within release year for pooled estimates. These intervals describe uncertainty conditional on the observed years and fitted models. Report CPI adjustment on the same films as a sensitivity. The 2014–2016 window has already been inspected.
2. **Explain where plot helps and which controls matter.** Report paired error changes by major genre and release window, with film counts and intervals. Use 30 unique films as an initial display floor and interpret the interval width alongside the count. Genre groups overlap. Pool production companies in the model and compare films with a company well represented in training against films without one when there is enough support. The coverage audit found that the largest genre × company cell in 2014–2016 has 18 films, or 17 after also splitting by collection status. Use those intersections to document coverage rather than rank effects. Inspect a few films where the plot model helped or hurt. For the unsupervised result, document the five NMF topics with top words, representative films, sizes, and a brief stability check.
3. **Check the controls on matched samples.** Drop one control block at a time from the full baseline to measure its incremental predictive value with the other controls present. Keep the release-time terms in every model used to estimate plot's contribution. Check whether that contribution persists when allowing genre-specific time trends. Run budget as a separate matched-cohort sensitivity, comparing the full controls with and without plot on the same budget-observed films. Report these as predictive comparisons for films with recorded revenue.
4. **Complete the IMRaD report.** Use a compact table for the main paired comparison, a PNG with uncertainty bars for supported slices, and a topic summary. Explain the time split, the definition of each control, the limits of currency and revenue coverage, and which conclusions survive the sensitivity checks. Write the abstract after the results and link the submission revision.

## Report

`latex/main.tex` is the 12-point IMRaD entry point. Include the abstract, all team names, Andrew IDs and campuses on the first page, and a code appendix link. The limit is 10 pages excluding the code appendix and at most nine figures and tables combined. Deadline in the project brief is December 9, 2026 at 11:59 p.m. ET on Gradescope.

From `latex/`, build an Overleaf ZIP with `zip -r ../overleaf-movie-revenue.zip main.tex sections references.bib figures`.

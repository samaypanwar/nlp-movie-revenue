# Movie overviews and box-office revenue

Do plot overviews improve prediction on later movie releases beyond genre and basic release metadata?

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
2. **Model comparison** (`02_model_comparison.ipynb`) uses five descriptive NMF topics and runs fixed ridge models with plot-only and metadata-plus-plot features. It uses expanding chronological folds and reports 2014–2016 separately. Compare paired log MAE overall, by genre and by genre and evaluation window. Production, franchise and budget each have their own matched comparison.

For the optional contextual representation, install with `poetry run python -m pip install 'sentence-transformers==5.2.3'`, then set `MOVIE_SENTENCE_EMBEDDINGS=1` in the kernel environment. The notebook pins the all-MiniLM-L6-v2 checkpoint revision and uses its tokenizer. The first run downloads the checkpoint.

## Revenue and scope

The raw file contains 45,466 rows. The current cleaner retains 32,105 overviews and 6,348 films with positive recorded revenue and valid dates. The CSV has no per-film currency, geographic revenue scope, or receipt timing. Report log recorded revenue as the primary target and the CPI-adjusted model as a sensitivity under a comparable nominal USD assumption. The adjustment multiplies by annual 2016 CPI-U divided by annual release-year CPI-U from [FRED CPIAUCNS](https://fred.stlouisfed.org/series/CPIAUCNS). It approximates receipt timing with release year. Treat the experiment as retrospective and conditional on the dataset's revenue coverage.

## Report

`latex/main.tex` is the 12-point IMRaD entry point. Include the abstract, all team names, Andrew IDs and campuses on the first page, and a code appendix link. The limit is 10 pages excluding the code appendix and at most nine figures and tables combined. Deadline in the project brief is December 9, 2026 at 11:59 p.m. ET on Gradescope.

From `latex/`, build an Overleaf ZIP with `zip -r ../overleaf-movie-revenue.zip main.tex sections references.bib`.

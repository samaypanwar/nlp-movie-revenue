# Movie overviews and box-office revenue

Can plot overviews improve out-of-time prediction of recorded movie revenue beyond basic release metadata?

## Run the data work

1. Obtain `movies_metadata.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/sibamsamanta07/movies-dataset-45k-films-with-budget-and-revenue). Place it in `~/datasets/nlp-movie-revenue/raw/`. The course snapshot has SHA-256 `37ec322786b136a24604564e55160be75f8937546f11cb1a990076971fa895fd`. Set `MOVIE_DATA_DIR` to use a different external directory.
2. Run `poetry install` and use the Poetry interpreter for the notebooks. From this directory, `poetry run python -m ipykernel install --user --name nlp-movie-revenue` registers that interpreter with Jupyter.
3. Open the notebooks in order with their working directory set to `notebooks/`. The cleaning notebook writes `derived/movie_corpus.csv` under the external data directory.

The source audit checks schema, coercion failures, zero sentinels, duplicates and language coverage. Cleaning normalizes whitespace, removes known placeholders and resolves duplicate IDs. EDA checks cohort sizes, release-year and label coverage, text lengths, repeated overviews and revenue quantiles. Save local figures and metrics in `outputs/`.

## Initial audit

The source snapshot has 45,466 rows and 24 columns. Cleaning leaves 32,105 overviews, including 6,348 with positive revenue and a parsed release date. The source has 38,052 zero-revenue entries. Positive budget is available for 75.5% of the supervised cohort. There are 29 rows with repeated normalized overviews in the text corpus and zero in the supervised cohort. The 2014–2016 releases offer 728 labeled films as a candidate holdout; the last labeled year, 2017, has only 86.

## Modeling plan

- Define one document as a single movie overview. Start with English-original-language films. Describe differences in labeled versus unlabeled movies. Verify currency, release-date semantics, and revenue coverage before interpreting error in dollars.
- Fit descriptive TF-IDF NMF on the text corpus. Compare topic counts (start with 5, 8, and 12) using top-word overlap across seeds, topic redundancy, representative documents and reconstruction error. Report the post-filter vocabulary and interpret topics with example films. Treat HW2's eight topics as a starting point.
- Predict log recorded positive revenue with ridge regression. Compare a training-median predictor, release metadata (year, month, runtime), metadata plus TF-IDF, and metadata plus NMF weights on identical rows and splits. Consider budget separately after checking coverage and provenance. Omit votes and popularity because their measurement timing is unknown.
- Reserve recent years for final evaluation after inspecting label counts by year. Tune on expanding chronological folds. Fit imputation, scaling, vocabulary and NMF inside each fold. Inspect exact and near-duplicate overviews before freezing splits and remove training duplicates that match later evaluation rows. Report log MAE as primary, plus log RMSE and paired fold differences. Inspect where text helps and fails.
- Treat the result as retrospective prediction for films with recorded revenue. A pre-release forecasting claim requires evidence that the text and metadata were available then. Consider CSSNMF only after the main comparison. Revenue alone cannot establish profitability.

## Report

`latex/main.tex` is the entry point. Use one 12-point IMRaD report with an abstract, team names, Andrew IDs and campuses on the first page, and a code appendix link. Keep the report within 10 pages excluding the code appendix, with at most nine figures and tables combined. Include data provenance, document and vocabulary counts, topic-count selection, leakage controls, comparative results, limitations and follow-up work. Deadline in the project brief is December 9, 2026 at 11:59 p.m. ET on Gradescope.

From `latex/`, build the Overleaf source ZIP with `zip -r ../overleaf-movie-revenue.zip main.tex sections references.bib`. In Overleaf, choose New Project → Upload Project and select the ZIP.

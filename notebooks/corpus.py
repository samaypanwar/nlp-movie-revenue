"""Read and prepare the movie overview corpus."""

import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(
    os.environ.get("MOVIE_DATA_DIR", "~/datasets/nlp-movie-revenue")
).expanduser()
RAW_PATH = DATA_DIR / "raw" / "movies_metadata.csv"
CLEAN_PATH = DATA_DIR / "derived" / "movie_corpus.csv"

PLACEHOLDERS = {
    "no overview found",
    "no overview available",
    "no overview",
    "no description available",
}


def prepare_movies(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Normalize source fields and return the English corpus and filter counts.

    Args:
        raw (pd.DataFrame): Source metadata with the original CSV columns.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Deduplicated corpus and sequential row counts.
    """
    movies = raw.assign(
        movie_id=pd.to_numeric(raw["id"], errors="coerce"),
        overview=raw["overview"].astype("string").str.split().str.join(" "),
        release_date=pd.to_datetime(raw["release_date"], format="%Y-%m-%d", errors="coerce"),
        revenue=pd.to_numeric(raw["revenue"], errors="coerce"),
        budget=pd.to_numeric(raw["budget"], errors="coerce"),
        runtime=pd.to_numeric(raw["runtime"], errors="coerce"),
    )
    counts = {"Source rows": len(movies)}
    movies = movies.loc[movies["movie_id"].gt(0) & movies["movie_id"].mod(1).eq(0)]
    counts["Valid ID"] = len(movies)
    movies = movies.loc[movies["original_language"].eq("en")]
    counts["English original language"] = len(movies)
    movies = movies.loc[movies["overview"].fillna("").ne("")]
    counts["Nonblank overview"] = len(movies)
    normalized = movies["overview"].str.casefold().str.strip(" .!?")
    movies = movies.loc[~normalized.isin(PLACEHOLDERS)]
    counts["After placeholder filter"] = len(movies)

    # Exclude conflicting records for the same ID from the analysis cohort.
    fields = ["movie_id", "overview", "release_date", "revenue", "budget", "runtime"]
    movies = movies.drop_duplicates(subset=fields)
    movies = movies.loc[~movies["movie_id"].duplicated(keep=False)].copy()
    movies["movie_id"] = movies["movie_id"].astype("int64")
    movies["revenue"] = movies["revenue"].where(movies["revenue"].gt(0))
    movies["budget"] = movies["budget"].where(movies["budget"].gt(0))
    movies["runtime"] = movies["runtime"].where(movies["runtime"].gt(0))
    counts["After ID deduplication"] = len(movies)
    counts["Positive revenue and known date"] = int(
        (movies["revenue"].notna() & movies["release_date"].notna()).sum()
    )
    return movies, pd.Series(counts, name="rows")

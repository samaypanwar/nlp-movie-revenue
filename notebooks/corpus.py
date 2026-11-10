"""Build the analysis CSV from the movie metadata and CPI snapshots."""

import argparse
from ast import literal_eval
from hashlib import sha256
import json
import os
from pathlib import Path

import pandas as pd


DATA_DIR = Path(
    os.environ.get("MOVIE_DATA_DIR", "~/datasets/nlp-movie-revenue")
).expanduser()
RAW_PATH = DATA_DIR / "raw" / "movies_metadata.csv"
CPI_PATH = DATA_DIR / "raw" / "cpi_monthly.csv"
CLEAN_PATH = DATA_DIR / "derived" / "movie_corpus.csv"

PLACEHOLDERS = {
    "no overview",
    "no overview available",
    "no overview found",
    "no description available",
}

CATEGORY_KEYS = {
    "genres": "name",
    "production_companies": "id",
    "production_countries": "iso_3166_1",
}

MOVIE_FIELDS = [
    "movie_id",
    "original_language",
    "overview",
    "release_date",
    "revenue",
    "budget",
    "runtime",
    *CATEGORY_KEYS,
    "collection_id",
]


def build_corpus(
    raw_path: Path = RAW_PATH,
    cpi_path: Path = CPI_PATH,
    output_path: Path = CLEAN_PATH,
) -> pd.Series:
    """Clean the movie snapshot, add release-year CPI adjustments, and write the CSV.

    Args:
        raw_path (Path): Movie metadata CSV.
        cpi_path (Path): Monthly FRED CPIAUCNS CSV.
        output_path (Path): Derived corpus CSV.

    Returns:
        pd.Series: Counts at each cleaning step.
    """
    movies = pd.read_csv(raw_path, low_memory=False)
    counts = {"Source rows": len(movies)}

    # Malformed rows in the source have nonnumeric movie IDs.
    ids = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.loc[ids.gt(0) & ids.mod(1).eq(0)].copy()
    movies["movie_id"] = ids.loc[movies.index].astype(int)
    counts["Valid ID"] = len(movies)

    movies = movies.loc[movies["original_language"].eq("en")].copy()
    counts["English original language"] = len(movies)

    movies["overview"] = movies["overview"].astype("string").str.split().str.join(" ")
    movies = movies.loc[movies["overview"].fillna("").ne("")].copy()
    counts["Nonblank overview"] = len(movies)

    normalized = movies["overview"].str.casefold().str.strip(" .!?")
    movies = movies.loc[~normalized.isin(PLACEHOLDERS)].copy()
    counts["After placeholder filter"] = len(movies)

    movies["release_date"] = pd.to_datetime(
        movies["release_date"], format="%Y-%m-%d", errors="coerce"
    )
    for field in ("revenue", "budget", "runtime"):
        values = pd.to_numeric(movies[field], errors="coerce").astype(float)
        movies[field] = values.where(values.gt(0))

    for field, key in CATEGORY_KEYS.items():
        movies[field] = movies[field].map(
            lambda value: json.dumps(sorted({str(item[key]) for item in literal_eval(value)})),
            na_action="ignore",
        )
    movies["collection_id"] = movies["belongs_to_collection"].map(
        lambda value: str(literal_eval(value)["id"]),
        na_action="ignore",
    )

    movies = movies[MOVIE_FIELDS].drop_duplicates()
    movies = movies.drop_duplicates("movie_id", keep=False).copy()
    counts["After ID deduplication"] = len(movies)

    dates = movies["release_date"].dt
    movies["release_year"] = dates.year.astype("Int64")
    movies["release_month"] = dates.month.astype("Int64")
    counts["Positive revenue and known date"] = int(
        (movies["revenue"].notna() & movies["release_date"].notna()).sum()
    )

    monthly = pd.read_csv(cpi_path, parse_dates=["observation_date"])
    annual = monthly.groupby(monthly["observation_date"].dt.year)["CPIAUCNS"].agg(
        ["mean", "count"]
    )
    cpi = annual.loc[annual["count"].eq(12), "mean"]
    adjustment = cpi.loc[2016] / movies["release_year"].map(cpi)
    movies["revenue_cpi_2016"] = movies["revenue"] * adjustment
    movies["budget_cpi_2016"] = movies["budget"] * adjustment
    counts["Positive revenue with CPI coverage"] = movies["revenue_cpi_2016"].count()

    if output_path.resolve() in {raw_path.resolve(), cpi_path.resolve()}:
        raise ValueError("Choose an output path separate from the raw input files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(output_path, index=False)

    return pd.Series(counts, name="rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--cpi", type=Path, default=CPI_PATH)
    parser.add_argument("--output", type=Path, default=CLEAN_PATH)
    args = parser.parse_args()

    print(build_corpus(args.raw, args.cpi, args.output).to_string())
    for path in (args.raw, args.cpi, args.output):
        print(f"{sha256(path.read_bytes()).hexdigest()}  {path}")

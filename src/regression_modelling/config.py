"""Paths only for the regression_modelling task. Constants live in constants.py."""
from crime_blockgroup_mapping.config import PROJECT_ROOT, RAW_DIR, INTERIM_DIR

# all task SQL templates live under data_wrangling/sql/{build,pull,explore}
SQL_DIR = PROJECT_ROOT / "src" / "regression_modelling" / "data_wrangling" / "sql"


def source_parquet(name: str):
    return INTERIM_DIR / "sources" / f"{name}.parquet"


def bias_parquet(name: str = "protected_attributes"):
    """Cache for the bias-testing-only protected-attribute table (data/interim/bias/).

    Kept in a separate tier from `sources/` so protected attributes can never be swept
    into the predictor matrix by construction (ADR 0004). Keyed by `geoid`.
    """
    return INTERIM_DIR / "bias" / f"{name}.parquet"


def transit_raw_dir(city: str):
    """Immutable downloaded GTFS feed zips for a city (data/raw/transit/{city}/).

    Tolerates folder names that differ from the city key: a space instead of an
    underscore (``san francisco`` for ``san_francisco``) or a short form
    (``kansas`` for ``kansas_city``). Falls back to the canonical path if none exist.
    """
    base = RAW_DIR / "transit"
    candidates = [city, city.replace("_", " "), city.split("_")[0]]
    for name in candidates:
        d = base / name
        if d.is_dir():
            return d
    return base / city


def transit_feed_zip(city: str, feed_id: str):
    """Resolve the GTFS zip for one feed by its Mobility Database file stem.

    Matches ``{feed_id}-*.zip`` (e.g. ``mdb-389-*.zip`` or ``tld-764-*.zip``) under the
    city's raw dir; if several snapshots are present the most recent (lexicographically
    last, since MDB names embed a sortable timestamp) is returned. Raises
    ``FileNotFoundError`` when absent.
    """
    d = transit_raw_dir(city)
    matches = sorted(d.glob(f"{feed_id}-*.zip"))
    if not matches:
        raise FileNotFoundError(
            f"No GTFS zip for feed {feed_id} under {d} "
            f"(expected {feed_id}-*.zip). Download it into that folder."
        )
    return matches[-1]


def transit_stops_parquet(city: str):
    """Per-stop feature intermediate for a city (data/interim/transit/stops/{city}.parquet)."""
    return INTERIM_DIR / "transit" / "stops" / f"{city}.parquet"


def transit_facilities_parquet(category: str):
    """Cached risky-facility point layer (data/interim/transit/facilities/{category}.parquet).

    Centroids of firmographics parcels for a co-location category (convenience/liquor/atm),
    pulled once from BigQuery then reused offline for the stop co-location join.
    """
    return INTERIM_DIR / "transit" / "facilities" / f"{category}.parquet"

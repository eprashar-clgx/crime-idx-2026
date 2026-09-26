"""Paths only for the regression_modelling task. Constants live in constants.py."""
from crime_blockgroup_mapping.config import PROJECT_ROOT, RAW_DIR, INTERIM_DIR

# all task SQL templates live under data_wrangling/sql/{build,pull,explore}
SQL_DIR = PROJECT_ROOT / "src" / "regression_modelling" / "data_wrangling" / "sql"


def source_parquet(name: str):
    return INTERIM_DIR / "sources" / f"{name}.parquet"


def agency_parquet(name: str):
    """Cache for agency-level (police-jurisdiction) artifacts (data/interim/agency/).

    Task-1 agency comparison intermediates: the block-group -> agency crosswalk
    (`bg_akey_crosswalk`) and our BG predictor set rolled up to agency level
    (`agency_predictors`). Keyed by the UCR agency key `akey`.
    """
    return INTERIM_DIR / "agency" / f"{name}.parquet"


def features_parquet(name: str = "bg_predictors"):
    """Engineered BG predictor table (data/interim/features/{name}.parquet). Keyed by geoid."""
    return INTERIM_DIR / "features" / f"{name}.parquet"


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


def tiger_roads_zip(state_fips: str, county_fips: str, year: int = 2024):
    """Immutable downloaded TIGER/Line county roads zip (data/raw/roadway/tiger/).

    One zip per (state, county) FIPS pair, shared across every city whose BGs fall in
    that county (e.g. multiple cities can pull the same county). Filename mirrors the
    Census source name so it is identifiable outside this repo:
    ``tl_{year}_{state_fips}{county_fips}_roads.zip``.
    """
    return RAW_DIR / "roadway" / "tiger" / f"tl_{year}_{state_fips}{county_fips}_roads.zip"


def roadway_tiger_county_parquet(state_fips: str, county_fips: str, year: int = 2024):
    """Cached, filtered (MTFCC-relevant) county roads layer, reprojected to EPSG:5070.

    data/interim/roadway/tiger/{state_fips}{county_fips}_{year}.parquet — one per county,
    reused across cities without re-parsing the raw shapefile each build.
    """
    return INTERIM_DIR / "roadway" / "tiger" / f"{state_fips}{county_fips}_{year}.parquet"


def roadway_tiger_junctions_parquet(state_fips: str, county_fips: str, year: int = 2024):
    """Cached street-junction points for one county, derived from TIGER EDGES topology.

    data/interim/roadway/tiger_junctions/{state_fips}{county_fips}_{year}.parquet — nodes
    where >= 3 public-street edges meet (EPSG:5070), with their degree.
    """
    return INTERIM_DIR / "roadway" / "tiger_junctions" / f"{state_fips}{county_fips}_{year}.parquet"


def tiger_edges_zip(state_fips: str, county_fips: str, year: int = 2024):
    """Immutable downloaded TIGER/Line county EDGES zip (data/raw/roadway/tiger/).

    The topological layer (every edge carries from/to node ids TNIDF/TNIDT) — used for
    junction degree, which the ROADS layer (one record per named road) cannot provide.
    """
    return RAW_DIR / "roadway" / "tiger" / f"tl_{year}_{state_fips}{county_fips}_edges.zip"


def hpms_county_parquet(state_usps: str, county_fips: str, year: int = 2024):
    """Immutable HPMS pull for one county (data/raw/roadway/hpms/{ST}_{county}_{year}.parquet).

    Server-side-filtered query of FHWA's HPMS_FULL_{ST}_{year} FeatureServer (functional
    class 1-4, mainline sections only), stored as GeoParquet in EPSG:4326 as received.
    """
    return RAW_DIR / "roadway" / "hpms" / f"{state_usps}_{county_fips}_{year}.parquet"

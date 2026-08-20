"""Registries + column lists for the regression_modelling task."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSource:
    name: str                       # 'vacancy' → cache file name
    backend: str                    # 'bq', 'gcs', or 'file' (materialized out-of-band)
    location: str                   # BQ: pull-sql name | GCS: path template | file: builder note
    key_col: str = "geoid"          # join key in the pulled data
    feature_cols: tuple = ()        # predictor columns to keep


@dataclass(frozen=True)
class TransitFeed:
    """One GTFS feed for a city (a city may have several, e.g. SF = Muni + BART).

    Feeds are identified by their stable Mobility Database file stem (`feed_id`, e.g.
    ``mdb-389`` or ``tld-764``); the downloaded zip is resolved by glob
    (``{feed_id}-*.zip``) so re-downloading a fresher snapshot of the same feed does not
    break paths.
    """
    agency: str                     # short label, e.g. 'cta', 'bart'
    feed_id: str                    # Mobility Database file stem; zip = {feed_id}-*.zip


# Per-city GTFS feeds, keyed by the same keys as CITIES (crime_blockgroup_mapping).
# One representative mid-2025 snapshot per feed (record feed_version on download).
# SF unions Muni + BART; shared stations are deduped by proximity in feeds.load_city_stops.
TRANSIT_FEEDS = {
    "chicago":       (TransitFeed("cta",   "mdb-389"),),
    "houston":       (TransitFeed("metro", "mdb-2060"),),
    "atlanta":       (TransitFeed("marta", "mdb-368"),),
    "san_francisco": (TransitFeed("muni",  "mdb-2886"), TransitFeed("bart", "mdb-53")),
    "pittsburgh":    (TransitFeed("prt",   "mdb-409"),),
    # Secondary (property-only crime) cities — transit supply features still valid.
    "jacksonville":  (TransitFeed("jta",   "tld-764"),),
    "kansas_city":   (TransitFeed("kcata", "mdb-187"),),
    "sacramento":    (TransitFeed("sacrt", "mdb-2137"),),
}

# Representative service date to pin trips/day and service span (a typical Wednesday,
# feed active ~June 2025). Overnight window lives in transit.feeds.
TRANSIT_REPRESENTATIVE_DATE = "2025-06-04"


FEATURE_SOURCES = {
    "vacancy": FeatureSource(
        name="vacancy",
        backend="bq",
        location="vacancy",         # → sql/pull/vacancy.sql
        key_col="geoid",
        feature_cols=("vacant_pct",),
    ),
    "liens": FeatureSource(
        name="liens",
        backend="bq",
        location="liens",          # → sql/pull/liens.sql
        key_col="geoid",
        feature_cols=("clip_liens_pct",),
    ),
    "foreclosures": FeatureSource(
        name="foreclosures",
        backend="bq",
        location="foreclosures",     # → sql/pull/foreclosures.sql
        key_col="geoid",
        feature_cols=("clip_foreclosure_pct",),
    ),
    "convenience_stores": FeatureSource(
        name="convenience_stores",
        backend="bq",
        location="convenience_stores",  # → sql/pull/convenience_stores.sql
        key_col="geoid",
        feature_cols=("unq_convenience_stores_clips",),
    ),
    "gas_stations": FeatureSource(
        name="gas_stations",
        backend="bq",
        location="gas_stations",      # → sql/pull/gas_stations.sql
        key_col="geoid",
        feature_cols=("unq_gas_stations_clips",),
    ),
    "liquor_stores": FeatureSource(
        name="liquor_stores",
        backend="bq",
        location="liquor_stores",     # → sql/pull/liquor_stores.sql
        key_col="geoid",
        feature_cols=("unq_liquor_stores_clips",),
    ),
    # Transit is materialized out-of-band by transit.build.build_all_transit (backend="file").
    # Covers the 5 POC cities only; null elsewhere on the national spine. feature_cols are the
    # candidate BG predictors (docs/transit_eda_plan.md §5); promote to PREDICTOR_COLS after EDA.
    "transit": FeatureSource(
        name="transit",
        backend="file",
        location="build via regression_modelling.data_wrangling.transit.build_all_transit",
        key_col="geoid",
        feature_cols=(
            "transit_stop_count",
            "transit_stop_density",
            "transit_nearest_stop_m",
            "transit_service_intensity",
            "transit_overnight_stop_count",
            "transit_overnight_stop_share",
            "transit_risky_stop_count",
            "transit_risky_stop_share",
            "transit_risky_allnight_count",
            "transit_route_mode_diversity",
        ),
    ),
}

# Store universes for the generic block-group builder (sql/build/stores.sql).
# {store} = table/column stem (bg_{store}, unq_{store}_clips); value = firmographics
# match predicate (NAICS/SIC codes and/or business-name LIKEs).
# NAICS codes are dual-vintage: subsector 447 (Gasoline Stations, 2017) was renumbered
# to 457 in NAICS 2022, and 445120 (Convenience Stores, 2017) became 445131 (Convenience
# Retailers, 2022). The firmographics view mixes vintages, so each concept lists both.
# Split is non-overlapping: gas-with-mart -> convenience (attractor); fuel-only -> gas.
STORE_DEFS = {
    "convenience_stores": (
        # convenience stores (445131=2022, 445120=2017) + gas w/ convenience mart (457110=2022, 447110=2017)
        "naics_6_digit_primary_code IN ('445131','445120','457110','447110')"
    ),
    "gas_stations": (
        # fuel-only stations (457120=2022, 447190=2017); mart stations counted under convenience
        "naics_6_digit_primary_code IN ('457120','447190')\n"
        "     AND NOT LOWER(business_name) LIKE '%charging station%'"
    ),
    "liquor_stores": (
        # beer/wine/liquor retailers, 4453x covers 445310 (2017) & 445320 (2022)
        "naics_6_digit_primary_code LIKE '4453%'\n"
        "     AND (LOWER(business_brand_name) LIKE '%liquor%'"
        " OR LOWER(business_name) LIKE '%liquor%')"
    ),
}

# crime categories we model (each has a *_count and *_rate column downstream)
TARGET_CATEGORIES = [
    "cl_total", "violent", "property",
    "assault", "murder", "rape", "robbery",
    "burglary", "larceny", "mvt",
]

# predictors that vary within a single city (Division is constant per-city → excluded)
PREDICTOR_COLS = [
    "det_pct",              # Percentage of housing units that are single family detached houses
    "in_household_pct",     # Percent of the population living in households
    "moved1yr_pct",         # Percentage of households moving in past year
    "city_centers_dist",    # Distance in miles from central business district of nearest city
    "pop_est_5mile",        # Population density within 5 miles
    "pop_ch_1mile",         # Population change within 1 mile
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
]

ZERO_FILL = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
]                                                       # 0 = none observed
MEDIAN_FILL = ["city_centers_dist", "pop_est_5mile", "pop_ch_1mile"]  # 0 would be wrong

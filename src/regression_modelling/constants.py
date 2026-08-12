"""Registries + column lists for the regression_modelling task."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSource:
    name: str                       # 'vacancy' → cache file name
    backend: str                    # 'bq' or 'gcs'
    location: str                   # BQ: pull-sql name | GCS: path template
    key_col: str = "geoid"          # join key in the pulled data
    feature_cols: tuple = ()        # predictor columns to keep


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
    "seven_eleven": FeatureSource(
        name="seven_eleven",
        backend="bq",
        location="seven_eleven",      # → sql/pull/seven_eleven.sql
        key_col="geoid",
        feature_cols=("unq_seven_eleven_clips",),
    ),
    "gas_stations": FeatureSource(
        name="gas_stations",
        backend="bq",
        location="gas_stations",      # → sql/pull/gas_stations.sql
        key_col="geoid",
        feature_cols=("unq_gas_station_clips",),
    ),
    "liquor_stores": FeatureSource(
        name="liquor_stores",
        backend="bq",
        location="liquor_stores",     # → sql/pull/liquor_stores.sql
        key_col="geoid",
        feature_cols=("unq_liquor_store_clips",),
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
    "unq_seven_eleven_clips",
    "unq_gas_station_clips",
    "unq_liquor_store_clips",
]

ZERO_FILL = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "unq_seven_eleven_clips",
    "unq_gas_station_clips",
    "unq_liquor_store_clips",
]                                                       # 0 = none observed
MEDIAN_FILL = ["city_centers_dist", "pop_est_5mile", "pop_ch_1mile"]  # 0 would be wrong

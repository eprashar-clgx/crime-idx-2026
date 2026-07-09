from core.config import PROJECT_ROOT, INTERIM_DIR
from dataclasses import dataclass

SQL_DIR = PROJECT_ROOT / "src" / "prediction" / "sql"

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

def source_parquet(name: str) -> str:
    return INTERIM_DIR / "sources" / f"{name}.parquet"
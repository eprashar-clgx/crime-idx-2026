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
}

def source_parquet(name: str) -> str:
    return INTERIM_DIR / "sources" / f"{name}.parquet"
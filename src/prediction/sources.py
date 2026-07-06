import tempfile
import pandas as pd
import pyreadstat
from gcsfs import GCSFileSystem
from google.cloud import bigquery

from core.config import (
    GCS_PROJECT, GCS_ROOT, UCR_YEAR,
    BQ_PROJECT, IDAP_PROJECT, BOUNDARY_DATASET, BQ_STAGING_DATASET,
)
from prediction.config import SQL_DIR, source_parquet

# --- GCS ---
def get_gcs_fs() -> GCSFileSystem:
    return GCSFileSystem(project=GCS_PROJECT, token="google_default")

def read_sav_from_gcs(gcs_path: str, fs: GCSFileSystem):
    with fs.open(gcs_path, "rb") as gcs_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(gcs_file.read())
            tmp.flush()
            tmp.close()
            df, meta = pyreadstat.read_sav(tmp.name)
    return df, meta

# --- BQ ---
_client = None
def _bq_client() -> bigquery.Client:
    global _client
    if _client is None:
        _client = bigquery.Client(project=BQ_PROJECT)
    return _client

_SQL_DEFAULTS = {
    "bq_project":       BQ_PROJECT,
    "idap_project":     IDAP_PROJECT,
    "boundary_dataset": BOUNDARY_DATASET,
    "staging_dataset":  BQ_STAGING_DATASET,
}

def load_sql(name: str, kind: str = "build", **overrides) -> str:
    text = (SQL_DIR / kind / f"{name}.sql").read_text()
    return text.format(**{**_SQL_DEFAULTS, **overrides})

def run_bq_build(name: str) -> None:
    """Materialize a feature table in BQ (runs the CREATE OR REPLACE DDL)."""
    _bq_client().query(load_sql(name, "build")).result()

def run_bq_pull(name: str) -> pd.DataFrame:
    """Pull a materialized feature table into a DataFrame."""
    return _bq_client().query(load_sql(name, "pull")).to_dataframe()

# --- generic source dispatch (registry-driven) ---
def pull_source(src, refresh: bool = False) -> pd.DataFrame:
    path = source_parquet(src.name)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    if src.backend == "bq":
        df = run_bq_pull(src.location)
    elif src.backend == "gcs":
        df, _ = read_sav_from_gcs(f"{GCS_ROOT}/{src.location.format(year=UCR_YEAR)}", get_gcs_fs())
    else:
        raise ValueError(f"unknown backend: {src.backend}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
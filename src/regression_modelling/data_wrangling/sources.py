import os
import tempfile
import pandas as pd
import pyreadstat
from gcsfs import GCSFileSystem
from google.cloud import bigquery

from crime_blockgroup_mapping.constants import (
    GCS_PROJECT, GCS_ROOT, UCR_YEAR,
    BQ_PROJECT, IDAP_PROJECT, IMAGERY_PROJECT, BOUNDARY_DATASET, BQ_STAGING_DATASET,
)
from regression_modelling.config import SQL_DIR, source_parquet

# --- GCS ---
def get_gcs_fs() -> GCSFileSystem:
    return GCSFileSystem(project=GCS_PROJECT, token="google_default")

def read_sav_from_gcs(gcs_path: str, fs: GCSFileSystem):
    # pyreadstat needs a real file path; /tmp is RAM-backed here, so always unlink.
    with tempfile.NamedTemporaryFile(suffix=".sav", delete=False) as tmp:
        tmp_path = tmp.name
        with fs.open(gcs_path, "rb") as gcs_file:
            tmp.write(gcs_file.read())
    try:
        df, meta = pyreadstat.read_sav(tmp_path)
    finally:
        os.unlink(tmp_path)
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
    "imagery_project":  IMAGERY_PROJECT,
    "boundary_dataset": BOUNDARY_DATASET,
    "staging_dataset":  BQ_STAGING_DATASET,
}

def load_sql(name: str, kind: str = "build", **overrides) -> str:
    text = (SQL_DIR / kind / f"{name}.sql").read_text()
    return text.format(**{**_SQL_DEFAULTS, **overrides})

def run_bq_build(name: str) -> None:
    """Materialize a feature table in BQ (runs the CREATE OR REPLACE DDL)."""
    _bq_client().query(load_sql(name, "build")).result()


def run_bq_build_store(store: str) -> None:
    """Materialize a store block-group count table via the generic stores.sql template.

    Renders sql/build/stores.sql with the store's stem + firmographics match predicate
    from STORE_DEFS (regression_modelling.constants). NOTE: firmographics live in a prd
    project; this must be run where those tables are reachable.
    """
    from regression_modelling.constants import STORE_DEFS
    sql = load_sql("stores", "build", store=store, match_predicate=STORE_DEFS[store])
    _bq_client().query(sql).result()


def run_bq_build_store_points(store: str) -> None:
    """Materialize a store POI point table (`{store}_points`) via sql/build/store_points.sql.

    Same STORE_DEFS universe as `run_bq_build_store`, geocoded with EAP lat/lon from
    clip_address_xref. NOTE: prd tables — run where reachable (or paste the rendered SQL
    into the BQ console).
    """
    from regression_modelling.constants import STORE_DEFS
    sql = load_sql("store_points", "build", store=store, match_predicate=STORE_DEFS[store])
    _bq_client().query(sql).result()


def run_bq_pull_store_points(store: str) -> pd.DataFrame:
    """Pull a materialized `{store}_points` table: clip_id, geoid, lat, lon."""
    return _bq_client().query(load_sql("store_points", "pull", store=store)).to_dataframe()


def run_bq_build_imagery() -> None:
    """Materialize the national BG Vexcel imagery table via sql/build/imagery.sql.

    NOTE: Vexcel lives in a prd project; run this where those tables are reachable (BQ
    console), which writes bg_imagery to the dev staging dataset that run_bq_pull can
    then read.
    """
    _bq_client().query(load_sql("imagery", "build")).result()

def run_bq_pull(name: str) -> pd.DataFrame:
    """Pull a materialized feature table into a DataFrame."""
    return _bq_client().query(load_sql(name, "pull")).to_dataframe()

# --- generic source dispatch (registry-driven) ---
def pull_source(src, refresh: bool = False) -> pd.DataFrame:
    path = source_parquet(src.name)
    # file-backed sources are materialized out-of-band by a dedicated builder (e.g. the
    # transit submodule); pull_source only reads the cached artifact — nothing to re-fetch.
    if src.backend == "file":
        if not path.exists():
            raise FileNotFoundError(
                f"file-backed source '{src.name}' not found at {path}; build it first "
                f"({src.location})."
            )
        return pd.read_parquet(path)
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


def run_bq_explore(name: str, **params) -> pd.DataFrame:
    """Run a parametrized exploratory query (sql/explore/{name}.sql) into a DataFrame.

    `params` fill template placeholders (e.g. name_predicate, naics_predicate) on top
    of the project/dataset defaults; unused extras are ignored by str.format.
    """
    return _bq_client().query(load_sql(name, "explore", **params)).to_dataframe()

"""Load the bias-testing-only protected-attribute table (ADR 0004).

Pulls the protected columns from the same ACS `.sav` as the demographic predictors, but
caches them to a SEPARATE tier (`data/interim/bias/`) keyed by `geoid`. These columns are
never returned by `assemble_features` and never enter `PREDICTOR_COLS` — they are joined to
model results by `geoid` only at bias-test time, against the fit's filtered geoid set.
"""
import pandas as pd

from crime_blockgroup_mapping.constants import GCS_ROOT, UCR_YEAR
from regression_modelling.config import bias_parquet
from regression_modelling.bias_testing.constants import PROTECTED_COLS
from regression_modelling.data_wrangling.sources import get_gcs_fs, read_sav_from_gcs


def build_protected_table(year: int = UCR_YEAR, refresh: bool = False) -> pd.DataFrame:
    """Protected-attribute table keyed by `geoid` (cached under data/interim/bias/).

    Reads only `PROTECTED_COLS` from bg_acs.sav; caches to `bias_parquet()`. Follows the
    split-ingestion convention (`refresh=False` reuses the cache).
    """
    path = bias_parquet()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    fs = get_gcs_fs()
    bg_acs, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/acs/5/{year-1}/bg_acs.sav", fs)
    out = bg_acs[["bg_key"] + PROTECTED_COLS].rename(columns={"bg_key": "geoid"}).copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"protected table {out.shape} → {path}")
    return out


def load_protected_table() -> pd.DataFrame:
    """Read the cached protected table; raise if it hasn't been built yet."""
    path = bias_parquet()
    if not path.exists():
        raise FileNotFoundError(
            f"protected table not built: {path}. Run build_protected_table(refresh=True)."
        )
    return pd.read_parquet(path)

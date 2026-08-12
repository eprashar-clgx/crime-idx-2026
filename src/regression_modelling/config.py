"""Paths only for the regression_modelling task. Constants live in constants.py."""
from crime_blockgroup_mapping.config import PROJECT_ROOT, INTERIM_DIR

# all task SQL templates live under data_wrangling/sql/{build,pull,explore}
SQL_DIR = PROJECT_ROOT / "src" / "regression_modelling" / "data_wrangling" / "sql"


def source_parquet(name: str):
    return INTERIM_DIR / "sources" / f"{name}.parquet"

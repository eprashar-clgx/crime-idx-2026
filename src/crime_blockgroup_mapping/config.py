"""Paths only. All constants/dataclasses/registries live in constants.py."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# data tiers
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# raw sources
BOUNDARIES_DIR = RAW_DIR / "boundaries"
PLACES_PATH    = BOUNDARIES_DIR / "cb_2024_us_place_500k"
MODEL_SAV      = RAW_DIR / "neighborhood_scout" / "location_inc_ns4_2025q4_block_group_data.sav"
LODES_PATH     = RAW_DIR / "lodes" / "location_inc_spatial_lodes_wac_2022_block_jobs.csv"

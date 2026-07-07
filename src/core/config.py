from pathlib import Path
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# data tiers
RAW_DIR       = DATA_DIR / "raw"
INTERIM_DIR   = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# raw sources (post-reorg locations)
BOUNDARIES_DIR = RAW_DIR / "boundaries"
PLACES_PATH    = BOUNDARIES_DIR / "cb_2024_us_place_500k"
MODEL_SAV      = RAW_DIR / "neighborhood_scout" / "location_inc_ns4_2025q4_block_group_data.sav"
LODES_PATH     = RAW_DIR / "lodes" / "location_inc_spatial_lodes_wac_2022_block_jobs.csv"


# NIBRS → model category mapping (Houston)
NIBRS_TO_CATEGORY = {
    '13A': 'assault',
    '220': 'burglary',
    '23A': 'larceny', '23B': 'larceny', '23C': 'larceny', '23D': 'larceny',
    '23E': 'larceny', '23F': 'larceny', '23G': 'larceny', '23H': 'larceny',
    '09A': 'murder', '09B': 'murder', '09C': 'murder',
    '240': 'mvt',
    '11A': 'rape', '11B': 'rape', '11C': 'rape', '11D': 'rape', '36B': 'rape',
    '120': 'robbery',
    '290': 'vandal',
    '200': 'fire',
}

# Chicago FBI Code → model category mapping
CHICAGO_FBI_TO_CATEGORY = {
    '01A': 'murder',
    '01B': 'murder',
    '02': 'rape',
    '03': 'robbery',
    '04A': 'assault',
    '04B': 'assault',
    '05': 'burglary',
    '06': 'larceny',
    '07': 'mvt',
    '09': 'fire',
    '14': 'vandal',
}

CRIME_CATEGORIES = [
    'assault', 'burglary', 'larceny', 'murder', 'mvt',
    'rape', 'robbery', 'vandal', 'fire',
    'violent', 'property', 'total', 'cl_total', 'wtotal', 'wprop'
]

@dataclass
class CityConfig:
    name: str
    state_fips: str
    place_fips: str
    crime_csv: str
    lat_col: str
    lon_col: str
    crime_type_col: str
    crime_type_mapping: dict = field(repr=False)

    @property
    def bg_zip(self) -> Path:
        return BOUNDARIES_DIR / f"cb_2025_{self.state_fips}_bg_500k.zip"

CITIES = {
    "houston": CityConfig(
        name="Houston",
        state_fips="48",
        place_fips="35000",
        crime_csv="raw/city_crime/houston/NIBRSPublicView2025.csv",
        lat_col="map_latitude",
        lon_col="map_longitude",
        crime_type_col="nibrs_class",
        crime_type_mapping=NIBRS_TO_CATEGORY,
    ),
    "chicago": CityConfig(
        name="Chicago",
        state_fips="17",
        place_fips="14000",
        crime_csv="raw/city_crime/chicago/Crimes_-_2025_20260514.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="fbi_code",
        crime_type_mapping=CHICAGO_FBI_TO_CATEGORY,
    ),
}

# --- BigQuery projects / datasets ---
BQ_PROJECT         = "clgx-gis-app-dev-06e3"        # billing + GIS/boundary + staging
IDAP_PROJECT       = "clgx-idap-bigquery-prd-a990"  # enterprise property data
BOUNDARY_DATASET   = "boundary"
BQ_STAGING_DATASET = "work_eprashar"                # where feature build tables land

# --- GCS + data tiers ---
GCS_PROJECT = "clgx-gis-app-dev-06e3"
GCS_ROOT    = "gs://geospatial-projects/location_inc"
UCR_YEAR    = 2024


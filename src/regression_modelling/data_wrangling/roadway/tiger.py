"""Load US Census TIGER/Line county roads, filtered to the MTFCC classes we need.

Public-domain source (see docs/features/roadway_plan.md §5): one zip per (state, county)
FIPS, downloaded once to data/raw/roadway/tiger/ and cached filtered+reprojected to
data/interim/roadway/tiger/ so repeat builds run offline.

MTFCC codes kept: S1100 (primary road / interstate & limited-access), S1200 (secondary
road / arterial), S1630 (ramp). Everything else (local roads S1400, etc.) is dropped
immediately after read — it dominates row count but is not part of any candidate feature.
"""
from __future__ import annotations

import urllib.request

import geopandas as gpd
import pandas as pd

from regression_modelling.config import roadway_tiger_county_parquet, tiger_roads_zip

TIGER_YEAR = 2024
TIGER_ROADS_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/ROADS/tl_{year}_{fips}_roads.zip"

# MTFCC classes this pipeline uses (see roadway_plan.md §5 "Technical notes for the build").
MTFCC_RAMP = "S1630"
MTFCC_INTERSTATE = "S1100"
MTFCC_ARTERIAL = "S1200"
_KEEP_MTFCC = (MTFCC_RAMP, MTFCC_INTERSTATE, MTFCC_ARTERIAL)

# Equal-area CRS (CONUS Albers) shared with transit's density/length math.
_EQUAL_AREA_CRS = "EPSG:5070"


def _download_county_roads(state_fips: str, county_fips: str, year: int = TIGER_YEAR) -> None:
    dest = tiger_roads_zip(state_fips, county_fips, year)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = TIGER_ROADS_URL.format(year=year, fips=f"{state_fips}{county_fips}")
    print(f"roadway[tiger]: downloading {url}")
    urllib.request.urlretrieve(url, dest)


def load_county_roads(state_fips: str, county_fips: str, year: int = TIGER_YEAR,
                      refresh: bool = False) -> gpd.GeoDataFrame:
    """Ramp/interstate/arterial road segments for one county, reprojected to EPSG:5070.

    Cached to data/interim/roadway/tiger/{state}{county}_{year}.parquet. Columns kept:
    geoid-agnostic (this is a county-wide layer, joined to BGs downstream) `mtfcc`,
    `fullname`, `geometry`.
    """
    cache = roadway_tiger_county_parquet(state_fips, county_fips, year)
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)

    _download_county_roads(state_fips, county_fips, year)
    zip_path = tiger_roads_zip(state_fips, county_fips, year)
    raw = gpd.read_file(f"zip://{zip_path}")
    raw.columns = raw.columns.str.lower()
    roads = raw[raw["mtfcc"].isin(_KEEP_MTFCC)][["mtfcc", "fullname", "geometry"]].copy()
    roads = roads.set_crs("EPSG:4269", allow_override=True).to_crs(_EQUAL_AREA_CRS)

    cache.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(cache)
    print(f"roadway[tiger]: {state_fips}{county_fips} -> {len(roads):,} road segments "
          f"(ramp/interstate/arterial) -> {cache}")
    return roads


def load_roads_for_counties(counties: list[tuple[str, str]], year: int = TIGER_YEAR,
                            refresh: bool = False) -> gpd.GeoDataFrame:
    """Concat road segments across every (state_fips, county_fips) pair a city touches."""
    frames = [load_county_roads(s, c, year, refresh=refresh) for s, c in counties]
    if not frames:
        return gpd.GeoDataFrame(columns=["mtfcc", "fullname", "geometry"], crs=_EQUAL_AREA_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=_EQUAL_AREA_CRS)

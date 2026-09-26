"""Load US Census TIGER/Line county layers: ROADS (ramps/interstates) and EDGES (junctions).

Public-domain source (see docs/features/roadway_plan.md §5): one zip per (state, county)
FIPS per layer, downloaded once to data/raw/roadway/tiger/ and cached filtered+reprojected
to data/interim/roadway/ so repeat builds run offline.

ROADS — one record per named road feature. MTFCC codes kept: S1100 (primary road /
interstate & limited-access), S1200 (secondary road), S1630 (ramp). Local streets (S1400)
etc. are dropped immediately after read.

EDGES — the topological layer: every edge is split at every node and carries from/to node
ids (TNIDF/TNIDT). Junction degree = number of public-street edges (S1200 + S1400) incident
on a node. ROADS can't provide this (its records cross at interior vertices, not shared
endpoints). Limited-access roads (S1100) and ramps (S1630) are excluded on purpose: TIGER
topology is planar, so an overpass would otherwise register as a false junction.
"""
from __future__ import annotations

import urllib.request

import geopandas as gpd
import pandas as pd

from regression_modelling.config import (
    roadway_tiger_county_parquet, roadway_tiger_junctions_parquet, tiger_edges_zip, tiger_roads_zip,
)

TIGER_YEAR = 2024
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/{layer}/tl_{year}_{fips}_{suffix}.zip"

# MTFCC classes this pipeline uses (see roadway_plan.md §5 "Technical notes for the build").
MTFCC_RAMP = "S1630"
MTFCC_INTERSTATE = "S1100"
MTFCC_SECONDARY = "S1200"
MTFCC_LOCAL = "S1400"
_KEEP_MTFCC = (MTFCC_RAMP, MTFCC_INTERSTATE, MTFCC_SECONDARY)
# Edges counted toward junction degree: at-grade public streets.
JUNCTION_MTFCC = (MTFCC_SECONDARY, MTFCC_LOCAL)
# A node needs this many incident street edges to be a real junction (1 = dead end,
# 2 = a street merely split by a non-road feature such as a boundary or stream).
MIN_JUNCTION_DEGREE = 3

# Equal-area CRS (CONUS Albers) shared with transit's density/length math.
_EQUAL_AREA_CRS = "EPSG:5070"


def _download(layer: str, suffix: str, dest, state_fips: str, county_fips: str, year: int) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = TIGER_URL.format(year=year, layer=layer, fips=f"{state_fips}{county_fips}", suffix=suffix)
    print(f"roadway[tiger]: downloading {url}")
    urllib.request.urlretrieve(url, dest)


def load_county_roads(state_fips: str, county_fips: str, year: int = TIGER_YEAR,
                      refresh: bool = False) -> gpd.GeoDataFrame:
    """Ramp/interstate/secondary road features for one county, reprojected to EPSG:5070.

    Cached to data/interim/roadway/tiger/{state}{county}_{year}.parquet. Columns kept:
    `mtfcc`, `fullname`, `geometry` (county-wide layer, joined to BGs downstream).
    """
    cache = roadway_tiger_county_parquet(state_fips, county_fips, year)
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)

    zip_path = tiger_roads_zip(state_fips, county_fips, year)
    _download("ROADS", "roads", zip_path, state_fips, county_fips, year)
    raw = gpd.read_file(f"zip://{zip_path}")
    raw.columns = raw.columns.str.lower()
    roads = raw[raw["mtfcc"].isin(_KEEP_MTFCC)][["mtfcc", "fullname", "geometry"]].copy()
    roads = roads.set_crs("EPSG:4269", allow_override=True).to_crs(_EQUAL_AREA_CRS)

    cache.parent.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(cache)
    print(f"roadway[tiger]: {state_fips}{county_fips} -> {len(roads):,} road features "
          f"(ramp/interstate/secondary) -> {cache}")
    return roads


def load_roads_for_counties(counties: list[tuple[str, str]], year: int = TIGER_YEAR,
                            refresh: bool = False) -> gpd.GeoDataFrame:
    """Concat road features across every (state_fips, county_fips) pair a city touches."""
    frames = [load_county_roads(s, c, year, refresh=refresh) for s, c in counties]
    if not frames:
        return gpd.GeoDataFrame(columns=["mtfcc", "fullname", "geometry"], crs=_EQUAL_AREA_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=_EQUAL_AREA_CRS)


def load_county_junctions(state_fips: str, county_fips: str, year: int = TIGER_YEAR,
                          refresh: bool = False) -> gpd.GeoDataFrame:
    """Street-junction points (degree >= MIN_JUNCTION_DEGREE) for one county, EPSG:5070.

    Degree counts incident S1200/S1400 edges per TIGER node id; the node's location is
    taken from the start/end vertex of any incident edge. Cached to
    data/interim/roadway/tiger_junctions/{state}{county}_{year}.parquet.
    """
    cache = roadway_tiger_junctions_parquet(state_fips, county_fips, year)
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)

    zip_path = tiger_edges_zip(state_fips, county_fips, year)
    _download("EDGES", "edges", zip_path, state_fips, county_fips, year)
    edges = gpd.read_file(f"zip://{zip_path}", columns=["MTFCC", "TNIDF", "TNIDT"])
    edges = edges[edges["MTFCC"].isin(JUNCTION_MTFCC)]

    ends = pd.concat([
        pd.DataFrame({"node": edges["TNIDF"].values,
                      "x": edges.geometry.map(lambda g: g.coords[0][0]).values,
                      "y": edges.geometry.map(lambda g: g.coords[0][1]).values}),
        pd.DataFrame({"node": edges["TNIDT"].values,
                      "x": edges.geometry.map(lambda g: g.coords[-1][0]).values,
                      "y": edges.geometry.map(lambda g: g.coords[-1][1]).values}),
    ])
    nodes = ends.groupby("node").agg(degree=("node", "size"), x=("x", "first"), y=("y", "first"))
    nodes = nodes[nodes["degree"] >= MIN_JUNCTION_DEGREE].reset_index()

    out = gpd.GeoDataFrame(nodes[["node", "degree"]],
                           geometry=gpd.points_from_xy(nodes["x"], nodes["y"]),
                           crs="EPSG:4269").to_crs(_EQUAL_AREA_CRS)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"roadway[tiger]: {state_fips}{county_fips} -> {len(out):,} street junctions "
          f"(degree>={MIN_JUNCTION_DEGREE}) -> {cache}")
    return out


def load_junctions_for_counties(counties: list[tuple[str, str]], year: int = TIGER_YEAR,
                                refresh: bool = False) -> gpd.GeoDataFrame:
    """Concat junction points across every (state_fips, county_fips) pair a city touches."""
    frames = [load_county_junctions(s, c, year, refresh=refresh) for s, c in counties]
    if not frames:
        return gpd.GeoDataFrame(columns=["node", "degree", "geometry"], crs=_EQUAL_AREA_CRS)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=_EQUAL_AREA_CRS)

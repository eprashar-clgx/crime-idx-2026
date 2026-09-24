"""Build the BG-level roadway feature table: TIGER roads + Overture junctions -> geoid.

`build_roadway(city)` materializes a per-city contribution; `build_all_roadway()` stacks
all model cities into data/interim/sources/roadway.parquet — the parquet the `roadway`
FeatureSource (backend="file") reads via `pull_source`, mirroring transit.build (see
docs/features/roadway_plan.md §6). Building is out-of-band: nothing in the normal pull
path triggers it.

Five candidate raw features (roadway_plan.md §4):
    roadway_nearest_ramp_m        BG centroid -> nearest ramp (R1)
    roadway_nearest_interstate_m  BG centroid -> nearest interstate/limited-access (R1)
    roadway_ramp_count            ramp segments intersecting the BG polygon (R1)
    roadway_arterial_density      km of interstate+arterial road per km^2, clipped to BG (R2)
    roadway_intersection_density  Overture junctions (degree>=3) per km^2 (R3, exploratory)
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd

from crime_blockgroup_mapping.constants import CITIES
from crime_blockgroup_mapping.boundaries import (
    load_city_boundary, load_state_block_groups, label_bgs_within_city,
)
from regression_modelling.config import source_parquet
from regression_modelling.data_wrangling.roadway.tiger import (
    load_roads_for_counties, MTFCC_RAMP, MTFCC_INTERSTATE, MTFCC_ARTERIAL,
)
from regression_modelling.data_wrangling.roadway.overture import load_city_intersections

_EQUAL_AREA_CRS = "EPSG:5070"

_KEEP = ["geoid", "city", "roadway_nearest_ramp_m", "roadway_nearest_interstate_m",
        "roadway_ramp_count", "roadway_arterial_density", "roadway_intersection_density"]


def _nearest_distance_m(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> np.ndarray:
    """Distance (meters) from each point to the nearest target geometry, both EPSG:5070."""
    if targets.empty:
        return np.full(len(points), np.inf)
    joined = gpd.sjoin_nearest(points, targets[["geometry"]], distance_col="_dist_m")
    # sjoin_nearest can emit >1 row per left geometry on exact ties; keep the closest.
    joined = joined.groupby(joined.index)["_dist_m"].min()
    return joined.reindex(points.index).to_numpy()


def build_roadway(city: str, refresh: bool = False) -> pd.DataFrame:
    """BG-level roadway predictors for one city, keyed by `geoid`.

    Steps: load within-city BGs (foundation, same as transit) -> pull TIGER roads for
    every county the city's BGs touch -> nearest-distance (ramp, interstate) from each BG
    centroid -> ramp count + arterial density clipped to each BG polygon -> Overture
    junction density. Every within-city BG is emitted (roadless BGs get a real, large
    nearest-distance and zero counts/densities — there are no structural nulls here,
    unlike transit, since TIGER covers every county nationally).
    """
    cfg = CITIES[city]
    bg = load_state_block_groups(cfg)
    bg = label_bgs_within_city(bg, load_city_boundary(cfg))
    bg_city = bg[bg["within_city"]][["geoid", "county_fips", "geometry"]].copy()
    bg_city = bg_city.set_geometry("geometry").to_crs(_EQUAL_AREA_CRS)

    counties = sorted({(cfg.state_fips, cf[len(cfg.state_fips):]) for cf in bg_city["county_fips"]})
    roads = load_roads_for_counties(counties, refresh=refresh)
    print(f"roadway[{city}]: {len(roads):,} road segments across {len(counties)} count(y/ies)")

    ramps = roads[roads["mtfcc"] == MTFCC_RAMP]
    interstates = roads[roads["mtfcc"] == MTFCC_INTERSTATE]
    arterials = roads[roads["mtfcc"].isin([MTFCC_INTERSTATE, MTFCC_ARTERIAL])]

    centroids = bg_city.copy()
    centroids["geometry"] = centroids.geometry.centroid
    out = bg_city[["geoid"]].copy()
    out["roadway_nearest_ramp_m"] = _nearest_distance_m(centroids, ramps)
    out["roadway_nearest_interstate_m"] = _nearest_distance_m(centroids, interstates)

    # Ramp count: segments intersecting the BG polygon (predicate="intersects" so a ramp
    # straddling a BG boundary counts for both, matching the "access point in this BG" idea).
    if ramps.empty:
        out["roadway_ramp_count"] = 0
    else:
        hit = gpd.sjoin(bg_city[["geoid", "geometry"]], ramps[["geometry"]], predicate="intersects")
        counts = hit.groupby("geoid").size()
        out["roadway_ramp_count"] = out["geoid"].map(counts).fillna(0).astype(int)

    # Arterial density: length clipped to each BG (overlay), summed per geoid / BG area.
    area_km2 = bg_city.set_index("geoid").geometry.area / 1e6
    if arterials.empty:
        out["roadway_arterial_density"] = 0.0
    else:
        clipped = gpd.overlay(arterials[["geometry"]], bg_city[["geoid", "geometry"]], how="intersection")
        length_km = clipped.length / 1000.0
        len_by_bg = length_km.groupby(clipped["geoid"]).sum()
        density = (len_by_bg / area_km2).reindex(bg_city["geoid"]).fillna(0.0)
        out["roadway_arterial_density"] = out["geoid"].map(density).to_numpy()

    # Overture junction density (R3, exploratory).
    bounds = load_city_boundary(cfg).to_crs("EPSG:4326").total_bounds  # (xmin, ymin, xmax, ymax)
    junctions = load_city_intersections(city, tuple(bounds), refresh=refresh)
    if junctions.empty:
        out["roadway_intersection_density"] = 0.0
    else:
        hit = gpd.sjoin(bg_city[["geoid", "geometry"]], junctions[["geometry"]], predicate="intersects")
        counts = hit.groupby("geoid").size()
        density = (counts / area_km2.reindex(counts.index)).reindex(bg_city["geoid"]).fillna(0.0)
        out["roadway_intersection_density"] = out["geoid"].map(density).to_numpy()

    out["city"] = city
    result = out[_KEEP].reset_index(drop=True)
    print(f"roadway[{city}]: BG table {result.shape} "
          f"({int((result.roadway_ramp_count > 0).sum())} BGs with >=1 ramp)")
    return result


def build_all_roadway(refresh: bool = False) -> pd.DataFrame:
    """Stack per-city BG roadway tables for all model cities and cache the registry parquet.

    Writes data/interim/sources/roadway.parquet (== source_parquet("roadway")), which the
    `roadway` FeatureSource reads. Unlike transit, TIGER is a national layer, so this could
    in principle run for any BG — scoped here to the same city set for parity with the
    other feature families (and because BG polygons/boundaries are already loaded per-city).
    """
    frames = [build_roadway(city, refresh=refresh) for city in CITIES]
    out = pd.concat(frames, ignore_index=True)
    path = source_parquet("roadway")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"roadway: BG feature table {out.shape} -> {path}")
    return out

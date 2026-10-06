"""Build the BG-level roadway feature table: TIGER roads/edges + HPMS arterials -> geoid.

`build_roadway(city)` materializes a per-city contribution; `build_all_roadway()` stacks
all model cities into data/interim/sources/roadway.parquet — the parquet the `roadway`
FeatureSource (backend="file") reads via `pull_source`, mirroring transit.build (see
docs/features/roadway_plan.md §6). Building is out-of-band: nothing in the normal pull
path triggers it. All sources are public domain (TIGER: Census; HPMS: FHWA).

Five candidate raw features (roadway_plan.md §4):
    roadway_nearest_ramp_m        BG centroid -> nearest ramp, TIGER S1630 (R1)
    roadway_nearest_interstate_m  BG centroid -> nearest interstate, TIGER S1100 (R1)
    roadway_ramp_count            TIGER ramp features intersecting the BG polygon (R1)
    roadway_arterial_density      km of HPMS arterial (f_system 3-4) per km^2, clipped to BG (R2)
    roadway_intersection_density  TIGER EDGES street junctions (degree>=3) per km^2 (R3)

Two street-network morphology features (roadway_plan.md §4 R3b), from TIGER EDGES node degree:
    roadway_x_ratio               4+-way junctions / all junctions (NaN: no junctions)
    roadway_deadend_share         dead ends / (dead ends + junctions) (NaN: no street nodes)
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
    load_roads_for_counties, load_junctions_for_counties, MTFCC_RAMP, MTFCC_INTERSTATE,
    street_nodes, classify_nodes,
)
from regression_modelling.data_wrangling.roadway.hpms import (
    load_hpms_for_counties, F_SYSTEM_ARTERIAL, FACILITY_MAINLINE,
)

_EQUAL_AREA_CRS = "EPSG:5070"
# Same-state counties within this distance of the city's BGs are also loaded for node degree,
# so a street cut by a county line is not miscounted as a dead end.
NODE_COUNTY_BUFFER_M = 500

_KEEP = ["geoid", "city", "roadway_nearest_ramp_m", "roadway_nearest_interstate_m",
        "roadway_ramp_count", "roadway_arterial_density", "roadway_intersection_density",
        "roadway_x_ratio", "roadway_deadend_share"]


def _nearest_distance_m(points: gpd.GeoDataFrame, targets: gpd.GeoDataFrame) -> np.ndarray:
    """Distance (meters) from each point to the nearest target geometry, both EPSG:5070."""
    if targets.empty:
        return np.full(len(points), np.inf)
    joined = gpd.sjoin_nearest(points, targets[["geometry"]], distance_col="_dist_m")
    # sjoin_nearest can emit >1 row per left geometry on exact ties; keep the closest.
    joined = joined.groupby(joined.index)["_dist_m"].min()
    return joined.reindex(points.index).to_numpy()


def _node_counties(bg_state: gpd.GeoDataFrame, bg_city: gpd.GeoDataFrame, state_fips: str,
                   buffer_m: float = NODE_COUNTY_BUFFER_M) -> list[tuple[str, str]]:
    """City counties plus any same-state county with a BG within `buffer_m` of the city's BGs.

    Cross-state neighbours are not added (the state BG layer stops at the state line), so
    streets crossing a state line (DC, Kansas City, NYC/Philadelphia–NJ) can still read as
    dead ends — ~4% of DC's dead-end nodes sit within 200 m of the line.
    """
    state = bg_state.to_crs(_EQUAL_AREA_CRS)
    zone = bg_city.union_all().buffer(buffer_m)
    hit = state.iloc[state.sindex.query(zone, predicate="intersects")]
    return sorted({(state_fips, cf[len(state_fips):]) for cf in hit["county_fips"]})


def street_morphology(bg_city: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> pd.DataFrame:
    """Per-BG x_ratio and deadend_share from classified street nodes (index = geoid).

    Nodes are assigned to the BG polygon they fall in ("within"; a node exactly on a BG
    line is dropped rather than double-counted). Undefined ratios are left NaN — the fill
    policy lives in ZERO_FILL / MEDIAN_FILL (constants.py), not here.
    """
    kinds = pd.concat([nodes[["geometry"]], classify_nodes(nodes)], axis=1)
    kinds = kinds[kinds[["is_t", "is_x", "is_dead"]].any(axis=1)]
    hit = gpd.sjoin(kinds, bg_city[["geoid", "geometry"]], predicate="within")
    c = hit.groupby("geoid")[["is_t", "is_x", "is_dead"]].sum().reindex(bg_city["geoid"]).fillna(0)
    junctions = c["is_t"] + c["is_x"]
    with np.errstate(invalid="ignore", divide="ignore"):
        x_ratio = c["is_x"] / junctions
        dead_share = c["is_dead"] / (c["is_dead"] + junctions)
    return pd.DataFrame({"roadway_x_ratio": x_ratio, "roadway_deadend_share": dead_share})


def build_roadway(city: str, refresh: bool = False) -> pd.DataFrame:
    """BG-level roadway predictors for one city, keyed by `geoid`.

    Steps: load within-city BGs (foundation, same as transit) -> pull TIGER roads, TIGER
    edge junctions and HPMS sections for every county the city's BGs touch ->
    nearest-distance (ramp, interstate) from each BG centroid -> ramp count, arterial
    density and junction density per BG polygon. Every within-city BG is emitted (roadless
    BGs get a real, large nearest-distance and zero counts/densities — there are no
    structural nulls here, unlike transit, since all three sources cover every county).
    """
    cfg = CITIES[city]
    bg = load_state_block_groups(cfg)
    bg = label_bgs_within_city(bg, load_city_boundary(cfg))
    bg_city = bg[bg["within_city"]][["geoid", "county_fips", "geometry"]].copy()
    bg_city = bg_city.set_geometry("geometry").to_crs(_EQUAL_AREA_CRS)

    counties = sorted({(cfg.state_fips, cf[len(cfg.state_fips):]) for cf in bg_city["county_fips"]})
    roads = load_roads_for_counties(counties, refresh=refresh)
    print(f"roadway[{city}]: {len(roads):,} TIGER road features across {len(counties)} count(y/ies)")

    ramps = roads[roads["mtfcc"] == MTFCC_RAMP]
    interstates = roads[roads["mtfcc"] == MTFCC_INTERSTATE]
    # HPMS principal + minor arterials only: interstates/freeways (f_system 1-2) are already
    # captured by nearest_interstate_m, so excluding them keeps R1 and R2 separable.
    hpms = load_hpms_for_counties(counties, refresh=refresh)
    arterials = hpms[hpms["f_system"].isin(F_SYSTEM_ARTERIAL) & hpms["facility_type"].isin(FACILITY_MAINLINE)]
    print(f"roadway[{city}]: {len(arterials):,} HPMS arterial sections (f_system 3-4)")

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

    # Street-junction density from TIGER EDGES topology (R3).
    junctions = load_junctions_for_counties(counties, refresh=refresh)
    if junctions.empty:
        out["roadway_intersection_density"] = 0.0
    else:
        hit = gpd.sjoin(bg_city[["geoid", "geometry"]], junctions[["geometry"]], predicate="intersects")
        counts = hit.groupby("geoid").size()
        density = (counts / area_km2.reindex(counts.index)).reindex(bg_city["geoid"]).fillna(0.0)
        out["roadway_intersection_density"] = out["geoid"].map(density).to_numpy()

    # Street-network morphology from node degree (R3b), over a county-line buffer.
    nodes = street_nodes(_node_counties(bg, bg_city, cfg.state_fips), refresh=refresh)
    morph = street_morphology(bg_city, nodes)
    out = out.join(morph, on="geoid")

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

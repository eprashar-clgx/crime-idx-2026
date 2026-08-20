"""Build the BG-level transit feature table: stops -> sjoin to geoid -> aggregate.

`build_transit(city)` materializes a per-city contribution; `build_all_transit()` stacks
all POC cities into data/interim/sources/transit.parquet — the parquet the `transit`
FeatureSource (backend="file") reads via `pull_source`. Building is out-of-band: nothing
in the normal pull path triggers it (see docs/adr/0002-gtfs-transit-ingestion.md).

Reuses the shared foundation for the spatial join (same pattern as crime ingestion):
`load_state_block_groups` + `label_bgs_within_city`, then sjoin stops within BG polygons.

Risky-facility points (H1/H3) come from the same firmographics CLIP source as the
`convenience_stores`/`liquor_stores` FeatureSources, but pulled as points (parcel
centroids) rather than BG counts. Those pulls need BigQuery credentials; when they are
unavailable the build still emits the supply-side features and leaves the H1/H3 columns
at zero (a printed note flags the skip).
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd

from crime_blockgroup_mapping.constants import CITIES
from crime_blockgroup_mapping.boundaries import (
    load_city_boundary, load_state_block_groups, label_bgs_within_city,
)
from regression_modelling.config import source_parquet, transit_facilities_parquet
from regression_modelling.constants import TRANSIT_FEEDS
from regression_modelling.data_wrangling.transit.feeds import load_city_stops
from regression_modelling.data_wrangling.transit.colocation import (
    RISKY_FACILITY_CATEGORIES, add_risky_flags, aggregate_stops_to_bg,
    nearest_facility_distance, _BG_FEATURE_COLS,
)

# Equal-area CRS (CONUS Albers) for BG area / density in km^2.
_EQUAL_AREA_CRS = "EPSG:5070"

# Risky co-location category -> StoreQuery name in distributions.eda.STORE_QUERIES.
_FACILITY_STORE = {"convenience": "convenience_stores", "liquor": "liquor_stores", "atm": "atm"}


def _stops_to_gdf(stops: pd.DataFrame) -> gpd.GeoDataFrame:
    """Per-stop DataFrame -> GeoDataFrame of points in EPSG:4326 (join CRS)."""
    geom = gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"])
    return gpd.GeoDataFrame(stops.copy(), geometry=geom, crs="EPSG:4326")


def load_facility_points(category: str, refresh: bool = False) -> pd.DataFrame:
    """Risky-facility point layer (lat/lon) for one category, cached to interim.

    Pulls firmographics parcel geometries via `distributions.eda.store_points` and reduces
    each parcel to its centroid. Caches to data/interim/transit/facilities/{category}.parquet
    so subsequent builds run offline.
    """
    cache = transit_facilities_parquet(category)
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    from regression_modelling.distributions.eda import store_points

    raw = store_points(_FACILITY_STORE[category])
    geom = gpd.GeoSeries.from_wkt(raw["parcel_polygon_at_eventtime"], crs="EPSG:4326")
    cent = geom.centroid
    out = pd.DataFrame({"clip_id": raw["clip_id"].values, "lat": cent.y.values, "lon": cent.x.values})
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"transit facilities[{category}]: {len(out)} points -> {cache}")
    return out


def load_risky_facilities(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """All risky-facility point layers, keyed by category. Missing/uncredentialed skip."""
    facilities: dict[str, pd.DataFrame] = {}
    for cat in RISKY_FACILITY_CATEGORIES:
        try:
            df = load_facility_points(cat, refresh=refresh)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully without BQ creds
            print(f"transit: facility '{cat}' unavailable ({type(exc).__name__}); H1/H3 skipped for it")
            continue
        if not df.empty:
            facilities[cat] = df
    return facilities


def build_transit(city: str, refresh: bool = False,
                  facilities: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """BG-level transit predictors for one city, keyed by `geoid`.

    Steps: load per-stop features (feeds) -> co-location + H3 flags (colocation) ->
    sjoin stops to BGs (foundation) -> aggregate to BG (colocation) -> add geometry-based
    density + nearest-stop distance. Every within-city BG is emitted (stopless BGs get
    zero counts but a real `transit_nearest_stop_m`). Prints matched/unmatched diagnostics.
    """
    cfg = CITIES[city]
    if facilities is None:
        facilities = load_risky_facilities(refresh=refresh)

    stops = load_city_stops(city, refresh=refresh)
    stops = add_risky_flags(stops, facilities)

    bg = load_state_block_groups(cfg)
    bg = label_bgs_within_city(bg, load_city_boundary(cfg))
    bg_city = bg[bg["within_city"]][["geoid", "geometry"]].copy()

    stops_gdf = _stops_to_gdf(stops)
    joined = gpd.sjoin(stops_gdf, bg_city, how="left", predicate="within")
    n_matched = int(joined["geoid"].notna().sum())
    print(f"transit[{city}]: stops matched to a within-city BG: {n_matched:,} / {len(stops):,}")
    matched = joined[joined["geoid"].notna()].copy()

    agg = aggregate_stops_to_bg(pd.DataFrame(matched.drop(columns="geometry")))

    out = bg_city.merge(agg, on="geoid", how="left")
    count_cols = [c for c in _BG_FEATURE_COLS if c != "transit_route_mode_diversity"]
    out[count_cols] = out[count_cols].fillna(0)
    out["transit_route_mode_diversity"] = out["transit_route_mode_diversity"].fillna(0.0)

    # Density (stops per km^2) needs equal-area geometry.
    area_km2 = out.set_geometry("geometry").to_crs(_EQUAL_AREA_CRS).area / 1e6
    out["transit_stop_density"] = np.where(area_km2 > 0, out["transit_stop_count"] / area_km2, 0.0)

    # Nearest-stop distance from each BG centroid (meaningful even for stopless BGs).
    cent = (out.set_geometry("geometry").to_crs(_EQUAL_AREA_CRS)
            .geometry.centroid.to_crs("EPSG:4326"))
    bg_centroids = pd.DataFrame({"stop_lat": cent.y.values, "stop_lon": cent.x.values},
                                index=out.index)
    if len(stops):
        out["transit_nearest_stop_m"] = nearest_facility_distance(
            bg_centroids, stops.rename(columns={"stop_lat": "lat", "stop_lon": "lon"}),
        ).values
    else:
        out["transit_nearest_stop_m"] = np.inf

    out["city"] = city
    keep = ["geoid", "city", "transit_stop_count", "transit_stop_density",
            "transit_nearest_stop_m", "transit_service_intensity",
            "transit_overnight_stop_count", "transit_overnight_stop_share",
            "transit_risky_stop_count", "transit_risky_stop_share",
            "transit_risky_allnight_count", "transit_route_mode_diversity"]
    result = pd.DataFrame(out)[keep]
    print(f"transit[{city}]: BG table {result.shape} "
          f"({int((result.transit_stop_count > 0).sum())} BGs with >=1 stop)")
    return result


def build_all_transit(refresh: bool = False) -> pd.DataFrame:
    """Stack per-city BG transit tables for all POC cities and cache the registry parquet.

    Writes data/interim/sources/transit.parquet (== source_parquet("transit")), which the
    `transit` FeatureSource reads. BGs outside the POC cities are simply absent and become
    null on the national left-join (handled by Phase-3 imputation). Facility points are
    loaded once and shared across cities.
    """
    facilities = load_risky_facilities(refresh=refresh)
    frames = [build_transit(city, refresh=refresh, facilities=facilities) for city in TRANSIT_FEEDS]
    out = pd.concat(frames, ignore_index=True)
    path = source_parquet("transit")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"transit: BG feature table {out.shape} -> {path}")
    return out

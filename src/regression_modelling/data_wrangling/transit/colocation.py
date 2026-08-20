"""Stop <-> facility co-location (H1) and the H3 interaction, plus BG-aggregation helpers.

H1: a stop within a short walk of a convenience store / liquor store / ATM is riskier.
H3: a stop that is BOTH facility-adjacent AND all-night is disproportionately risky.

Design rule (see docs/hypothesis.md §4.3 / docs/transit_eda_plan.md §4): build the
interaction at STOP level first, then aggregate to BG. Multiplying two BG-level averages
would falsely light up a BG that has a facility-adjacent daytime stop and a separate
all-night stop elsewhere.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# H1 co-location radius (meters). ~150m ≈ a short walk / same intersection.
RISKY_RADIUS_M = 150.0

# Facility categories treated as risky attractors (kept separate; may load differently).
RISKY_FACILITY_CATEGORIES = ("convenience", "liquor", "atm")


def shannon_equitability(counts) -> float:
    """Normalized Shannon diversity (equitability) index of a set of category counts.

    Returns a value in [0, 1]: 0 = single dominant category, 1 = perfectly even mix.
    Used for transit route-mode diversity (over route_type) and, elsewhere, land-use mix.
    Matches the normalized form in docs/hypothesis.md §4.1 (1-smoothing omitted here since
    we pass raw observed counts; empty/degenerate inputs return 0.0).

    >>> shannon_equitability([5, 5])
    1.0
    >>> shannon_equitability([10, 0, 0])
    0.0
    """
    c = np.asarray([x for x in counts if x is not None], dtype="float64")
    c = c[c > 0]
    n_cats = c.size
    if n_cats <= 1:
        return 0.0
    p = c / c.sum()
    h = -(p * np.log(p)).sum()
    return float(h / math.log(n_cats))


def nearest_facility_distance(stops: pd.DataFrame, facilities: pd.DataFrame,
                              lat_col: str = "stop_lat", lon_col: str = "stop_lon",
                              fac_lat_col: str = "lat", fac_lon_col: str = "lon") -> pd.Series:
    """Distance (meters) from each stop to its nearest facility point.

    Vectorized nearest-neighbor via a KD-tree over a local equirectangular projection
    (adequate at the intra-city scale that matters for a ~150m co-location test).
    Returns one value per stop row aligned to ``stops.index``; ``inf`` where the
    facility layer is empty.
    """
    from scipy.spatial import cKDTree

    idx = stops.index
    if facilities is None or facilities.empty or stops.empty:
        return pd.Series(np.inf, index=idx, dtype="float64")

    lat0 = np.radians(float(stops[lat_col].mean()))
    cos0 = np.cos(lat0)

    def _xy(lat, lon):
        lat = np.radians(np.asarray(lat, dtype="float64"))
        lon = np.radians(np.asarray(lon, dtype="float64"))
        return np.column_stack([lon * cos0 * _EARTH_M, lat * _EARTH_M])

    fac = facilities.dropna(subset=[fac_lat_col, fac_lon_col])
    if fac.empty:
        return pd.Series(np.inf, index=idx, dtype="float64")

    tree = cKDTree(_xy(fac[fac_lat_col], fac[fac_lon_col]))
    dist, _ = tree.query(_xy(stops[lat_col], stops[lon_col]), k=1)
    return pd.Series(dist, index=idx, dtype="float64")


def add_risky_flags(stops: pd.DataFrame, facilities_by_cat: dict[str, pd.DataFrame],
                    radius_m: float = RISKY_RADIUS_M) -> pd.DataFrame:
    """Add per-stop co-location columns (H1) and the H3 AND flag.

    Adds, for each category in ``facilities_by_cat``:
        dist_{cat}_m        nearest-facility distance (meters)
        near_{cat}          1 if within ``radius_m``
    plus:
        near_risky          1 if near ANY risky category
        risky_allnight      near_risky AND overnight_flag        (H3, stop-level)
    """
    out = stops.copy()
    near_cols = []
    for cat, facilities in facilities_by_cat.items():
        d = nearest_facility_distance(out, facilities)
        out[f"dist_{cat}_m"] = d
        near = (d <= radius_m).astype(int)
        out[f"near_{cat}"] = near
        near_cols.append(f"near_{cat}")

    if near_cols:
        out["near_risky"] = out[near_cols].max(axis=1).astype(int)
    else:
        out["near_risky"] = 0
    overnight = out["overnight_flag"] if "overnight_flag" in out.columns else 0
    out["risky_allnight"] = (out["near_risky"].astype(bool) & np.asarray(overnight).astype(bool)).astype(int)
    return out


def aggregate_stops_to_bg(stops: pd.DataFrame, key: str = "geoid") -> pd.DataFrame:
    """Aggregate per-stop features to BG-level transit predictors, keyed by ``geoid``.

    Produces (see docs/transit_eda_plan.md §5), per BG:
        transit_stop_count, transit_service_intensity (sum weekday trips/day),
        transit_overnight_stop_count, transit_overnight_stop_share,
        transit_risky_stop_count, transit_risky_stop_share,          (H1)
        transit_risky_allnight_count,                                (H3)
        transit_route_mode_diversity (shannon_equitability over route_type)
    Density/nearest-distance terms that need BG geometry (area, centroid) are added in
    build.build_transit where the BG layer is in scope.
    """
    if stops.empty:
        return pd.DataFrame(columns=[key] + _BG_FEATURE_COLS)

    df = stops.copy()
    for col, default in (("near_risky", 0), ("risky_allnight", 0), ("overnight_flag", 0)):
        if col not in df.columns:
            df[col] = default

    grouped = df.groupby(key)
    out = grouped.agg(
        transit_stop_count=("stop_id", "size"),
        transit_service_intensity=("n_trips_day", "sum"),
        transit_overnight_stop_count=("overnight_flag", "sum"),
        transit_overnight_stop_share=("overnight_flag", "mean"),
        transit_risky_stop_count=("near_risky", "sum"),
        transit_risky_stop_share=("near_risky", "mean"),
        transit_risky_allnight_count=("risky_allnight", "sum"),
    )
    out["transit_route_mode_diversity"] = grouped["route_types"].apply(_mode_diversity)
    return out.reset_index()


def _mode_diversity(route_type_sets) -> float:
    """Shannon equitability of route_type presence across a BG's stops."""
    counter: dict = {}
    for s in route_type_sets:
        for rt in (s or ()):
            counter[rt] = counter.get(rt, 0) + 1
    return shannon_equitability(list(counter.values()))


_EARTH_M = 6_371_000.0

_BG_FEATURE_COLS = [
    "transit_stop_count", "transit_service_intensity",
    "transit_overnight_stop_count", "transit_overnight_stop_share",
    "transit_risky_stop_count", "transit_risky_stop_share",
    "transit_risky_allnight_count", "transit_route_mode_diversity",
]

"""Read GTFS feed zip(s) into a per-stop feature table.

Per-stop features drive H2 (overnight/24-7 service) and the transit-supply context
variables. `gtfs-kit` is imported lazily so this module imports without the dependency
installed.

GTFS gotchas handled here (see docs/transit_eda_plan.md §2):
  1. `stop_times` departure/arrival times can exceed 24:00:00 (e.g. 25:30:00 = 1:30am
     next service day) — parsed as seconds-since-service-midnight, never dropped. This
     is exactly how overnight service is detected (a stop whose service span starts
     before 05:00 or ends at/after 24:00 has overnight service).
  2. `parent_station` — child platforms are collapsed to their station so a rail station
     is not counted as many stops.
  3. `location_type` — only boardable stops (0 / blank) are kept; stations (1) and
     entrances (2) are dropped from the point set.
  4. A single representative service date is used (not the union of all service_ids),
     so trips/day is not overcounted. gtfs-kit's `compute_stop_stats` respects
     `frequencies.txt` (headway-based feeds such as CTA and PRT) when counting trips.
"""
from __future__ import annotations

import datetime as _dt
import pandas as pd
import gtfs_kit as gk
import numpy as np
from scipy.spatial import cKDTree

from regression_modelling.config import transit_feed_zip, transit_stops_parquet
from regression_modelling.constants import TRANSIT_FEEDS, TRANSIT_REPRESENTATIVE_DATE

# Overnight exposure window (local service time), inclusive start / exclusive end.
OVERNIGHT_START_H = 0
OVERNIGHT_END_H = 5

# Two stops within this distance are treated as the same physical station when
# deduping across operators (e.g. BART + Muni share Embarcadero/Civic Center/Powell).
SHARED_STATION_M = 120.0

_SECONDS_PER_DAY = 24 * 3600


def gtfs_seconds(time_str: str) -> float | None:
    """Parse a GTFS `HH:MM:SS` time to seconds since service midnight.

    Handles hours >= 24 (service that runs past midnight). Returns None for blank/NaN.

    >>> gtfs_seconds("25:30:00")
    91800.0
    >>> gtfs_seconds("00:15:00")
    900.0
    """
    if time_str is None:
        return None
    s = str(time_str).strip()
    if not s or s.lower() == "nan":
        return None
    parts = s.split(":")
    if len(parts) != 3:
        return None
    h, m, sec = (int(parts[0]), int(parts[1]), int(parts[2]))
    return float(h * 3600 + m * 60 + sec)


def is_overnight_seconds(sec: float | None,
                         start_h: int = OVERNIGHT_START_H,
                         end_h: int = OVERNIGHT_END_H) -> bool:
    """True if a seconds-since-service-midnight time falls in the overnight window.

    Times past 24:00 wrap to the next day, so 25:30 (91800s) -> 01:30 -> overnight.
    """
    if sec is None:
        return False
    hour_of_day = (sec / 3600.0) % 24.0
    return start_h <= hour_of_day < end_h


def _resolve_service_date(feed, service_date: str | None) -> str:
    """Pick a representative full-service weekday (GTFS ``YYYYMMDD``).

    Honors an explicit in-feed ``service_date``; otherwise scores candidate weekdays by
    the number of active trips and returns a peak-service weekday nearest the target
    anchor (``TRANSIT_REPRESENTATIVE_DATE``). Scoring by trip volume avoids near-empty
    special-service dates (e.g. a holiday exception in ``calendar_dates``) that are
    technically "active" but run almost no service — a real trap: MARTA's full weekday
    service is confined to a narrow date window, with only a handful of special trips
    elsewhere in the feed range.
    """
    dates = feed.get_dates()
    if not dates:
        raise ValueError("feed has no active service dates")
    if service_date:
        explicit = service_date.replace("-", "")
        if explicit in dates:
            return explicit

    target_dt = _dt.datetime.strptime(TRANSIT_REPRESENTATIVE_DATE.replace("-", ""), "%Y%m%d")
    weekdays = [d for d in dates if _dt.datetime.strptime(d, "%Y%m%d").weekday() < 5]
    pool = weekdays or dates
    counts = {d: len(feed.get_trips(d)) for d in pool}
    peak = max(counts.values())
    if peak == 0:  # degenerate feed with no calendar-based trips; fall back to proximity
        return min(pool, key=lambda d: abs(_dt.datetime.strptime(d, "%Y%m%d") - target_dt))
    strong = [d for d, c in counts.items() if c >= 0.8 * peak]
    return min(strong, key=lambda d: abs(_dt.datetime.strptime(d, "%Y%m%d") - target_dt))


def _route_types_by_stop(feed) -> pd.Series:
    """frozenset of GTFS ``route_type`` codes serving each ``stop_id``."""
    st = feed.stop_times[["trip_id", "stop_id"]]
    tr = feed.trips[["trip_id", "route_id"]]
    rt = feed.routes[["route_id", "route_type"]]
    merged = st.merge(tr, on="trip_id").merge(rt, on="route_id")
    return merged.groupby("stop_id")["route_type"].agg(
        lambda s: frozenset(int(x) for x in s)
    )


def _clean_feed_ids(feed):
    """Strip stray whitespace from GTFS id columns (defensive against malformed feeds).

    Some agencies ship ids padded with spaces (e.g. SacRT's ``calendar.service_id`` is
    ``' 1'`` while ``trips.service_id`` is ``'1'``), which silently breaks gtfs-kit's
    calendar<->trips join and yields *zero* active service. Trimming the join-relevant id
    columns in place fixes it without touching feature logic.
    """
    id_cols = {
        "calendar": ["service_id"],
        "calendar_dates": ["service_id"],
        "trips": ["service_id", "trip_id", "route_id"],
        "stop_times": ["trip_id", "stop_id"],
        "stops": ["stop_id", "parent_station"],
        "routes": ["route_id"],
        "frequencies": ["trip_id"],
    }
    for table, cols in id_cols.items():
        df = getattr(feed, table, None)
        if df is None:
            continue
        for col in cols:
            if col in df.columns and (df[col].dtype == object
                                      or str(df[col].dtype).startswith("string")):
                df[col] = df[col].str.strip()
        setattr(feed, table, df)
    return feed


def read_stop_features(zip_path, service_date: str | None = None) -> pd.DataFrame:
    """Per-stop features for one GTFS zip on a representative service date.

    Returns one row per boardable stop (or parent station) with:
        stop_id, stop_lat, stop_lon,
        n_trips_day, n_routes, first_dep_s, last_dep_s, span_hours,
        overnight_flag, route_types (frozenset), service_date

    Uses gtfs-kit's ``compute_stop_stats`` for trip/headway stats (respects
    ``frequencies.txt``); overnight service is inferred from the service span
    (gotcha 1). Child platforms are collapsed onto ``parent_station`` (gotcha 2) and
    only boardable stops are kept (gotcha 3).
    """
    

    feed = _clean_feed_ids(gk.read_feed(str(zip_path), dist_units="km"))
    date = _resolve_service_date(feed, service_date)

    ss = feed.compute_stop_stats([date]).copy()
    ss["first_dep_s"] = ss["start_time"].map(gtfs_seconds)
    ss["last_dep_s"] = ss["end_time"].map(gtfs_seconds)
    ss["span_hours"] = (ss["last_dep_s"] - ss["first_dep_s"]) / 3600.0
    ss["overnight_flag"] = (
        (ss["first_dep_s"] < OVERNIGHT_END_H * 3600)
        | (ss["last_dep_s"] >= _SECONDS_PER_DAY)
    ).astype(int)
    ss = ss.rename(columns={"num_trips": "n_trips_day", "num_routes": "n_routes"})

    stops = feed.stops.copy()
    if "location_type" in stops.columns:
        lt = pd.to_numeric(stops["location_type"], errors="coerce").fillna(0)
        stops = stops[lt == 0]
    keep = ["stop_id", "stop_lat", "stop_lon"]
    if "parent_station" in stops.columns:
        keep.append("parent_station")
    stops = stops[keep].copy()
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")

    types = _route_types_by_stop(feed)
    df = stops.merge(
        ss[["stop_id", "n_trips_day", "n_routes", "first_dep_s",
            "last_dep_s", "span_hours", "overnight_flag"]],
        on="stop_id", how="inner",
    )
    df["route_types"] = df["stop_id"].map(types).fillna("").map(
        lambda x: x if isinstance(x, frozenset) else frozenset()
    )
    df = _collapse_parent_stations(df)
    df = df.dropna(subset=["stop_lat", "stop_lon"])
    df["service_date"] = date
    print(
        f"  {getattr(zip_path, 'name', zip_path)}: date={date} "
        f"stops={len(df)} overnight={int(df['overnight_flag'].sum())}"
    )
    return df.reset_index(drop=True)


def _collapse_parent_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse child platforms onto ``parent_station`` (gotcha 2).

    Rail stations expose one boardable ``stop_id`` per platform sharing a
    ``parent_station``; counting each as a stop over-inflates rail. Rows without a
    parent are kept as-is. Aggregation: station centroid (mean lat/lon), summed
    trips, unioned routes/route_types, any-platform overnight, widest span.
    """
    if "parent_station" not in df.columns:
        return df
    parent = df["parent_station"].astype("string").str.strip()
    has_parent = parent.notna() & (parent != "") & (parent.str.lower() != "nan")
    singles = df[~has_parent].drop(columns=["parent_station"])
    grouped = df[has_parent].copy()
    if grouped.empty:
        return singles.reset_index(drop=True)
    grouped["stop_id"] = parent[has_parent].values

    def _agg(g: pd.DataFrame) -> pd.Series:
        route_types = frozenset().union(*g["route_types"]) if len(g) else frozenset()
        return pd.Series({
            "stop_lat": g["stop_lat"].mean(),
            "stop_lon": g["stop_lon"].mean(),
            "n_trips_day": g["n_trips_day"].sum(),
            "n_routes": len(route_types),
            "first_dep_s": g["first_dep_s"].min(),
            "last_dep_s": g["last_dep_s"].max(),
            "span_hours": (g["last_dep_s"].max() - g["first_dep_s"].min()) / 3600.0,
            "overnight_flag": int(g["overnight_flag"].max()),
            "route_types": route_types,
        })

    collapsed = grouped.groupby("stop_id", as_index=False).apply(_agg)
    return pd.concat([singles, collapsed], ignore_index=True)


def load_city_stops(city: str, refresh: bool = False) -> pd.DataFrame:
    """Per-stop feature table for a city, unioning all its feeds (e.g. SF = Muni + BART).

    Reads each feed in ``TRANSIT_FEEDS[city]`` via ``read_stop_features``, tags the
    operating ``agency``, concatenates, and dedups physically shared stations across
    operators by proximity (``SHARED_STATION_M``) so they are not double-counted.
    Caches to ``data/interim/transit/stops/{city}.parquet`` (the pipeline ``refresh``
    pattern). ``route_types`` is stored as a sorted list for parquet round-tripping.
    """
    cache = transit_stops_parquet(city)
    if cache.exists() and not refresh:
        out = pd.read_parquet(cache)
        out["route_types"] = out["route_types"].map(
            lambda x: frozenset(int(v) for v in x)
        )
        return out

    feeds = TRANSIT_FEEDS.get(city)
    if not feeds:
        raise KeyError(f"no transit feeds registered for city {city!r}")

    print(f"transit[{city}]: reading {len(feeds)} feed(s)")
    frames = []
    for feed in feeds:
        zip_path = transit_feed_zip(city, feed.feed_id)
        part = read_stop_features(zip_path)
        part.insert(0, "agency", feed.agency)
        frames.append(part)
    stops = pd.concat(frames, ignore_index=True)

    if len(feeds) > 1:
        stops = _dedup_shared_stations(stops)

    out = stops.copy()
    out["route_types"] = out["route_types"].map(lambda s: sorted(int(v) for v in s))
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"transit[{city}]: {len(stops)} stops -> {cache}")
    return stops


def _dedup_shared_stations(stops: pd.DataFrame) -> pd.DataFrame:
    """Drop cross-operator duplicate stations that sit within ``SHARED_STATION_M``.

    Keeps the higher-service row (more trips/day) of each shared pair and unions the
    route_types so mode diversity is preserved. Approximate metric distance via a local
    equirectangular projection — adequate at ~100m scales.
    """

    df = stops.reset_index(drop=True)
    lat0 = np.radians(df["stop_lat"].mean())
    x = np.radians(df["stop_lon"].to_numpy()) * np.cos(lat0) * 6_371_000.0
    y = np.radians(df["stop_lat"].to_numpy()) * 6_371_000.0
    pts = np.column_stack([x, y])

    tree = cKDTree(pts)
    pairs = tree.query_pairs(SHARED_STATION_M)
    drop: set[int] = set()
    for i, j in pairs:
        if df.at[i, "agency"] == df.at[j, "agency"]:
            continue
        keep, other = (i, j) if df.at[i, "n_trips_day"] >= df.at[j, "n_trips_day"] else (j, i)
        if keep in drop:
            keep, other = other, keep
        df.at[keep, "route_types"] = df.at[keep, "route_types"] | df.at[other, "route_types"]
        df.at[keep, "n_routes"] = len(df.at[keep, "route_types"])
        drop.add(other)
    if drop:
        print(f"  deduped {len(drop)} shared cross-operator station(s)")
    return df.drop(index=list(drop)).reset_index(drop=True)

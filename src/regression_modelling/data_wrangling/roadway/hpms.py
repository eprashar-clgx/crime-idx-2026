"""Load FHWA HPMS road sections (functional class + AADT) for one county.

HPMS is the federal Highway Performance Monitoring System: state DOTs report every
Federal-aid road section with its FHWA functional classification (`f_system`) and traffic
counts (`aadt`). It's the authoritative definition of "arterial" — TIGER's S1200 only covers
numbered highways, so most big-city arterials sit unlabeled inside S1400. US federal
publication, no licence restrictions stated (see docs/features/roadway_plan.md §5).

Pulled per county from FHWA's public ArcGIS FeatureServer (`HPMS_FULL_{ST}_{year}`),
filtered server-side so only the sections we use cross the wire:
  f_system 1-4      1 Interstate, 2 Other freeway/expressway, 3 Other principal arterial,
                    4 Minor arterial (collectors 5-6 and locals 7 are not pulled)
  facility_type 1-2 one-way / two-way mainline. Type 6 ("non-inventory direction") is the
                    mirrored opposite carriageway of a divided road — keeping it would
                    double-count length; 5/7 (non-mainline, planned) aren't roads we measure.
  facility_type 4   ramps (any f_system; in practice 1-3, the class of the road served).
                    Several states (IL, TX, OH, MO, WA) leave county_id null on ramps, so
                    ramps are pulled spatially: the county's mainline bbox + ~2 km.
Every section carries AADT (total, single-unit truck, combination truck) and lane counts.
Pages through the 2000-record server limit. Cached raw (EPSG:4326, as received) to
data/raw/roadway/hpms/{ST}_{county}_{year}.parquet.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import geopandas as gpd
import pandas as pd

from regression_modelling.config import hpms_county_parquet

HPMS_YEAR = 2024
HPMS_URL = "https://geo.dot.gov/server/rest/services/Hosted/HPMS_FULL_{st}_{year}/FeatureServer/0/query"
F_SYSTEM_PULL = (1, 2, 3, 4)
F_SYSTEM_ARTERIAL = (3, 4)
F_SYSTEM_INTERSTATE = (1,)
FACILITY_MAINLINE = (1, 2)
FACILITY_RAMP = 4
_FIELDS = ("objectid", "route_id", "f_system", "facility_type", "aadt", "aadt_single_unit",
           "aadt_combination", "through_lanes", "access_control", "speed_limit", "nhs",
           "county_id")
_PAGE = 2000
_RAMP_BBOX_PAD_DEG = 0.02

_EQUAL_AREA_CRS = "EPSG:5070"

# State FIPS -> USPS code (HPMS service names are keyed by postal code).
STATE_USPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
    "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
    "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
    "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}


def _query_page(url: str, where: str, offset: int, bbox: tuple | None = None,
                retries: int = 3) -> dict:
    q = {"where": where, "outFields": ",".join(_FIELDS), "outSR": 4326, "f": "geojson",
         "orderByFields": "objectid", "resultOffset": offset, "resultRecordCount": _PAGE}
    if bbox is not None:
        q.update(geometry=",".join(map(str, bbox)), geometryType="esriGeometryEnvelope",
                 inSR=4326, spatialRel="esriSpatialRelIntersects")
    params = urllib.parse.urlencode(q)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{params}", timeout=120) as resp:
                return json.load(resp)
        except Exception:  # noqa: BLE001 - transient server/network errors; retry then raise
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _query_all(url: str, where: str, bbox: tuple | None = None) -> gpd.GeoDataFrame:
    features, offset = [], 0
    while True:
        page = _query_page(url, where, offset, bbox)
        if "error" in page:
            raise RuntimeError(f"HPMS query failed ({url}, {where}): {page['error']}")
        batch = page.get("features", [])
        features.extend(batch)
        if len(batch) < _PAGE:
            break
        offset += _PAGE
    if not features:
        return gpd.GeoDataFrame(columns=[*_FIELDS, "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")[[*_FIELDS, "geometry"]]


def _pull_county(state_usps: str, county_fips: str, year: int) -> gpd.GeoDataFrame:
    url = HPMS_URL.format(st=state_usps, year=year)
    mainline = _query_all(url, (
        f"county_id={int(county_fips)} AND f_system IN ({','.join(map(str, F_SYSTEM_PULL))})"
        f" AND facility_type IN ({','.join(map(str, FACILITY_MAINLINE))})"))
    if mainline.empty:
        return mainline
    x0, y0, x1, y1 = mainline.total_bounds
    pad = _RAMP_BBOX_PAD_DEG
    ramps = _query_all(url, f"facility_type={FACILITY_RAMP}", (x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return gpd.GeoDataFrame(pd.concat([mainline, ramps], ignore_index=True), crs="EPSG:4326")


def load_county_hpms(state_fips: str, county_fips: str, year: int = HPMS_YEAR,
                     refresh: bool = False) -> gpd.GeoDataFrame:
    """HPMS mainline sections (f_system 1-4) + ramps for one county, in EPSG:5070."""
    st = STATE_USPS[state_fips]
    cache = hpms_county_parquet(st, county_fips, year)
    raw = gpd.read_parquet(cache) if cache.exists() and not refresh else None
    # Caches written before ramps/truck AADT were pulled lack those rows/columns -> re-pull.
    if raw is not None and not set(_FIELDS) <= set(raw.columns):
        raw = None
    if raw is None:
        print(f"roadway[hpms]: querying HPMS_FULL_{st}_{year} county {county_fips}")
        raw = _pull_county(st, county_fips, year)
        cache.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(cache)
        print(f"roadway[hpms]: {st} {county_fips} -> {len(raw):,} sections "
              f"(f_system 1-4 mainline + ramps) -> {cache}")
    return raw.to_crs(_EQUAL_AREA_CRS)


def load_hpms_for_counties(counties: list[tuple[str, str]], year: int = HPMS_YEAR,
                           refresh: bool = False) -> gpd.GeoDataFrame:
    """Concat HPMS sections across every (state_fips, county_fips) pair a city touches."""
    frames = [load_county_hpms(s, c, year, refresh=refresh) for s, c in counties]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return gpd.GeoDataFrame(columns=[*_FIELDS, "geometry"], geometry="geometry", crs=_EQUAL_AREA_CRS)
    # Ramps come from padded bboxes, so neighbouring counties overlap -> dedupe on objectid.
    out = pd.concat(frames, ignore_index=True).drop_duplicates("objectid")
    return gpd.GeoDataFrame(out, crs=_EQUAL_AREA_CRS).reset_index(drop=True)

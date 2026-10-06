"""Build the BG-level neighbourhood-context table: neighbour averages + store densities.

`build_neighbourhood()` writes data/interim/sources/neighbourhood.parquet, the parquet the
`neighbourhood` FeatureSource (backend="file") reads via `pull_source`. National coverage:
it runs on every BG in the 2025 cartographic-boundary layer, not just the model cities.
Build it AFTER the demographic and store sources are cached, because it reads them.

Neighbours = queen contiguity: BGs whose polygons share an edge or a vertex, self excluded.
The national layer is used so neighbours across state lines count (e.g. Kansas City MO/KS,
DC/MD/VA, NYC/NJ). BGs with no contiguous neighbour (islands) get NaN neighbour means.

    {col}_nbr (shares)  unweighted mean of the neighbours' raw values, ignoring NaN
    {col}_nbr (stores)  sum of the neighbours' store counts (NaN counted as 0)
    {col}_per_km2       the BG's own store count / land area (ALAND), area floored at
                        MIN_AREA_KM2 so tiny BGs don't explode the ratio
    bg_area_km2         land area (ALAND / 1e6), unfloored

Features that already carry a spatial summary are not re-averaged: the property-distress
shares (`*_lag6`, docs/features/property_distress.md §2) and the population rings.
Transit/roadway/imagery cover in-city BGs only, so their neighbour means would be biased at
the city edge. See docs/features/neighbourhood_context.md.
"""
from __future__ import annotations

import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal.weights import Queen

from crime_blockgroup_mapping.config import BOUNDARIES_DIR
from regression_modelling.config import source_parquet
from regression_modelling.constants import (
    DENSITY_SUFFIX, FEATURE_SOURCES, MIN_AREA_KM2, NBR_SUFFIX,
    NEIGHBOUR_MEAN_INPUTS, NEIGHBOUR_SUM_INPUTS, STORE_DENSITY_INPUTS,
)
from regression_modelling.data_wrangling.sources import pull_source
from regression_modelling.data_wrangling.features import build_demographic_features

BG_YEAR = 2025
BG_URL = "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_bg_500k.zip"
_STORE_SOURCES = ("convenience_stores", "gas_stations", "liquor_stores")


def national_bg_zip(year: int = BG_YEAR):
    """Immutable national cartographic-boundary BG zip (data/raw/boundaries/)."""
    return BOUNDARIES_DIR / f"cb_{year}_us_bg_500k.zip"


def load_national_block_groups(year: int = BG_YEAR) -> gpd.GeoDataFrame:
    """All US BGs (`geoid`, `aland`, `geometry`), downloaded once, in the zip's native CRS."""
    path = national_bg_zip(year)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = BG_URL.format(year=year)
        print(f"neighbourhood: downloading {url}")
        urllib.request.urlretrieve(url, path)
    bg = gpd.read_file(f"zip://{path}", columns=["GEOID", "ALAND"])
    return bg.rename(columns={"GEOID": "geoid", "ALAND": "aland"})


def neighbour_aggregate(W, values: pd.DataFrame, how: str) -> pd.DataFrame:
    """Aggregate each column of `values` over the neighbours encoded in sparse `W`.

    W is an n×n 0/1 adjacency matrix aligned with `values` rows (self excluded).
    how="mean": NaN-aware mean (NaN neighbours are skipped; no valid neighbour -> NaN).
    how="sum":  sum with NaN treated as 0.
    """
    x = values.to_numpy(dtype=float)
    filled = np.nan_to_num(x, nan=0.0)
    total = W @ filled
    if how == "sum":
        out = total
    elif how == "mean":
        n_valid = W @ (~np.isnan(x)).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(n_valid > 0, total / n_valid, np.nan)
    else:
        raise ValueError(f"neighbour_aggregate: unknown how={how!r}")
    return pd.DataFrame(out, index=values.index,
                        columns=[f"{c}{NBR_SUFFIX}" for c in values.columns])


def _raw_inputs(refresh: bool) -> pd.DataFrame:
    """Raw (pre-imputation) BG inputs keyed by geoid: demographic shares + store counts."""
    df = build_demographic_features(refresh=refresh)[["geoid", *NEIGHBOUR_MEAN_INPUTS]]
    for name in _STORE_SOURCES:
        src = FEATURE_SOURCES[name]
        s = pull_source(src, refresh=refresh).rename(columns={src.key_col: "geoid"})
        df = df.merge(s[["geoid", *src.feature_cols]], on="geoid", how="outer")
    return df


def build_neighbourhood(refresh: bool = False) -> pd.DataFrame:
    """National neighbourhood-context table keyed by geoid; caches the registry parquet."""
    bg = load_national_block_groups()
    bg = bg.merge(_raw_inputs(refresh), on="geoid", how="left").reset_index(drop=True)

    w = Queen.from_dataframe(bg, use_index=False, silence_warnings=True)
    W = w.sparse.tocsr()
    print(f"neighbourhood: {len(bg):,} BGs, mean {w.mean_neighbors:.1f} queen neighbours, "
          f"{len(w.islands):,} islands")

    out = bg[["geoid"]].copy()
    out["bg_area_km2"] = bg["aland"] / 1e6
    out = out.join(neighbour_aggregate(W, bg[list(NEIGHBOUR_MEAN_INPUTS)], "mean"))
    out = out.join(neighbour_aggregate(W, bg[list(NEIGHBOUR_SUM_INPUTS)], "sum"))
    area = out["bg_area_km2"].clip(lower=MIN_AREA_KM2)
    for c in STORE_DENSITY_INPUTS:
        out[f"{c}{DENSITY_SUFFIX}"] = bg[c].fillna(0) / area
    out["n_neighbours"] = np.asarray(W.sum(axis=1)).ravel().astype(int)

    path = source_parquet("neighbourhood")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    print(f"neighbourhood: table {out.shape} -> {path}")
    return out

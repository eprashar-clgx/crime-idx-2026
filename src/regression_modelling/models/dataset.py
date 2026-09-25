"""Rows, target and eligibility — everything that answers "what are we modelling?",
with no reference to how a model is fit (ADR 0008).

Owns three things:

1. **City eligibility** — `target_pool` is the single owner of which cities a target is
   modelled on (capability + observed usability).
2. **The pooled frame** — per-city cached tables concatenated and filtered.
3. **The target** — `make_target`, plus the per-city demean used by the within estimator.

Why the target lives here rather than inside the fit: `make_target` is *groupwise by
city*, so a city's target depends only on its own rows. Dropping other cities from the
frame cannot change it — verified to the last bit (max |precomputed - per-fold| = 0.0 for
both `lograte` and `lograte_within_city`). It is therefore fold-INVARIANT and safe to
compute once, up front.

The contrast worth remembering: the predictor **scaler** is NOT fold-invariant. It pools
mean/sd across train cities, so computing it here would let a held-out city standardize
itself and inflate every LOCO number. The scaler stays fold-local, inside the estimator
(see `training.py`).
"""
from typing import Iterator

import numpy as np
import pandas as pd

from crime_blockgroup_mapping.config import PROCESSED_DIR
from crime_blockgroup_mapping.constants import CITIES
from regression_modelling.data_wrangling.dataset import build_model_table

TARGET_MODES = ("rate_within_city", "rate", "rate_daytime_within_city",
                "rate_daytime", "logcount", "lograte", "lograte_within_city")

_DAYTIME_MODES = {"rate_daytime", "rate_daytime_within_city"}
_LOGRATE_MODES = {"lograte", "lograte_within_city"}

#: Column `make_target` writes into when a mode is attached to a pooled frame.
TARGET_COL = "_y"


# =========================================================================== #
# City eligibility                                                            #
# =========================================================================== #

def full_coverage_cities() -> list[str]:
    """POC cities whose crime source geolocates violent crime (incl. rape).

    Derived from `CityConfig.property_only` so it can never drift from the per-city
    config: property-only cities are excluded.

    This is **source capability only** — it does not check whether a city's extract
    actually produced crime. Prefer `target_pool`, which applies both tests.
    """
    return [c for c, cfg in CITIES.items() if not cfg.property_only]


def property_cities() -> list[str]:
    """Cities usable for the property target — every POC source carries property crime.

    Exists as the counterpart to `full_coverage_cities` so neither pool is spelled out
    inline as we scale past the original five (ADR 0008).
    """
    return list(CITIES.keys())


def _capable_cities(target: str) -> list[str]:
    """Cities whose crime source *can* support `target` (source capability).

    `wtotal` needs geolocated violent crime, so `property_only` sources are excluded.
    `wprop` only needs property crime, which every POC source carries.
    """
    if target == "wtotal":
        return full_coverage_cities()
    if target == "wprop":
        return property_cities()
    raise ValueError(f"unknown target {target!r} (expected 'wtotal' or 'wprop')")


def target_pool(target: str, refresh: bool = False, verbose: bool = True) -> list[str]:
    """The cities `target` is actually modelled on — the one owner of city eligibility.

    Two independent tests, per ADR 0008:

    1. **Source capability** — can this city's crime feed support the target at all?
       Declared up front in `CityConfig.property_only`, never re-derived from data.
    2. **Observed usability** — did the extract actually produce crime? A city whose
       counts are entirely zero (BG geometry present, crime never joined) yields a
       constant rate, which makes correlation, ranking and level metrics undefined and
       poisons pooled LEVEL metrics. Columbus is the known case.

    Every exclusion is logged with its reason, because at 15-20 cities a silently
    shrinking pool is the kind of thing that costs a day to notice.
    """
    capable = _capable_cities(target)
    rate_col = f"{target}_rate"

    usable, degenerate, missing = [], [], []
    for city in capable:
        try:
            df = load_city_table(city, refresh=refresh)
        except FileNotFoundError:
            missing.append(city)
            continue
        if rate_col not in df.columns or float(df[rate_col].abs().sum()) == 0:
            degenerate.append(city)
            continue
        usable.append(city)

    if verbose:
        skipped = [c for c in CITIES if c not in capable]
        print(f"target_pool({target!r}): {len(usable)} cities")
        if skipped:
            print(f"  excluded {len(skipped)} (source lacks violent-crime coords): "
                  f"{', '.join(skipped)}")
        if degenerate:
            print(f"  excluded {len(degenerate)} (all-zero crime extract, upstream gap): "
                  f"{', '.join(degenerate)}")
        if missing:
            print(f"  excluded {len(missing)} (no cached model table): "
                  f"{', '.join(missing)}")

    return usable


# =========================================================================== #
# The pooled frame                                                            #
# =========================================================================== #

def _model_table_path(city: str):
    return PROCESSED_DIR / "regression_modelling" / f"{city}_model_table.parquet"


def load_city_table(city: str, refresh: bool = False) -> pd.DataFrame:
    """Load one city's cached model table (modeling reloads locally, per the split-
    ingestion-from-modeling convention). Rebuilds via `build_model_table` only when the
    cached parquet is missing or `refresh=True`."""
    path = _model_table_path(city)
    if refresh or not path.exists():
        return build_model_table(city, refresh=refresh)
    return pd.read_parquet(path)


def load_pooled_table(cities: list[str] | None = None, drop_zero_pop: bool = True,
                      daytime_pop_floor: float = 100.0,
                      refresh: bool = False) -> pd.DataFrame:
    """Concatenate the per-city model tables into one pooled BG frame for LOCO CV.

    - Tags every row with a `city` column (the fold grouping key).
    - Drops zero/NaN-population BGs when `drop_zero_pop` (the rate target needs a
      denominator; ADR 0003). The surviving `geoid` set is what spatial weights and the
      bias table must later align to.
    - Applies a `daytime_pop_floor` (default 100): BGs whose daytime population
      (residents + LODES jobs) is below the floor are small-denominator ARTIFACTS whose
      rate is measurement noise (a few crimes over ~no people) — they manufacture the
      skew that crushes the raw-rate OLS. Dropping them is the diagnosed target-
      stabilization treatment; genuine hotspots (hundreds of crimes over a solid daytime
      population) sit far above the floor and are preserved. Set 0 to disable.

    Prints per-city and pooled row counts so the fold sizes are visible.
    """
    cities = cities or full_coverage_cities()
    frames = []
    for c in cities:
        df = load_city_table(c, refresh=refresh).copy()
        df.insert(0, "city", c)
        frames.append(df)
        print(f"  {c:16} {df.shape[0]:>6} BGs")

    pooled = pd.concat(frames, ignore_index=True)
    print(f"  {'pooled':16} {pooled.shape[0]:>6} BGs across {len(cities)} cities")

    if drop_zero_pop:
        before = len(pooled)
        pooled = pooled[pooled["population"].notna() & (pooled["population"] > 0)].copy()
        dropped = before - len(pooled)
        print(f"  dropped {dropped} zero/NaN-pop BGs -> {len(pooled)} remain "
              f"(filtered geoid set for fit + Moran's I + bias join)")

    if daytime_pop_floor and "daytime_pop" in pooled.columns:
        before = len(pooled)
        pooled = pooled[pooled["daytime_pop"] >= daytime_pop_floor].copy()
        dropped = before - len(pooled)
        print(f"  dropped {dropped} BGs below daytime_pop {daytime_pop_floor:.0f} "
              f"(small-denominator rate artifacts) -> {len(pooled)} remain")

    return pooled


def load_pool(target: str, mode: str | None = None, cities: list[str] | None = None,
              winsor_upper: float | None = None, refresh: bool = False,
              verbose: bool = True, **kwargs) -> pd.DataFrame:
    """One call for the standard setup: resolve the pool, load it, attach the target.

    `cities` defaults to `target_pool(target)`, so callers cannot drift from the
    eligibility rule by accident. When `mode` is given the target is attached as the
    `_y` column (safe because `make_target` is fold-invariant — see module docstring);
    otherwise the frame comes back mode-agnostic and reusable.

    `make_target` stays public so notebooks can still build one-off target variants on
    the same frame.
    """
    cities = cities or target_pool(target, refresh=refresh, verbose=verbose)
    pooled = load_pooled_table(cities, refresh=refresh, **kwargs)
    if mode is not None:
        pooled[TARGET_COL] = make_target(pooled, mode=mode, category=target,
                                         winsor_upper=winsor_upper)
    return pooled

# =========================================================================== #
# The target                                                                  #
# =========================================================================== #

def _rate_col(mode: str, category: str) -> str:
    """The actual-rate column a mode is built on: daytime denominator (population +
    LODES jobs) for the `*_daytime` modes, plain population rate otherwise."""
    suffix = "_rate_daytime" if mode in _DAYTIME_MODES else "_rate"
    return f"{category}{suffix}"


def make_target(df: pd.DataFrame, mode: str = "rate_within_city",
                category: str = "cl_total", city_col: str = "city",
                winsor_upper: float | None = None) -> pd.Series:
    """Build the regression target `y` for a fit mode (Series aligned to `df.index`).

    - ``rate_within_city`` / ``rate_daytime_within_city`` — per-city z-scored rate
      (plain / daytime denominator), each city by ITS OWN mean/sd. Learns *relative*
      within-city BG risk (city level+scale removed).
    - ``rate`` / ``rate_daytime`` — raw rate (absolute level, one intercept).
      ``rate_daytime`` (per 1k of population + jobs) is the preferred interpretable
      target — daytime denominator + the loader's daytime_pop floor already tame the
      small-denominator skew.
    - ``logcount`` — comparator: `{category}_logcount` = log(count+1).
    - ``lograte`` — PRODUCT TARGET (pooled): `log1p({category}_rate)`, the absolute log
      rate on one intercept. With `category="wtotal"` this is `log(weighted total rate)`,
      the single headline modeling target (ADR 0003/0005). Winsorize acts on the rate
      BEFORE the log, same as the rate modes.
    - ``lograte_within_city`` — PRODUCT TARGET (LOCO within-city): the same log rate
      z-scored per city, so the fit learns *relative* within-city risk with the
      cross-city level+scale removed.

    **Fold-invariant, but not row-set-invariant.** Every branch is either pointwise or
    grouped by `city_col`, so a city's target depends only on its own rows: restricting
    `df` to a train fold leaves the surviving rows bit-identical. That is why
    `load_pool` may attach it once up front rather than recomputing it per fold.

    The `*_within_city` modes are nevertheless sensitive to *which rows of a city* are
    present, because the z-score moments are taken over exactly those rows. Build the
    target BEFORE dropping rows with null predictors, not after — `training.run_cv`
    does. (The retired `experiments._run_gbm` did the opposite, so a `within_city` spec
    silently produced a different target for GBM than for ridge; no reported number was
    affected, because no live notebook ran that combination.)

    `winsor_upper`, when set, caps the actual rate at that absolute value BEFORE any
    within-city standardization — a gentle top-tail winsorize for whatever the daytime
    floor leaves behind. A fixed constant (not a data percentile) keeps it leakage-free.
    """
    if mode == "logcount":
        return df[f"{category}_logcount"]
    if mode not in TARGET_MODES:
        raise ValueError(f"unknown target mode {mode!r}; expected one of {TARGET_MODES}")

    if mode in _LOGRATE_MODES:
        rate = df[f"{category}_rate"].astype(float)
        if winsor_upper is not None:
            rate = rate.clip(upper=winsor_upper)
        val = np.log1p(rate)
    else:
        val = df[_rate_col(mode, category)].astype(float)
        if winsor_upper is not None:
            val = val.clip(upper=winsor_upper)
    if mode.endswith("_within_city"):
        g = val.groupby(df[city_col])
        sd = g.transform("std").replace(0, np.nan)
        return (val - g.transform("mean")) / sd
    return val


def demean_by_city(df: pd.DataFrame, predictors, city_col: str = "city") -> pd.DataFrame:
    """Subtract each row's OWN city mean from every predictor (the within transform).

    Pair with a `*_within_city` target so both sides of the regression are per-city
    demeaned — the textbook "within"/fixed-effects estimator via Frisch-Waugh,
    implemented so it survives test time.

    **Leakage-safe even for a held-out city**, and therefore fold-invariant like the
    target: it uses the city's own OBSERVED predictors (X is not the outcome), so at
    predict time the held-out city is demeaned by its own X-means. No city dummy is
    needed and no outcome information leaks.
    """
    block = df[list(predictors)]
    return block - block.groupby(df[city_col].to_numpy()).transform("mean")


# =========================================================================== #
# Predictor sets                                                              #
# =========================================================================== #

def predictor_set(name: str, category: str, transit: str = "gtfs") -> list[str]:
    """Materialize a named predictor set into a column list.

    `name` is one of `PREDICTOR_SETS` ("ours" | "ours+approved"), optionally suffixed
    "+agency" to append the target-paired agency anchor column — the LEVEL feature that
    carries cross-city calibration.

    `transit` selects the deployability sub-variant (ADR 0009):
      - ``"gtfs"`` — the GTFS-derived transit block. Higher signal, but scoring a new
        city requires ingesting that city's feed first.
      - ``"acs"``  — swaps the GTFS block for ACS commute/vehicle columns, which are
        nationally available, so the model can score a city with no bespoke ingestion.

    Returned unfiltered; `training.run_cv` narrows it to the columns the pool carries and
    records the survivors on the `FoldRun`.
    """
    from regression_modelling.constants import (
        PREDICTOR_SETS, TRANSIT_MODEL_PREDICTORS, ACS_TRANSIT_MODEL_PREDICTORS,
        AGENCY_ANCHOR_COL,
    )
    add_agency = name.endswith("+agency")
    base = "ours+approved" if add_agency else name
    preds = list(PREDICTOR_SETS[base])
    if transit == "acs":
        preds = ([p for p in preds if p not in TRANSIT_MODEL_PREDICTORS]
                 + list(ACS_TRANSIT_MODEL_PREDICTORS))
    if add_agency:
        preds = preds + [AGENCY_ANCHOR_COL[category]]
    return preds

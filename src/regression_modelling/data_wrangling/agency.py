"""Agency-level (police-jurisdiction) roll-up of BG predictors for the Task-1
adjusted-R^2 comparison against the existing national agency crime model.

Context
-------
The existing national model (``docs/business_context/crime_ct_models_revised.ipynb``)
fits a population-weighted ``RidgeCV``/``log1p`` regression on a severity-weighted
total agency crime rate (``wtotal_pt_m``) using a governance-approved
demographic/ACS predictor set (``new_predictors``). Task 1 holds that harness
constant and swaps in *our* BG predictor set (demographics + property + imagery +
POI, rolled up to agency level, plus ACS commute-mode transit proxies) to see how
adjusted R^2 moves.

Method
------
Mirrors the colleague's ``data_for_models.ipynb`` (cells 7, 30-37): build a
block-group -> agency crosswalk carrying block-population weights, then
population-weighted average any BG column up to the agency (``akey``) level. Raw
distress shares are aggregated first and log1p-transformed afterwards (aggregate
raw -> transform), matching both the colleague's pipeline and our BG feature build
(``dataset.build_model_table``).

Heavy GCS pulls (the 1.3 GB ``block_place.sav``) cache to ``data/interim/agency``;
downstream reloads are local. Functions take ``refresh: bool = False`` and only
re-fetch when the cache is missing or ``refresh=True``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crime_blockgroup_mapping.constants import GCS_ROOT, UCR_YEAR
from regression_modelling.config import agency_parquet, features_parquet
from regression_modelling.constants import (
    DEMOGRAPHIC_MODEL_TRANSFORMS, PREDICTOR_SETS, PROPERTY_MODEL_TRANSFORMS,
)
from regression_modelling.data_wrangling.sources import get_gcs_fs, read_sav_from_gcs

# Model-form -> raw column for the log1p'd predictors (``vacant_pct_log`` -> ``vacant_pct``).
_LOG_SPECS = {**PROPERTY_MODEL_TRANSFORMS, **DEMOGRAPHIC_MODEL_TRANSFORMS}
_RAW_OF = {f"{c}_log": c for c in _LOG_SPECS}

# Our national agency set = the promoted BG set (``selected_v1``) minus the families that
# only exist in the POC cities: GTFS transit, roadway morphology (HPMS/TIGER builds) and the
# adjacent-BG neighbour means (redundant once BGs are averaged to an agency anyway).
# DERIVED so it cannot drift from the BG model.
NATIONAL_PREDICTORS = [
    f for f in PREDICTOR_SETS["selected_v1"]
    if not f.startswith(("transit_", "roadway_")) and "_nbr" not in f
]
# Raw BG columns rolled up (population-weighted); log1p is applied AFTER aggregation.
_ROLLUP_COLS = [_RAW_OF.get(f, f) for f in NATIONAL_PREDICTORS]

# Transit's national stand-in: ACS commute-mode shares, reshaped (heavy zeros, right skew).
TRANSIT_PROXY_PREDICTORS = ["bus_pct_log", "train_pct_log", "has_transit"]

# ACS journey-to-work "train" = streetcar + subway + railroad (the colleague's
# train_pct definition); bus_pct is a direct ACS column.
_TRAIN_PARTS = ["streetcar_pct", "subway_pct", "railroad_pct"]


def wavg(df: pd.DataFrame, group: str, weight: str, values: list[str]) -> pd.DataFrame:
    """Population-weighted average of `values` within `group` (colleague cell 7).

    NaNs are dropped per-column before weighting, so a group's average uses only
    the observations that reported that column. Returns a frame indexed by `group`.
    """
    agg = df.copy()
    out = pd.DataFrame(columns=[group]).set_index(group)
    for col in values:
        agg[col] = agg[col] * agg[weight]
        summed = agg[agg[col].notna()].groupby(group)[[col, weight]].sum()
        out = out.join(summed, how="outer")
        out[col] = out[col] / out[weight]
        out = out.drop(columns=weight)
    return out


def build_bg_akey_crosswalk(refresh: bool = False) -> pd.DataFrame:
    """Block-group -> agency crosswalk with block-population weights.

    Rebuilds the colleague's block-level place matching (data_for_models cells
    30-31, 37): each 2020 census block is matched to a UCR agency (`akey`) via its
    Census place (falling back to county subdivision, then balance-of-county), then
    block populations are summed to the (bg_key, akey) grain. A block group that
    straddles multiple agencies yields several rows whose POP100 splits its people.

    Returns columns ``[bg_key, akey, POP100]``; caches to data/interim/agency.
    """
    path = agency_parquet("bg_akey_crosswalk")
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    fs = get_gcs_fs()
    year = UCR_YEAR
    root = GCS_ROOT.replace("gs://", "")

    # muni_key -> akey (agencies with population coverage)
    crosswalk, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{year}/ucr_crosswalk.sav", fs)
    muni_agency = (
        crosswalk[crosswalk["popest_geo"] > 0]
        .groupby("muni_key")["akey"].first()
    )
    print(f"muni->akey crosswalk: {muni_agency.nunique():,} agencies, "
          f"{len(muni_agency):,} muni keys")

    # block -> place/cousub/county, with block populations
    blocks, _ = read_sav_from_gcs(
        f"{GCS_ROOT}/demographic/population_estimates/{year}/block_place.sav", fs
    )
    need = {"STATE", "PLACE_CDP", "COUSUB", "COUNTY_2020", "GEOID20", "POP100"}
    missing = need - set(blocks.columns)
    if missing:
        raise KeyError(f"block_place.sav missing expected columns: {sorted(missing)}")
    print(f"block_place: {len(blocks):,} blocks")

    # place -> cousub -> balance-of-county fallbacks (colleague cell 31)
    blocks["muni_key"] = blocks["STATE"] + blocks["PLACE_CDP"]
    blocks = blocks.join(muni_agency, on="muni_key")

    unmatched = blocks["akey"].isna()
    blocks.loc[unmatched, "muni_key"] = blocks["STATE"] + blocks["COUSUB"]
    blocks = blocks.drop(columns="akey").join(muni_agency, on="muni_key")

    unmatched = blocks["akey"].isna()
    blocks.loc[unmatched, "muni_key"] = blocks["STATE"] + "99" + blocks["COUNTY_2020"]
    blocks["muni_key"] = blocks["muni_key"].replace({"3499013": "3446380"})
    blocks = blocks.drop(columns="akey").join(muni_agency, on="muni_key")

    matched_pop = blocks.loc[blocks["akey"].notna(), "POP100"].sum()
    total_pop = blocks["POP100"].sum()
    print(f"blocks matched to an agency: {matched_pop / total_pop:.1%} of population")

    blocks["bg_key"] = blocks["GEOID20"].str[:12]
    bg_muni = (
        blocks[blocks["akey"].notna()]
        .groupby(["bg_key", "akey"], as_index=False)["POP100"].sum()
    )
    print(f"bg->agency crosswalk: {len(bg_muni):,} (bg, agency) pairs; "
          f"{bg_muni['bg_key'].nunique():,} block groups; "
          f"{bg_muni['akey'].nunique():,} agencies")

    path.parent.mkdir(parents=True, exist_ok=True)
    bg_muni.to_parquet(path)
    return bg_muni


def rollup_predictors_to_agency(refresh: bool = False) -> pd.DataFrame:
    """Population-weighted roll-up of our BG predictor set to agency (`akey`) level.

    Joins the national BG predictor table (``data/interim/features/bg_predictors``)
    to the block-population crosswalk, averages every raw predictor to the agency
    grain, then log1p-transforms the property distress shares into ``{col}_log``
    (aggregate raw -> transform). Columns are exactly ``NATIONAL_PREDICTORS``; a cache
    missing any of them is rebuilt. Transit is intentionally excluded here — it is
    POC-city-only at BG level and is replaced by ACS commute-mode proxies
    (``load_transit_proxies``). Returns a frame indexed by ``akey``.
    """
    path = agency_parquet("agency_predictors")
    if path.exists() and not refresh:
        cached = pd.read_parquet(path)
        if set(NATIONAL_PREDICTORS) <= set(cached.columns):
            return cached[NATIONAL_PREDICTORS]
        print("agency_predictors cache predates NATIONAL_PREDICTORS; rebuilding")

    bg = pd.read_parquet(features_parquet("bg_predictors"))
    if "geoid" not in bg.columns:
        raise KeyError("bg_predictors.parquet must carry a 'geoid' block-group key")
    bg = bg.rename(columns={"geoid": "bg_key"})

    raw_cols = [c for c in _ROLLUP_COLS if c in bg.columns]
    missing = [c for c in _ROLLUP_COLS if c not in bg.columns]
    if missing:
        print(f"note: {len(missing)} raw predictor(s) absent from bg_predictors: {missing}")

    cross = build_bg_akey_crosswalk()
    merged = cross.merge(bg[["bg_key"] + raw_cols], on="bg_key", how="left")
    print(f"rolling up {len(raw_cols)} predictors over "
          f"{merged['bg_key'].nunique():,} block groups -> "
          f"{merged['akey'].nunique():,} agencies")

    agency = wavg(merged, group="akey", weight="POP100", values=raw_cols)

    # aggregate raw -> then log1p, matching the BG feature build (dataset.build_model_table):
    # property distress shares (+ spatial lags) and the right-skewed pop_est_5mile ring count
    # (ADR 0006). Single source of truth = the constants' transform specs.
    for col in _LOG_SPECS:
        if col in agency.columns:
            agency[f"{col}_log"] = np.log1p(agency[col].clip(lower=0))
    agency = agency.drop(columns=[c for c in _LOG_SPECS if c in agency.columns])
    agency = agency[[f for f in NATIONAL_PREDICTORS if f in agency.columns]]

    path.parent.mkdir(parents=True, exist_ok=True)
    agency.to_parquet(path)
    return agency


def load_transit_proxies(refresh: bool = False) -> pd.DataFrame:
    """ACS commute-mode transit proxies at agency (`akey`) level.

    Derives ``bus_pct`` and ``train_pct`` (= streetcar + subway + railroad, the
    colleague's "train" definition) from the agency-level ACS extract
    (``muni_acs.sav``), collapsing muni keys to agencies via the UCR crosswalk with
    population weighting. Returns a frame indexed by ``akey`` with columns
    ``[bus_pct, train_pct]``; caches to data/interim/agency.

    Coverage note: ``muni_acs`` stores county-level ACS under a 5-char ``SSCCC`` key,
    but the UCR crosswalk keys **balance-of-county** agencies as ``SS99CCC`` (the "99"
    placeholder). A naive ``muni_key`` join therefore misses ~3,100 large
    unincorporated-county agencies (34% of population) — they get no ACS record and
    silently drop from the modeling set. We remap those ``SS99CCC`` keys to the 5-char
    county key so they match muni_acs's county rows (recovering ~99.5% coverage),
    mirroring the county fallback in :func:`build_bg_akey_crosswalk` and the colleague's
    baseline-predictor assembly. County-wide ACS slightly over-covers the unincorporated
    balance, but it is the same approximation the baseline uses and far better than NaN.
    """
    path = agency_parquet("agency_transit_proxies")
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    fs = get_gcs_fs()
    year = UCR_YEAR
    muni_acs, _ = read_sav_from_gcs(
        f"{GCS_ROOT}/demographic/acs/5/{year - 1}/muni_acs.sav", fs
    )
    key = "muni_key" if "muni_key" in muni_acs.columns else muni_acs.columns[0]
    parts = [c for c in _TRAIN_PARTS if c in muni_acs.columns]
    muni_acs["train_pct"] = muni_acs[parts].sum(axis=1)
    pop_col = "population_acs" if "population_acs" in muni_acs.columns else None

    crosswalk, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{year}/ucr_crosswalk.sav", fs)
    cw = crosswalk[crosswalk["popest_geo"] > 0][["muni_key", "akey"]].copy()
    cw["muni_key"] = cw["muni_key"].astype(str)
    # ACS join key: remap balance-of-county agency keys (SS99CCC) to the 5-char county
    # key (SSCCC) muni_acs actually stores; place/cousub keys pass through unchanged.
    is_county99 = (cw["muni_key"].str.len() == 7) & (cw["muni_key"].str[2:4] == "99")
    cw["acs_key"] = cw["muni_key"].where(
        ~is_county99, cw["muni_key"].str[:2] + cw["muni_key"].str[4:])
    muni_agency = cw.groupby("acs_key")["akey"].first()

    df = muni_acs[[key, "bus_pct", "train_pct"] + ([pop_col] if pop_col else [])].copy()
    df = df.rename(columns={key: "acs_key"})
    df["acs_key"] = df["acs_key"].astype(str)
    df = df.join(muni_agency, on="acs_key")
    df = df[df["akey"].notna()]
    if pop_col:
        df = df.rename(columns={pop_col: "_w"})
        df["_w"] = df["_w"].fillna(0).clip(lower=0.01)
        out = wavg(df, group="akey", weight="_w", values=["bus_pct", "train_pct"])
    else:
        out = df.groupby("akey")[["bus_pct", "train_pct"]].mean()

    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def add_transit_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``TRANSIT_PROXY_PREDICTORS`` from raw ``bus_pct`` / ``train_pct`` (0-100 shares).

    ``has_transit`` flags any non-zero transit commuting; ``log1p`` compresses the right
    skew while keeping the zeros. Missing shares stay missing (``has_transit`` too).
    """
    out = df.copy()
    known = out[["bus_pct", "train_pct"]].notna().all(axis=1)
    out["has_transit"] = ((out["bus_pct"] > 0) | (out["train_pct"] > 0)).astype(float)
    out.loc[~known, "has_transit"] = np.nan
    out["bus_pct_log"] = np.log1p(out["bus_pct"].clip(lower=0))
    out["train_pct_log"] = np.log1p(out["train_pct"].clip(lower=0))
    return out


def build_wtotal_target(ct_muni_df: pd.DataFrame, fs=None) -> pd.DataFrame:
    """Add the severity-weighted total agency crime rate ``wtotal_pt_m`` in place.

    Faithful replica of ``crime_ct_models_revised.ipynb`` cells 15-17: projects
    city crime rates to the target year where needed, floors negatives at 0,
    broadcasts each agency's max rate, then forms the equal-representation relative
    total (each of the 7 primary crimes contributes 1/7 of its national-relative
    rate) and rescales by the national violent+property level. Returns the same
    frame for chaining.
    """
    fs = fs or get_gcs_fs()
    year = UCR_YEAR
    df = ct_muni_df
    crimes = ["violent", "murder", "rape", "robbery", "assault",
              "property", "burglary", "larceny", "mvt"]
    primary_crimes = [c for c in crimes if c not in ("violent", "property")]

    for i in crimes:
        col = f"{i}_pt_m"
        if df.groupby("akey")[col].std().max() == 0:
            df[col] = df[col] * (1 + ((df["year"] - year) * df[f"pctch_{i}_pt"]))
        df[col] = df[col].mask(df[col] < 0, 0)
        df.set_index(["akey", "year"], inplace=True)
        df[col] = df.groupby(["akey", "year"])[col].max()
        df.reset_index(inplace=True)

    df.loc[df[["violent_pt_m", "property_pt_m"]].notna().any(axis=1), "total_rel_m"] = 0
    us_df, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{year}/ucr_us.sav", fs)
    for i in crimes:
        df[f"{i}_pt_u"] = us_df[f"ucr_{i}_rate_u"][0]
    for i in primary_crimes:
        df["total_rel_m"] = df["total_rel_m"] + df[f"{i}_pt_m"] / df[f"{i}_pt_u"] / 7
    df["wtotal_pt_m"] = df["total_rel_m"] * df[["violent_pt_u", "property_pt_u"]].sum(axis=1)
    return df


def adjusted_r2(r2: float, n: int, p: int) -> float:
    """Adjusted R^2 = 1 - (1 - R^2)(n - 1)/(n - p - 1) for `n` obs, `p` predictors."""
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

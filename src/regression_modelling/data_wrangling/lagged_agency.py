"""Lagged UCR agency crime anchor — the Model D cross-city leveller (ADR 0007).

Why this exists
---------------
The refresh models the **within-city, city-demeaned** crime rate (04's scale case:
95% of the variance is within-city, and that is where the incumbent fails worst).
Demeaning deliberately throws away the between-city *level* — but the deployed index
(RiskMeter / Spatial API) is a nationally-indexed absolute score whose post-processing
*assumes the model already carries level*. Something must restore one number of level
per city.

ADR 0007 shows a *learned* structural leveller (a crime-free social-disorganization
composite, "C'") cannot do this: with 5–10 cities the between-city level is not
learnable and extrapolates catastrophically under leave-one-city-out. The only viable
leveller is an **observed** per-city number — lagged UCR agency crime — because it is
*given* for a held-out city rather than extrapolated, and crime-on-crime is ~1:1 so it
needs almost no estimation. It is city-constant, so it re-scales cities without
re-ordering neighborhoods (bounding the lagged-crime feedback-loop critique).

What this builds
----------------
For each POC city, one observed severity-weighted crime level from the year strictly
**before** the target (``UCR_YEAR - 1``), on the SAME weighted relative-risk scale as
the BG target (``compute_weighted_scores`` — equal-representation average of
national-relative crime rates). Output columns ``wtotal_rate``/``wprop_rate`` are
directly comparable to the model's ``wtotal``/``wprop`` targets.

Method
------
1. Resolve each city -> UCR agency (``akey``) via the same place crosswalk the BG->agency
   roll-up uses (``muni_key = state_fips + place_fips`` -> ``ucr_crosswalk.sav``), with a
   manual override for consolidated city-counties whose place key is absent.
2. Pull the agency's raw offense counts + population for the lag year from the merged
   ``ucr_history.sav`` agency x year panel (two most recent UCR vintages concatenated for
   coverage); if that year is missing for the agency, fall back to its most recent year
   strictly before ``UCR_YEAR``.
3. Convert counts to per-1,000-resident rates, then run ``compute_weighted_scores`` to
   get ``wtotal_rate``/``wprop_rate`` on the target's scale. Population denominator is the
   agency's reported ``Population``; when that is missing (agency reporting gaps) it falls
   back to the crosswalk population estimate ``popest_geo``.

Caches to ``data/interim/agency/lagged_agency_anchor.parquet``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crime_blockgroup_mapping.constants import CITIES, GCS_ROOT, UCR_YEAR
from crime_blockgroup_mapping.scores import PRIMARY_CRIMES, compute_weighted_scores
from regression_modelling.config import agency_parquet
from regression_modelling.data_wrangling.sources import get_gcs_fs, read_sav_from_gcs

# Cities whose Census place key is not in the agency place crosswalk (or whose city ORI
# stopped reporting) and must be pinned to their real reporting agency by hand.
CITY_AKEY_OVERRIDES = {
    # Jacksonville is a consolidated city-county: the place key 12-35000 is absent from
    # the crosswalk and the old city ORI (JACKSONVILLE_FL) stopped reporting after 2020.
    # Crime for the consolidated city reports under the Sheriff's Office.
    "jacksonville": "JACKSONVILLESHERIFFSOFFICE_FL_C",
}


def resolve_city_akey(refresh: bool = False) -> pd.DataFrame:
    """Map each POC city to its UCR agency key (`akey`).

    Primary path mirrors the BG->agency roll-up: ``muni_key = state_fips + place_fips``
    joined to ``ucr_crosswalk.sav`` (population-covered agencies only). Cities in
    :data:`CITY_AKEY_OVERRIDES` are pinned directly. Returns columns
    ``[city, name, muni_key, akey, source]``.
    """
    fs = get_gcs_fs()
    cw, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{UCR_YEAR}/ucr_crosswalk.sav", fs)
    muni_agency = cw[cw["popest_geo"] > 0].groupby("muni_key")["akey"].first()

    rows = []
    for city, cfg in CITIES.items():
        muni_key = f"{cfg.state_fips}{cfg.place_fips}"
        if city in CITY_AKEY_OVERRIDES:
            akey, source = CITY_AKEY_OVERRIDES[city], "override"
        else:
            akey = muni_agency.get(muni_key)
            source = "place_crosswalk"
        if akey is None:
            raise KeyError(
                f"No agency for {city} (muni_key={muni_key}); add a CITY_AKEY_OVERRIDES entry"
            )
        rows.append(dict(city=city, name=cfg.name, muni_key=muni_key, akey=akey, source=source))
    return pd.DataFrame(rows)


def _agency_pop_estimates() -> pd.Series:
    """Crosswalk population estimate (`popest_geo`) per agency — the denominator fallback."""
    fs = get_gcs_fs()
    cw, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{UCR_YEAR}/ucr_crosswalk.sav", fs)
    return cw.groupby("akey")["popest_geo"].max()


def _load_history_panel(fs) -> pd.DataFrame:
    """Merged agency x year offense panel across the two most recent UCR vintages.

    Each yearly UCR drop ships an ``ucr_history.sav`` covering a rolling ~6-year window,
    and later vintages carry revisions but drop the oldest years. Concatenating the
    ``UCR_YEAR`` and ``UCR_YEAR-1`` files and de-duplicating on ``(akey, year)`` — keeping
    the newer vintage's revised figures — maximizes lagged coverage for chronically
    under-reporting agencies (e.g. Pittsburgh, whose only pre-target year is 2018).
    """
    frames = []
    for vintage in (UCR_YEAR - 1, UCR_YEAR):
        df, _ = read_sav_from_gcs(f"{GCS_ROOT}/crime/{vintage}/ucr_history.sav", fs)
        frames.append(df.assign(_vintage=vintage))
    merged = pd.concat(frames, ignore_index=True).sort_values("_vintage")
    merged = merged.drop_duplicates(["akey", "year"], keep="last").drop(columns="_vintage")
    return merged


def build_lagged_agency_anchor(lag_year: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Per-city observed lagged agency crime level on the BG-target scale.

    Parameters
    ----------
    lag_year : year of the anchor. Defaults to ``UCR_YEAR - 1`` (strictly before the
        target, most recent available). If an agency did not report that exact year, the
        most recent year strictly before ``UCR_YEAR`` is used instead.
    refresh : rebuild the cache even if present.

    Returns
    -------
    DataFrame indexed by ``city`` with ``[name, akey, year_used, population, pop_source,
    wtotal_rate, wprop_rate]``. ``wtotal_rate``/``wprop_rate`` share the model target's
    weighted relative-risk scale (``compute_weighted_scores``).
    """
    lag_year = lag_year or (UCR_YEAR - 1)
    path = agency_parquet("lagged_agency_anchor")
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    fs = get_gcs_fs()
    hist = _load_history_panel(fs)
    hist = hist[hist["year"] < UCR_YEAR].copy()
    xwalk = resolve_city_akey()
    pop_est = _agency_pop_estimates()
    count_cols = PRIMARY_CRIMES  # murder, rape, robbery, assault, burglary, larceny, mvt

    records = []
    for r in xwalk.itertuples(index=False):
        panel = hist[hist["akey"] == r.akey]
        if panel.empty:
            raise KeyError(f"{r.city}: agency {r.akey} has no rows in ucr_history < {UCR_YEAR}")
        row = panel[panel["year"] == lag_year]
        if row.empty:  # agency didn't report the target lag year -> most recent prior year
            row = panel.sort_values("year").iloc[[-1]]
        row = row.iloc[0]

        pop = row.get("Population")
        pop_source = "reported"
        if pop is None or pd.isna(pop) or pop <= 0:
            pop, pop_source = pop_est.get(r.akey), "popest_geo"
        if pop is None or pd.isna(pop) or pop <= 0:
            raise ValueError(f"{r.city}: no usable population for {r.akey}")

        rec = dict(city=r.city, name=r.name, akey=r.akey, year_used=int(row["year"]),
                   population=float(pop), pop_source=pop_source)
        for c in count_cols:
            rec[f"{c}_rate"] = float(row[c]) / pop * 1000.0  # per 1,000 residents
        records.append(rec)

    df = pd.DataFrame(records)
    df = compute_weighted_scores(df)
    keep = ["city", "name", "akey", "year_used", "population", "pop_source",
            "wtotal_rate", "wprop_rate"]
    df = df[keep].set_index("city").sort_index()

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df

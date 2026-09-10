"""Pooled + leave-one-city-out (LOCO) cross-validation harness (ADR 0003).

Built one component at a time.
  Component 1 = pooled loader + LOCO fold split.
  Component 2 = leakage-safe fit/predict core (train-only scaling, target transforms,
                held-out prediction).
The LOCO driver loop and the held-out metrics land in later components.

Cities are the FULL-COVERAGE set (violent incl. rape) — derived from the
`property_only` flag on `CityConfig`, never hardcoded (config-over-hardcoding).
"""
from typing import Iterator

import numpy as np
import pandas as pd
import statsmodels.api as sm

from crime_blockgroup_mapping.config import PROCESSED_DIR
from crime_blockgroup_mapping.constants import CITIES
from regression_modelling.constants import PREDICTOR_COLS
from regression_modelling.data_wrangling.dataset import build_model_table


def full_coverage_cities() -> list[str]:
    """POC cities whose crime source geolocates violent crime (incl. rape).

    Derived from `CityConfig.property_only` so it can never drift from the per-city
    config: property-only cities (SF, Pittsburgh, Columbus, Jacksonville, Sacramento)
    are excluded; the survivors are Houston, Chicago, Atlanta, Kansas City, Detroit.
    """
    return [c for c, cfg in CITIES.items() if not cfg.property_only]


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

    - Tags every row with a `city` column (the LOCO grouping key).
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


def loco_folds(pooled: pd.DataFrame,
               city_col: str = "city") -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """Yield leave-one-city-out folds: (held_out_city, train_df, holdout_df).

    Every city holds out exactly once (rotating LOCO). Train is all other cities pooled;
    holdout is the single left-out city — the generalization target ADR 0003 cares about.
    """
    for city in sorted(pooled[city_col].unique()):
        holdout = pooled[pooled[city_col] == city]
        train = pooled[pooled[city_col] != city]
        yield city, train, holdout


# =========================================================================== #
# Component 2 — leakage-safe fit/predict core                                 #
# =========================================================================== #
# Three target forms share ONE design matrix; the scaler is fit on TRAIN cities
# only and applied to the holdout, so no holdout information touches the fit.

TARGET_MODES = ("rate_within_city", "rate", "rate_daytime_within_city",
                "rate_daytime", "logcount", "lograte", "lograte_within_city")

_DAYTIME_MODES = {"rate_daytime", "rate_daytime_within_city"}
_LOGRATE_MODES = {"lograte", "lograte_within_city"}


def _rate_col(mode: str, category: str) -> str:
    """The actual-rate column a mode is built on: daytime denominator (population +
    LODES jobs) for the `*_daytime` modes, plain population rate otherwise."""
    suffix = "_rate_daytime" if mode in _DAYTIME_MODES else "_rate"
    return f"{category}{suffix}"


def make_target(df: pd.DataFrame, mode: str = "rate_within_city",
                category: str = "cl_total", city_col: str = "city",
                winsor_upper: float | None = None) -> pd.Series:
    """Build the regression target `y` for a fit mode (Series aligned to `df.index`).

    - ``rate_within_city`` / ``rate_daytime_within_city`` — HEADLINE (c): per-city
      z-scored rate (plain / daytime denominator), each city by ITS OWN mean/sd. Learns
      *relative* within-city BG risk (city level+scale removed). Computed groupwise, so on
      a pooled TRAIN frame each city standardizes to itself and no cross-city level leaks.
    - ``rate`` / ``rate_daytime`` — REPORTED (a): raw rate (absolute level, one intercept).
      ``rate_daytime`` = `{category}_rate_daytime` (per 1k of population + jobs) is the
      preferred interpretable target — daytime denominator + the loader's daytime_pop floor
      already tame the small-denominator skew.
    - ``logcount`` — comparator: `{category}_logcount` = log(count+1).
    - ``lograte`` — PRODUCT TARGET (pooled): `log1p({category}_rate)`, the absolute log
      rate on one intercept. With `category="wtotal"` this is `log(weighted total rate)`,
      the single headline modeling target (ADR 0003/0005). Winsorize acts on the rate
      BEFORE the log, same as the rate modes.
    - ``lograte_within_city`` — PRODUCT TARGET (LOCO within-city): the same log rate
      z-scored per city (each city by its own mean/sd), so the fit learns *relative*
      within-city risk with the cross-city level+scale removed.

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


def fit_scaler(train: pd.DataFrame, predictors=PREDICTOR_COLS) -> pd.Series:
    """Z-score parameters (mean, sd) fit on TRAIN predictors only (leakage-safe).

    Zero-variance columns get sd=1 so they map to 0 instead of dividing by zero.
    """
    mu = train[predictors].mean()
    sd = train[predictors].std(ddof=0).replace(0, 1.0)
    return mu, sd


def apply_scaler(df: pd.DataFrame, scaler, predictors=PREDICTOR_COLS) -> pd.DataFrame:
    """Apply a train-fit (mu, sd) to any frame -> standardized predictor block."""
    mu, sd = scaler
    return (df[predictors] - mu) / sd


def _demean_predictors(df: pd.DataFrame, predictors, city_col: str = "city") -> pd.DataFrame:
    """Subtract each row's OWN city mean from every predictor (the within transform).

    Target-paired usage (see make_target / ADR 0003): pair this with the within-city
    z-score target so both sides of the regression are per-city demeaned — the textbook
    "within"/fixed-effects estimator via Frisch-Waugh, but implemented so it survives test
    time. Crucially this is LEAKAGE-SAFE even for a held-out city: it uses the city's own
    OBSERVED predictors (X is not the outcome), so at LOCO predict time the held-out city
    is demeaned by its own X-means — no city dummy is needed and no outcome information
    leaks. Cities absent from `df` simply demean against whatever rows are present.
    """
    block = df[list(predictors)]
    return block - block.groupby(df[city_col].to_numpy()).transform("mean")


def fit_fold(train: pd.DataFrame, mode: str = "rate_within_city",
             predictors=PREDICTOR_COLS, category: str = "cl_total",
             robust: str = "HC3", winsor_upper: float | None = None,
             demean_by_city: bool = False, city_col: str = "city") -> dict:
    """Fit a standardized OLS on ONE train fold. Scaler is fit on this train frame only.

    When ``demean_by_city`` is True the predictors are per-city demeaned BEFORE scaling
    (the within estimator — pair with a ``*_within_city`` target; see `_demean_predictors`).
    Returns a dict bundling the fitted model, its HC3-robust wrapper, the train scaler, and
    the metadata (incl. ``demean_by_city``/``city_col``) needed to score a holdout.
    """
    d = train.copy()
    predictors = [p for p in predictors if p in d.columns]
    d["_y"] = make_target(d, mode, category, winsor_upper=winsor_upper)
    d = d.dropna(subset=list(predictors) + ["_y"])

    block = _demean_predictors(d, predictors, city_col) if demean_by_city else d[predictors]
    scaler = (block.mean(), block.std(ddof=0).replace(0, 1.0))
    X = sm.add_constant((block - scaler[0]) / scaler[1], has_constant="add")
    result = sm.OLS(d["_y"].to_numpy(), X).fit()
    robust_res = result.get_robustcov_results(cov_type=robust)

    return {
        "result": result, "robust": robust_res, "scaler": scaler,
        "predictors": list(predictors), "mode": mode, "category": category,
        "n_train": int(result.nobs),
        "demean_by_city": demean_by_city, "city_col": city_col,
    }


def predict_fold(fit: dict, holdout: pd.DataFrame) -> pd.DataFrame:
    """Score a holdout city with a fitted fold. Returns the holdout rows (predictors
    non-null) with an added `y_pred` risk-score column.

    The holdout predictors are standardized with the TRAIN scaler (never re-fit), so the
    prediction uses no holdout information. When the fit was ``demean_by_city``, the holdout
    is first demeaned by ITS OWN city means (leakage-safe; see `_demean_predictors`), then
    scaled. `y_pred` is a *relative risk score* in the within-city modes — ranking is what
    the concentration metric consumes, so it is not un-standardized.
    """
    d = holdout.dropna(subset=fit["predictors"]).copy()
    block = (_demean_predictors(d, fit["predictors"], fit.get("city_col", "city"))
             if fit.get("demean_by_city") else d[fit["predictors"]])
    mu, sd = fit["scaler"]
    X = sm.add_constant((block - mu) / sd, has_constant="add")
    d["y_pred"] = np.asarray(fit["result"].predict(X))
    return d


# =========================================================================== #
# Component 3 — LOCO driver                                                   #
# =========================================================================== #
# Rotating leave-one-city-out: fit on the train cities, score the held-out city,
# repeat. The concatenated `scored` frame is every BG's OUT-OF-SAMPLE prediction
# (each city predicted only while it was the holdout) — the input to the metrics.


def run_loco(pooled: pd.DataFrame, mode: str = "rate_within_city",
             predictors=PREDICTOR_COLS, category: str = "cl_total",
             winsor_upper: float | None = None,
             demean_by_city: bool = False) -> dict:
    """Run rotating LOCO for one target mode.

    Returns a dict:
      - ``scored``  : DataFrame of ALL holdout rows with a `y_pred` risk score and a
                      `holdout_city` tag — every BG scored while its city was held out.
      - ``fits``    : {city: fit-dict from `fit_fold`} for per-fold coefficient inspection.
      - ``mode`` / ``category`` / ``predictors`` / ``split`` : run metadata.
    Pass ``demean_by_city=True`` to pair per-city demeaned predictors with a within-city
    target (the within estimator). Prints per-fold train/holdout sizes.
    """
    print(f"LOCO — target mode = {mode!r} ({category})"
          + (", demeaned-X" if demean_by_city else "")
          + (f", winsor@{winsor_upper:g}" if winsor_upper is not None else ""))
    fits, parts = {}, []
    for city, train, holdout in loco_folds(pooled):
        fit = fit_fold(train, mode=mode, predictors=predictors, category=category,
                       winsor_upper=winsor_upper, demean_by_city=demean_by_city)
        scored = predict_fold(fit, holdout).assign(holdout_city=city)
        fits[city] = fit
        parts.append(scored)
        print(f"  holdout={city:14} n_train={fit['n_train']:>5} scored={len(scored):>5}"
              f" adjR2_in={fit['result'].rsquared_adj:6.3f}")

    scored = pd.concat(parts, ignore_index=True)
    print(f"  -> {len(scored)} BGs scored out-of-sample across {len(fits)} folds")
    return {"scored": scored, "fits": fits, "mode": mode, "split": "loco",
            "category": category, "predictors": list(predictors)}


def run_all_modes(pooled: pd.DataFrame, predictors=PREDICTOR_COLS,
                  category: str = "cl_total", winsor_upper: float | None = None) -> dict[str, dict]:
    """Convenience: run LOCO for every target mode. Returns {mode: run_loco result}."""
    return {mode: run_loco(pooled, mode=mode, predictors=predictors, category=category,
                           winsor_upper=winsor_upper)
            for mode in TARGET_MODES}


# =========================================================================== #
# Component 3b — stratified 80/20 holdout driver                              #
# =========================================================================== #
# A single random split, stratified by city (every city contributes ~80% train
# / 20% test). This is the INTERPOLATION protocol: "predict unseen BGs in cities
# we have partly seen." It leaks spatially (a BG's neighbour can sit across the
# split) and knows each city's level, so its R2 is OPTIMISTIC — the gap between
# it and LOCO (extrapolation) is exactly the value of having seen a city before.


def stratified_split(pooled: pd.DataFrame, test_size: float = 0.20, seed: int = 0,
                     city_col: str = "city") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random train/test split stratified by city (each city split ~(1-test_size)/test_size).

    Returns (train, test). The split is on rows only; the scaler + any per-city demean are
    still fit on TRAIN inside `fit_fold`, so it stays leakage-safe on the fit side.
    """
    test = pooled.groupby(city_col, group_keys=False).sample(frac=test_size, random_state=seed)
    train = pooled.drop(test.index)
    print(f"  stratified {int((1-test_size)*100)}/{int(test_size*100)} split "
          f"-> train={len(train)} test={len(test)} across "
          f"{pooled[city_col].nunique()} cities (seed={seed})")
    return train, test


def run_holdout(pooled: pd.DataFrame, mode: str = "lograte",
                predictors=PREDICTOR_COLS, category: str = "cl_total",
                test_size: float = 0.20, seed: int = 0,
                winsor_upper: float | None = None,
                demean_by_city: bool = False) -> dict:
    """Fit on a stratified TRAIN split, score the held-out TEST rows. Mirrors `run_loco`'s
    return schema (``scored`` tagged with `holdout_city` = each row's own city, plus a
    single ``fit``) so `loco_metrics` / `plot_lorenz` work unchanged. ``fits`` is left empty
    so no per-city in-sample adj-R2 is attached; read `fit['result'].rsquared_adj` for the
    pooled in-sample number.
    """
    print(f"HOLDOUT — target mode = {mode!r} ({category})"
          + (", demeaned-X" if demean_by_city else "")
          + (f", winsor@{winsor_upper:g}" if winsor_upper is not None else ""))
    train, test = stratified_split(pooled, test_size=test_size, seed=seed)
    fit = fit_fold(train, mode=mode, predictors=predictors, category=category,
                   winsor_upper=winsor_upper, demean_by_city=demean_by_city)
    scored = predict_fold(fit, test).assign(holdout_city=lambda d: d["city"])
    print(f"  n_train={fit['n_train']:>5} scored={len(scored):>5}"
          f" adjR2_in={fit['result'].rsquared_adj:6.3f}")
    return {"scored": scored, "fit": fit, "fits": {}, "mode": mode, "split": "holdout",
            "category": category, "predictors": list(predictors)}


# =========================================================================== #
# Component 4 — held-out metrics                                              #
# =========================================================================== #
# Concentration/Lorenz (headline), capture@k, oracle-normalized skill, plus
# out-of-sample R2/RMSE/MAE (rate mode) and Spearman. Implemented locally rather
# than importing carrier_eval's Lorenz: the architecture forbids task-to-task
# imports; consolidating both into the shared foundation is the ADR 0005 follow-up.


def _outcome_col(df: pd.DataFrame, category: str) -> str:
    """The crime-COUNT column a concentration/ranking metric captures. Weighted-rate
    categories carry no count of their own, so map to the composite they represent:
    `wprop` -> `property_count` (burglary+larceny+mvt), `wtotal` -> `cl_total_count`.
    Anything else uses `{category}_count`, falling back to `cl_total_count`."""
    weighted = {"wprop": "property_count", "wtotal": "cl_total_count"}
    col = weighted.get(category, f"{category}_count")
    return col if col in df.columns else "cl_total_count"


def _lorenz_points(score, outcome, weight):
    """Cumulative (x, y) for a concentration curve, BGs sorted by `score` descending.
    x = cumulative share of `weight` (population or 1-per-BG); y = cumulative share of
    `outcome` (crime count). Origin (0,0) is prepended."""
    order = np.argsort(-np.asarray(score, dtype=float))
    o = np.asarray(outcome, dtype=float)[order]
    w = np.asarray(weight, dtype=float)[order]
    cum_o = np.concatenate([[0], np.cumsum(o) / o.sum()])
    cum_w = np.concatenate([[0], np.cumsum(w) / w.sum()])
    return cum_w, cum_o


def _gini(x, y):
    """2*AUC - 1 for a cumulative curve (0 = no skill / diagonal, ->1 = perfect)."""
    return float(2 * np.trapezoid(y, x) - 1)


def concentration_stats(df: pd.DataFrame, score_col: str = "y_pred",
                        x_unit: str = "population", category: str = "cl_total",
                        capture_at: float = 0.20) -> dict:
    """Concentration metrics for one scored frame (a city, or pooled).

    x_unit="population" (default) weights the x-axis by BG population — the coherent
    choice for a per-capita rate index (the null diagonal = constant rate everywhere).
    x_unit="daytime_pop" weights by population + LODES jobs (the coherent null for the
    daytime-rate target). x_unit="bg" gives every BG equal x-weight (per-place framing).
    The `oracle` ranking is `count/weight` (= rate under a population weight, = count under
    bg), i.e. the best achievable concentration; `skill` normalizes the model's Gini by it.
    """
    outcome = df[_outcome_col(df, category)].to_numpy(dtype=float)
    if x_unit == "population":
        weight = df["population"].to_numpy(dtype=float)
    elif x_unit == "daytime_pop":
        weight = df["daytime_pop"].to_numpy(dtype=float)
    else:
        weight = np.ones(len(df))
    x, y = _lorenz_points(df[score_col].to_numpy(), outcome, weight)
    gini = _gini(x, y)

    density = outcome / np.where(weight > 0, weight, np.nan)     # oracle ranking key
    xo, yo = _lorenz_points(np.nan_to_num(density, nan=-np.inf), outcome, weight)
    gini_oracle = _gini(xo, yo)

    return {
        "gini": round(gini, 3),
        "gini_oracle": round(gini_oracle, 3),
        "skill": round(gini / gini_oracle, 3) if gini_oracle else np.nan,
        f"capture@{int(capture_at*100)}": round(float(np.interp(capture_at, x, y)), 3),
    }


def error_stats(df: pd.DataFrame, score_col: str = "y_pred",
                rate_col: str = "cl_total_rate") -> dict:
    """Out-of-sample R2 / RMSE / MAE. Meaningful only for the ABSOLUTE rate modes
    (`rate`, `rate_daytime`), where `y_pred` and the actual `rate_col` share units."""
    y = df[rate_col].to_numpy(dtype=float)
    yhat = df[score_col].to_numpy(dtype=float)
    resid = y - yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "r2_oos": round(1 - np.sum(resid ** 2) / ss_tot, 3) if ss_tot else np.nan,
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 2),
        "mae": round(float(np.mean(np.abs(resid))), 2),
    }


def hotspot_metrics(df: pd.DataFrame, score_col: str = "y_pred",
                    category: str = "cl_total", k: float = 0.20) -> dict:
    """Classification-style ranking metrics for a scored frame.

    The true hotspots are the top-`k` fraction of BGs by ACTUAL crime count; the model
    flags the top-`k` by predicted score. Because both selections are the same size,
    precision == recall (== the hit rate of the flagged list). `capture` is the share of
    total crime COUNT that lands in the predicted top-`k` BGs.
    """
    n = len(df)
    topn = max(1, int(round(k * n)))
    score = df[score_col].to_numpy(dtype=float)
    count = df[_outcome_col(df, category)].to_numpy(dtype=float)
    pred_top = np.argsort(-score)[:topn]
    actual_top = np.argsort(-count)[:topn]
    hits = np.intersect1d(pred_top, actual_top).size
    prec = hits / topn
    tag = f"{int(k * 100)}%bg"
    return {
        f"precision@{tag}": round(prec, 3),
        f"recall@{tag}": round(prec, 3),
        f"capture@{tag}": round(float(count[pred_top].sum() / count.sum()), 3),
    }


def loco_metrics(run: dict, x_unit: str = "population", capture_at: float = 0.20) -> pd.DataFrame:
    """Per-holdout-city + pooled metrics table for a LOCO run.

    Concentration + Spearman for every mode; R2/RMSE/MAE added only for the ABSOLUTE rate
    modes (`rate`, `rate_daytime`). Spearman and the error stats compare against the actual
    rate column matching the mode's denominator (daytime rate for the `*_daytime` modes).
    'POOLED' stacks all out-of-sample rows into one curve/score.
    """
    from scipy.stats import spearmanr
    scored, cat, mode = run["scored"], run["category"], run["mode"]
    rate_col = _rate_col(mode, cat)
    fits = run.get("fits", {})

    def _row(name, g):
        d = {"holdout": name, "n": len(g)}
        d.update(concentration_stats(g, x_unit=x_unit, category=cat, capture_at=capture_at))
        d["spearman"] = round(spearmanr(g["y_pred"], g[rate_col]).statistic, 3)
        if mode in ("rate", "rate_daytime"):
            d.update(error_stats(g, rate_col=rate_col))
        elif mode == "lograte":
            # y_pred and the target share log-rate units -> compare against log1p(rate)
            gg = g.assign(_logactual=np.log1p(g[rate_col].astype(float)))
            d.update(error_stats(gg, rate_col="_logactual"))
        # in-sample adjusted R2 of the fold's fit (OLS folds only; GBM fits carry no R2)
        f = fits.get(name)
        if isinstance(f, dict) and "result" in f:
            d["adj_r2_in"] = round(float(f["result"].rsquared_adj), 3)
        return d

    rows = [_row(city, g) for city, g in scored.groupby("holdout_city")]
    rows.append(_row("POOLED", scored))
    return pd.DataFrame(rows).set_index("holdout")


def plot_lorenz(run: dict, x_unit: str = "population", category: str = None, ax=None):
    """Plot per-city + pooled concentration curves for a LOCO run (BGs sorted by y_pred).
    Y = cumulative crime captured; X = cumulative population (or BG) share."""
    import matplotlib.pyplot as plt
    cat = category or run["category"]
    scored = run["scored"]
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(6, 5))

    for city, g in scored.groupby("holdout_city"):
        outcome = g[_outcome_col(g, cat)].to_numpy(dtype=float)
        weight = (g["population"].to_numpy(dtype=float) if x_unit == "population"
                  else np.ones(len(g)))
        x, y = _lorenz_points(g["y_pred"].to_numpy(), outcome, weight)
        ax.plot(x, y, lw=1.3, alpha=.85,
                label=f"{city} (Gini={_gini(x, y):.2f})")

    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="no skill")
    ax.set_xlabel(f"cumulative share of {'population' if x_unit=='population' else 'block groups'}")
    ax.set_ylabel(f"cumulative share of {cat} crime captured")
    ax.set_title(f"LOCO concentration — {run['mode']} (x={x_unit})")
    ax.legend(fontsize=8, loc="lower right")
    if standalone:
        plt.tight_layout(); plt.show()
    return ax

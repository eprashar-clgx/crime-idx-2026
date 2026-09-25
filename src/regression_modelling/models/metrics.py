"""Held-out metrics: everything that reads a `FoldRun` and returns a number (ADR 0008).

Consumes the `results.FoldRun` contract and nothing else — in particular it never
imports `training`, so metrics stay indifferent to how a run was produced. A new fold
protocol or estimator arrives here already supported.

The vocabulary is fixed in CONTEXT.md ("Metric vocabulary"); the short version:

- **Skill scores** (`r2_oos`, `skill`) are level-sensitive — they punish a model that
  ranks correctly but sits at the wrong level.
- **Correlations** (`within_corr2`, `between_city_level_r`, `spearman`) are
  affine-invariant, so they measure ranking only. Never call one an R².
- **Rank capture** (`recall@10`, `capture@20`, `gini`) is what the product actually
  ships: which blocks land in the flagged list.

Implemented locally rather than importing carrier_eval's Lorenz: the architecture
forbids task-to-task imports; consolidating both into the shared foundation is the
ADR 0005 follow-up.
"""
import numpy as np
import pandas as pd

from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.results import FoldRun


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
    The `oracle` ranking is `count/weight` (= rate under a population weight, = count
    under bg), i.e. the best achievable concentration; `skill` normalizes the model's
    Gini by it.
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
    """Out-of-sample R2 / RMSE / MAE. Meaningful only where `y_pred` and the actual
    `rate_col` share units (the absolute rate modes, or log-rate vs log1p(rate))."""
    y = df[rate_col].to_numpy(dtype=float)
    yhat = df[score_col].to_numpy(dtype=float)
    resid = y - yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "r2_oos": round(1 - np.sum(resid ** 2) / ss_tot, 3) if ss_tot else np.nan,
        "rmse": round(float(np.sqrt(np.mean(resid ** 2))), 2),
        "mae": round(float(np.mean(np.abs(resid))), 2),
    }


def within_city_recall(run: FoldRun, category: str | None = None,
                       danger_top: float = 0.25, nets=(0.25,),
                       safe_below: float = 0.50) -> pd.DataFrame:
    """Dangerous-block recall, computed WITHIN city (mirrors 04_bg_comparison).

    The business-cost framing from 04: under-prediction in risky blocks is the costly
    error, over-prediction in safe blocks is cheap. So rank BGs *within their own city*
    and ask, for the truly dangerous ones, where the model puts them.

    - Truly dangerous = the top ``danger_top`` share of a city's BGs by OBSERVED
      within-city rate (`{category}_rate`). Default **0.25** = the worst quartile.
    - For those blocks, take the model's within-city percentile of `y_pred`:
        ``recall@top{N}`` (one per ``net`` in ``nets``) = share the model also ranks in
          its own within-city top-``net`` (higher = better). With the default symmetric
          setting (``danger_top`` = ``net`` = 0.25) this reads plainly as: *of a city's
          observed worst-quartile blocks, the share the model also puts in its predicted
          worst quartile*. Because the two selections are the same size, chance = N.
        ``called_safe`` = share the model buries below its within-city median
          (`safe_below`) — the costly miss (lower = better).
    Per-city rows + a POOLED row that averages membership over every city's dangerous
    blocks (so each city is weighted by its dangerous-block count, matching 04).
    """
    cat = category or run.category
    rate_col = _rate_col(run.mode, cat)
    scored = run.scored

    rows, pooled_mp = [], []
    for city, g in scored.groupby("holdout_city"):
        obs_pct = g[rate_col].astype(float).rank(pct=True)
        mod_pct = g["y_pred"].astype(float).rank(pct=True)
        danger = obs_pct >= (1 - danger_top)
        if danger.sum() == 0:
            continue
        mp = mod_pct[danger]
        pooled_mp.append(mp)
        row = {"holdout": city, "n_danger": int(danger.sum())}
        for net in nets:
            row[f"recall@top{int(net * 100)}"] = round(float((mp >= (1 - net)).mean()), 3)
        row["called_safe"] = round(float((mp < safe_below).mean()), 3)
        rows.append(row)

    allmp = pd.concat(pooled_mp)
    prow = {"holdout": "POOLED", "n_danger": int(len(allmp))}
    for net in nets:
        prow[f"recall@top{int(net * 100)}"] = round(float((allmp >= (1 - net)).mean()), 3)
    prow["called_safe"] = round(float((allmp < safe_below).mean()), 3)
    rows.append(prow)
    return pd.DataFrame(rows).set_index("holdout")


def fold_metrics(run: FoldRun, x_unit: str = "population",
                 capture_at: float = 0.20) -> pd.DataFrame:
    """Per-holdout-city + pooled metrics table for any `FoldRun`.

    Concentration + Spearman for every mode; R2/RMSE/MAE added where units line up
    (the absolute rate modes, and `lograte` against log1p(rate)). 'POOLED' stacks all
    out-of-sample rows into one curve/score.
    """
    from scipy.stats import spearmanr
    scored, cat, mode = run.scored, run.category, run.mode
    rate_col = _rate_col(mode, cat)

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
        # in-sample adjusted R2 of the fold's fit (OLS folds only; GBM carries no R2)
        fit = run.fits.get(name)
        result = getattr(fit, "result", None)
        if result is not None:
            d["adj_r2_in"] = round(float(result.rsquared_adj), 3)
        return d

    rows = [_row(city, g) for city, g in scored.groupby("holdout_city")]
    rows.append(_row("POOLED", scored))
    return pd.DataFrame(rows).set_index("holdout")


#: Retained name — the table is no longer LOCO-specific now that one driver serves both
#: protocols, but the notebooks call it by this name.
loco_metrics = fold_metrics


# =========================================================================== #
# Cross-city decomposition                                                    #
# =========================================================================== #
# Pooled BG r2_oos is dominated by within-city scatter (~95% of the variance), so it
# hides whether a model is good at ranking blocks, at levelling cities, or both. These
# two split the question.

def within_corr2_pooled(run: FoldRun) -> float:
    """Mean per-city within-city **squared correlation** (Pearson² of y_pred vs
    log1p(rate)) — within-city variance explained.

    Deliberately not called "R²": it is affine-invariant, so a model that ranks a city
    correctly but levels it badly scores identically to one that gets both right. The
    level-sensitive counterpart is `r2_oos`; the gap between them is the calibration
    debt. See the metric vocabulary in CONTEXT.md.
    """
    rate_col = _rate_col(run.mode, run.category)
    r2s = []
    for _, g in run.scored.groupby("holdout_city"):
        if len(g) <= 2:
            continue
        y = np.log1p(g[rate_col].astype(float))
        p = g["y_pred"].astype(float)
        r = np.corrcoef(y, p)[0, 1]
        if np.isfinite(r):
            r2s.append(r * r)
    return round(float(np.mean(r2s)), 3) if r2s else float("nan")


def between_city_level_r(run: FoldRun) -> float:
    """Between-city Pearson of city MEANS (predicted vs observed log-rate) — the
    cross-city LEVEL skill an absolute model must have. Isolates the leveller's actual
    job from the within-city scatter that dominates the pooled number."""
    rate_col = _rate_col(run.mode, run.category)
    g = run.scored.copy()
    g["_obs"] = np.log1p(g[rate_col].astype(float))
    means = g.groupby("holdout_city").agg(pred=("y_pred", "mean"), obs=("_obs", "mean"))
    if len(means) < 3:
        return float("nan")
    r = np.corrcoef(means["pred"], means["obs"])[0, 1]
    return round(float(r), 3) if np.isfinite(r) else float("nan")


def between_city_level(run: FoldRun) -> pd.Series:
    """Per-city predicted vs observed mean log-rate, for inspecting *which* cities a
    model mislevels rather than just how badly it does overall."""
    rate_col = _rate_col(run.mode, run.category)
    g = run.scored.copy()
    g["_obs"] = np.log1p(g[rate_col].astype(float))
    return g.groupby("holdout_city").agg(pred=("y_pred", "mean"), obs=("_obs", "mean"))


def compare_runs(runs: dict[str, FoldRun], x_unit: str = "population") -> pd.DataFrame:
    """One row per named run: the headline numbers side by side.

    The comparison table `experiments.run_grid` used to build, minus the spec grid —
    a run is now just a `FoldRun`, so anything comparable can go in the dict regardless
    of which estimator or fold protocol produced it.
    """
    rows = []
    for name, run in runs.items():
        pooled = fold_metrics(run, x_unit=x_unit).loc["POOLED"]
        rows.append({
            "run": name,
            "split": run.split,
            "mode": run.mode,
            "n": int(pooled["n"]),
            "r2_oos": pooled.get("r2_oos"),
            "within_corr2": within_corr2_pooled(run),
            "between_r": between_city_level_r(run),
            "skill": pooled.get("skill"),
            "spearman": pooled.get("spearman"),
            "recall@10": within_city_recall(run, danger_top=0.10,
                                            nets=(0.10,)).loc["POOLED", "recall@top10"],
        })
    return pd.DataFrame(rows).set_index("run")

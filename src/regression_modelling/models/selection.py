"""Stepwise predictor selection under a paired, city-aware stop rule (ADR 0010).

Ridge drives the search (fast, stable); LightGBM confirms the result via ``confirm``.
Every fit is a full ``run_cv`` (LOCO by default), so each decision is judged on
out-of-sample cities.

The single decision is "is the SMALLER set acceptable?" (``StopRule.drop_ok``):

- ``r2_oos`` falls by less than ``r2_tol``, AND
- ``recall@net`` is not worse in at least ``min_share_not_worse`` of held-out cities.

``StopRule.score`` sets how ``r2_oos`` is aggregated: ``"pooled"`` (all rows; New York is
~30% of them and dominates) or ``"city_mean"`` (each held-out city weighted equally).

Backward elimination drops a predictor when that holds; forward selection adds one when it
does NOT (i.e. leaving it out would cost too much). One rule, both directions.

Per-step tolerance accumulates: ten drops of -0.004 each pass the rule yet cost 0.04. For
backward runs ``StopRule.cum_tol`` adds a best-so-far guard — a drop is also refused if
pooled ``r2_oos`` would fall more than ``cum_tol`` below the best run seen on the path.

All candidate fits are scored on the same rows: the pool is reduced up front to rows
complete in every candidate and fixed column, because ``run_cv`` otherwise drops NaNs
per predictor set and two runs would be compared on different BGs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.metrics import city_scores
from regression_modelling.models.results import FoldRun
from regression_modelling.models.training import run_cv

#: Ties within this tolerance count as "not worse" (recall is a share of a small integer).
_EPS = 1e-12


@dataclass(frozen=True)
class PairedDelta:
    """Smaller set minus larger set: ``d_r2`` (pooled r2_oos) and the share of held-out
    cities whose ``recall@net`` did not get worse."""
    d_r2: float
    share_not_worse: float
    n_cities: int


@dataclass(frozen=True)
class StopRule:
    r2_tol: float = 0.005
    min_share_not_worse: float = 0.5
    recall_net: float = 0.10
    #: backward only: max r2_oos shortfall vs the best run on the path (None = off)
    cum_tol: float | None = None
    #: r2_oos aggregation: "pooled" or "city_mean" (see ``score_run``)
    score: str = "pooled"

    def drop_ok(self, delta: PairedDelta) -> bool:
        """True if the smaller set is acceptable versus the larger one."""
        return (delta.d_r2 > -self.r2_tol
                and delta.share_not_worse >= self.min_share_not_worse)


def score_run(run: FoldRun, net: float = 0.10, score: str = "pooled") -> pd.Series:
    """``r2_oos`` and ``recall`` of a run, pooled or as an equal-weight mean over cities."""
    cs = city_scores(run, net)
    if score == "pooled":
        return cs.loc["POOLED", ["r2_oos", "recall"]].astype(float)
    if score == "city_mean":
        return cs.drop("POOLED")[["r2_oos", "recall"]].astype(float).mean()
    raise ValueError("score must be 'pooled' or 'city_mean'")


def paired_delta(smaller: FoldRun, larger: FoldRun, net: float = 0.10,
                 score: str = "pooled") -> PairedDelta:
    """Compare two runs scored on the same rows, city by city. ``d_r2`` is aggregated per
    ``score``; the recall comparison is always per city."""
    a = city_scores(smaller, net)
    b = city_scores(larger, net)
    cities = [c for c in a.index if c != "POOLED" and c in b.index]
    not_worse = (a.loc[cities, "recall"] >= b.loc[cities, "recall"] - _EPS)
    if score == "pooled":
        d_r2 = a.loc["POOLED", "r2_oos"] - b.loc["POOLED", "r2_oos"]
    elif score == "city_mean":
        d_r2 = a.loc[cities, "r2_oos"].mean() - b.loc[cities, "r2_oos"].mean()
    else:
        raise ValueError("score must be 'pooled' or 'city_mean'")
    return PairedDelta(d_r2=float(d_r2),
                       share_not_worse=float(not_worse.mean()) if cities else np.nan,
                       n_cities=len(cities))


@dataclass
class SelectionResult:
    """``selected`` excludes ``fixed``; ``predictors`` is what the final run fit.
    ``path`` has one row per candidate evaluated (``chosen`` marks the step taken)."""
    selected: list[str]
    fixed: list[str]
    path: pd.DataFrame
    final_run: FoldRun
    rule: StopRule
    run_ids: list[str] = field(default_factory=list)

    @property
    def predictors(self) -> list[str]:
        return [*self.fixed, *self.selected]


def common_rows(pool: pd.DataFrame, category: str, columns: Iterable[str],
                mode: str = "lograte") -> pd.DataFrame:
    """Rows complete in every column and the target rate, slimmed to what a fit needs."""
    cols = list(dict.fromkeys(columns))
    missing = [c for c in cols if c not in pool.columns]
    if missing:
        raise KeyError(f"columns not in pool: {missing}")
    rate = _rate_col(mode, category)
    keep = [c for c in dict.fromkeys(["city", "geoid", "population", rate, *cols])
            if c in pool.columns]
    return pool[keep].dropna(subset=[rate, *cols]).reset_index(drop=True)


def _fit(pool, category, predictors, *, estimator, split, mode, gbm_params, seed,
         ridge_alphas) -> FoldRun:
    return run_cv(pool, category, mode=mode, predictors=list(predictors),
                  estimator=estimator, split=split, gbm_params=gbm_params,
                  ridge_alphas=ridge_alphas, seed=seed, keep_fits=False, verbose=False)


def stepwise(pool: pd.DataFrame, category: str, candidates: Iterable[str], *,
             direction: str = "backward", fixed: Iterable[str] = (),
             start: Iterable[str] | None = None, estimator: str = "ridge",
             split: str = "loco", mode: str = "lograte", rule: StopRule = StopRule(),
             gbm_params: dict | None = None, ridge_alphas=None, seed: int = 0,
             max_steps: int | None = None, n_jobs: int = 8,
             log_name: str | None = None, log_root=None,
             verbose: bool = True) -> SelectionResult:
    """Greedy stepwise selection over ``candidates``; ``fixed`` columns are never touched.

    Parameters
    ----------
    direction : ``"backward"`` starts from all candidates and drops, per step, the
        acceptable drop with the largest ``d_r2``. ``"forward"`` starts from ``start``
        (default: nothing beyond ``fixed``) and adds, per step, the candidate whose
        omission the rule rejects with the largest r2 gain. Forward needs at least one
        fixed/start column (e.g. the agency anchor) so the base model can be fit.
    rule : the ``StopRule`` (thresholds are its fields).
    estimator, split, mode, gbm_params, ridge_alphas, seed : passed to ``run_cv``.
        For ``"gbm"``/``"xgb"`` a fixed ``random_state`` (= ``seed``) is added unless given.
    n_jobs : parallel candidate fits per step (joblib processes).
    log_name : if set, every fit is written to the experiment log under this name; only
        the base and chosen-step runs keep their predictions.
    """
    if direction not in ("backward", "forward"):
        raise ValueError("direction must be 'backward' or 'forward'")
    fixed = list(fixed)
    candidates = list(dict.fromkeys(c for c in candidates if c not in set(fixed)))
    if estimator in ("gbm", "xgb"):
        gbm_params = {"random_state": seed, **(gbm_params or {})}
    data = common_rows(pool, category, [*fixed, *candidates], mode)
    fit_kw = dict(estimator=estimator, split=split, mode=mode, gbm_params=gbm_params,
                  seed=seed, ridge_alphas=ridge_alphas)

    backward = direction == "backward"
    if backward:
        current = list(candidates) if start is None else [c for c in start if c in candidates]
    else:
        current = [] if start is None else [c for c in start if c in candidates]
        if not (fixed or current):
            raise ValueError("forward selection needs a non-empty fixed or start set")

    if verbose:
        print(f"{direction} {estimator}/{split} on {category}: {len(data)} common rows, "
              f"{data['city'].nunique()} cities, {len(candidates)} candidates, "
              f"{len(fixed)} fixed")

    run_ids: list[str] = []

    def _log(run, tags, save):
        if log_name is None:
            return
        from regression_modelling.logging.experiments import log_run
        rid = log_run(run, log_name, estimator=estimator,
                      params={"gbm_params": gbm_params, "rule": rule.__dict__,
                              "direction": direction, "seed": seed},
                      save_predictions=save, tags=tags, root=log_root)
        if save:
            run_ids.append(rid)

    base = _fit(data, category, [*fixed, *current], **fit_kw)
    _log(base, {"step": 0, "action": "base"}, True)
    b0 = score_run(base, rule.recall_net, rule.score)
    best_r2 = float(b0.r2_oos)
    if verbose:
        print(f"  step 0  base  n_pred={len(fixed) + len(current):>2}  "
              f"r2[{rule.score}]={b0.r2_oos:.4f}  recall@{rule.recall_net:g}={b0.recall:.3f}")

    rows: list[dict] = []
    step = 0
    while max_steps is None or step < max_steps:
        options = list(current) if backward else [c for c in candidates if c not in current]
        if not options or (backward and not fixed and len(current) == 1):
            break
        step += 1
        trial_sets = ([[*fixed, *[x for x in current if x != c]] for c in options]
                      if backward else [[*fixed, *current, c] for c in options])
        runs = Parallel(n_jobs=max(1, min(n_jobs, len(trial_sets))))(
            delayed(_fit)(data, category, s, **fit_kw) for s in trial_sets)

        step_rows = []
        for c, run in zip(options, runs):
            if backward:
                d = paired_delta(run, base, rule.recall_net, rule.score)
                ok, score = rule.drop_ok(d), d.d_r2
            else:
                d = paired_delta(base, run, rule.recall_net, rule.score)
                ok, score = not rule.drop_ok(d), -d.d_r2
            sc = score_run(run, rule.recall_net, rule.score)
            if backward and rule.cum_tol is not None and sc.r2_oos < best_r2 - rule.cum_tol:
                ok = False
            step_rows.append({"step": step, "action": "drop" if backward else "add",
                              "predictor": c, "d_r2": d.d_r2,
                              "share_not_worse": d.share_not_worse, "eligible": ok,
                              "r2_oos": sc.r2_oos, "recall": sc.recall,
                              "n_predictors": len(run.predictors), "_score": score,
                              "_run": run})

        eligible = [r for r in step_rows if r["eligible"]]
        chosen = max(eligible, key=lambda r: r["_score"]) if eligible else None
        for r in step_rows:
            r["chosen"] = r is chosen
            _log(r["_run"], {"step": step, "action": r["action"],
                             "predictor": r["predictor"]}, r["chosen"])
            rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        if chosen is None:
            if verbose:
                print(f"  step {step}  stop — no {step_rows[0]['action']} passes the rule")
            break
        (current.remove if backward else current.append)(chosen["predictor"])
        base = chosen["_run"]
        best_r2 = max(best_r2, float(chosen["r2_oos"]))
        if verbose:
            print(f"  step {step}  {chosen['action']:<4} {chosen['predictor']:<40} "
                  f"d_r2={chosen['d_r2']:+.4f}  not_worse={chosen['share_not_worse']:.2f}  "
                  f"r2={chosen['r2_oos']:.4f}  recall={chosen['recall']:.3f}")

    return SelectionResult(selected=list(current), fixed=fixed, path=pd.DataFrame(rows),
                           final_run=base, rule=rule, run_ids=run_ids)


def confirm(pool: pd.DataFrame, category: str, full: Iterable[str],
            reduced: Iterable[str], *, fixed: Iterable[str] = (),
            estimator: str = "gbm", split: str = "loco", mode: str = "lograte",
            rule: StopRule = StopRule(), gbm_params: dict | None = None,
            ridge_alphas=None, seed: int = 0) -> dict:
    """Re-check a selection with a second estimator (LightGBM by default).

    Fits ``fixed + full`` and ``fixed + reduced`` on the same common rows and applies the
    same ``rule``. ``adopt_reduced`` True means the reduced set is enough for this estimator
    too (promote one set); False means it costs this estimator too much (promote both).
    """
    fixed, full, reduced = list(fixed), list(full), list(reduced)
    if estimator in ("gbm", "xgb"):
        gbm_params = {"random_state": seed, **(gbm_params or {})}
    data = common_rows(pool, category, [*fixed, *full, *reduced], mode)
    kw = dict(estimator=estimator, split=split, mode=mode, gbm_params=gbm_params,
              seed=seed, ridge_alphas=ridge_alphas)
    run_full, run_red = Parallel(n_jobs=2)(
        delayed(_fit)(data, category, [*fixed, *s], **kw) for s in (full, reduced))
    d = paired_delta(run_red, run_full, rule.recall_net, rule.score)
    scores = pd.DataFrame({"full": score_run(run_full, rule.recall_net, rule.score),
                           "reduced": score_run(run_red, rule.recall_net, rule.score)})
    return {"delta": d, "adopt_reduced": rule.drop_ok(d), "scores": scores,
            "full_run": run_full, "reduced_run": run_red}
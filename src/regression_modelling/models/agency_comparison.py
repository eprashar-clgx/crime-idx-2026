"""National agency-level Ridge comparison: the incumbent's predictor sets vs ours.

Every set is scored with the same harness: a population-weighted ``RidgeCV`` on
``log1p(wtotal_pt_m)`` with standardized predictors, over agencies with
``population >= 1,000`` and complete predictors in every set being compared.

A single 75/25 split moves test r² by about ±0.05, which is as large as the gaps between
some sets. ``repeated_split_scores`` therefore scores every set on the *same* N random
splits and ``summarise_scores`` reports mean ± SD. The incumbent's own split
(``random_state=99999``) is kept as an extra column for continuity.

``stepwise_select`` chooses a subset by population-weighted K-fold CV on the training
agencies only (forward adds / backward drops while CV r² improves by more than ``tol``).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from regression_modelling.data_wrangling.agency import (
    NATIONAL_PREDICTORS,
    TRANSIT_PROXY_PREDICTORS,
    adjusted_r2,
)

TARGET = "wtotal_pt_m"
WEIGHT = "population_m"
POP_CUT = 1000
INCUMBENT_SEED = 99999

# The incumbent national model, progressively stripped by governance (all pre-z-scored
# columns of ct_muni_df).
ORIGINAL_PREDICTORS = [
    "zlg10_pop_est_5mile", "zlg10_jobs_5min", "zin_household_pct", "zcol_pct",
    "zsqrt_moved1yr_pct", "zsqrt_vacant_pct", "zlg10_job_density", "zgini_index",
    "zcity_centers_dist", "zmrpct_val_rm", "encentral", "zusrpct_total_sc", "zsqrt_md_retax",
    "zdet_pct", "southatlantic", "zlg10_dorms_pct", "midatlantic", "zsqrt_unemp_pct",
    "zlg10_grandkids_pct", "zmarried_pct", "znonfam_pct", "zsqrt_singleparent_pct",
]
# Internal-team flag: household-composition demographics.
FLAGGED = ["zlg10_grandkids_pct", "zmarried_pct", "znonfam_pct", "zsqrt_singleparent_pct"]
# Department-disallowed: the 4 flagged + jobs (2), school index, real-estate tax,
# unemployment, home value per room, college, dorms, gini.
DEPT_DISALLOWED = FLAGGED + [
    "zlg10_jobs_5min", "zlg10_job_density", "zsqrt_md_retax", "zusrpct_total_sc",
    "zsqrt_unemp_pct", "zmrpct_val_rm", "zcol_pct", "zlg10_dorms_pct", "zgini_index",
]
# Department-approved features that our national set already represents.
DEPT_REDUNDANT = ["zlg10_pop_est_5mile", "zsqrt_vacant_pct", "zsqrt_moved1yr_pct",
                  "zcity_centers_dist"]

APPROVED_PREDICTORS = [p for p in ORIGINAL_PREDICTORS if p not in FLAGGED]
DEPT_APPROVED_PREDICTORS = [p for p in ORIGINAL_PREDICTORS if p not in DEPT_DISALLOWED]
OUR_PREDICTORS = NATIONAL_PREDICTORS + TRANSIT_PROXY_PREDICTORS
DEPT_PLUS_OURS = [p for p in DEPT_APPROVED_PREDICTORS if p not in DEPT_REDUNDANT] + OUR_PREDICTORS


def comparison_sets() -> dict[str, list[str]]:
    """The fixed rows of the comparison, in reporting order."""
    return {
        f"original ({len(ORIGINAL_PREDICTORS)})": ORIGINAL_PREDICTORS,
        f"approved ({len(APPROVED_PREDICTORS)})": APPROVED_PREDICTORS,
        f"dept-approved ({len(DEPT_APPROVED_PREDICTORS)})": DEPT_APPROVED_PREDICTORS,
        f"ours ({len(OUR_PREDICTORS)})": OUR_PREDICTORS,
        f"dept-approved + ours ({len(DEPT_PLUS_OURS)})": DEPT_PLUS_OURS,
    }


def modelling_rows(muni: pd.DataFrame, predictor_sets: Iterable[Iterable[str]],
                   pop_cut: int = POP_CUT) -> pd.DataFrame:
    """Agencies with a target, ``population >= pop_cut`` and complete predictors in every set."""
    cols = sorted({c for s in predictor_sets for c in s})
    ok = muni[TARGET].notna() & (muni[WEIGHT] >= pop_cut) & muni[cols].notna().all(axis=1)
    return muni[ok]


def make_ridge() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(regressor=RidgeCV(), func=np.log1p, inverse_func=np.expm1)


def fit_ridge(train: pd.DataFrame, predictors: list[str]):
    """Fit scaler + log1p Ridge on ``train``; returns ``(scaler, model)``."""
    scaler = StandardScaler().fit(train[predictors])
    model = make_ridge().fit(scaler.transform(train[predictors]), train[TARGET],
                             sample_weight=train[WEIGHT])
    return scaler, model


def predict(scaler, model, df: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    return model.predict(scaler.transform(df[predictors]))


def evaluate(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]) -> dict:
    """Population-weighted train/test r², test adjusted r² and test MAE for one split."""
    scaler, model = fit_ridge(train, predictors)
    ptr, pte = predict(scaler, model, train, predictors), predict(scaler, model, test, predictors)
    r2_te = r2_score(test[TARGET], pte, sample_weight=test[WEIGHT])
    return dict(
        n_predictors=len(predictors), n_train=len(train), n_test=len(test),
        R2_train=r2_score(train[TARGET], ptr, sample_weight=train[WEIGHT]),
        R2_test=r2_te,
        adjR2_test=adjusted_r2(r2_te, len(test), len(predictors)),
        MAE_test=mean_absolute_error(test[TARGET], pte, sample_weight=test[WEIGHT]),
    )


def repeated_split_scores(df: pd.DataFrame, sets: dict[str, list[str]],
                          seeds: Iterable[int], test_size: float = 0.25) -> pd.DataFrame:
    """Score every set on the same random splits; one row per (seed, set)."""
    rows = []
    for seed in seeds:
        train, test = train_test_split(df, test_size=test_size, random_state=seed)
        for name, preds in sets.items():
            rows.append(dict(seed=seed, set=name, **evaluate(train, test, preds)))
    return pd.DataFrame(rows)


def summarise_scores(scores: pd.DataFrame, incumbent_seed: int = INCUMBENT_SEED) -> pd.DataFrame:
    """Mean ± SD over the random splits, plus the incumbent-seed adjR2 when present."""
    order = list(dict.fromkeys(scores["set"]))
    rep = scores[scores["seed"] != incumbent_seed]
    out = rep.groupby("set").agg(
        n_predictors=("n_predictors", "mean"),
        adjR2_mean=("adjR2_test", "mean"), adjR2_sd=("adjR2_test", "std"),
        R2_test_mean=("R2_test", "mean"), R2_train_mean=("R2_train", "mean"),
        MAE_mean=("MAE_test", "mean"), n_splits=("seed", "nunique"),
    )
    inc = scores[scores["seed"] == incumbent_seed]
    if len(inc):
        out[f"adjR2_seed{incumbent_seed}"] = inc.set_index("set")["adjR2_test"]
    return out.reindex(order)


def weighted_cv_r2(train: pd.DataFrame, predictors: list[str], n_splits: int = 5,
                   seed: int = 0) -> float:
    """Population-weighted out-of-fold r² of the Ridge harness on ``train``."""
    if not predictors:
        return float("-inf")
    oof = np.empty(len(train))
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(train):
        scaler, model = fit_ridge(train.iloc[tr], predictors)
        oof[va] = predict(scaler, model, train.iloc[va], predictors)
    return r2_score(train[TARGET], oof, sample_weight=train[WEIGHT])


def stepwise_select(train: pd.DataFrame, pool: list[str], direction: str = "forward",
                    tol: float = 0.001, n_splits: int = 5,
                    seed: int = 0) -> tuple[list[str], pd.DataFrame]:
    """Greedy CV-r² selection over ``pool``.

    ``forward`` starts empty and adds the best feature while the gain exceeds ``tol``.
    ``backward`` starts from the full pool and drops the feature whose removal helps most,
    stopping when every drop costs more than ``tol``, so a feature must earn at least
    ``tol`` of CV r² to stay. Returns the selected list and a step history.
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
    cv = lambda preds: weighted_cv_r2(train, preds, n_splits=n_splits, seed=seed)  # noqa: E731
    hist = []
    if direction == "forward":
        selected, remaining, best = [], list(pool), float("-inf")
        while remaining:
            score, feat = max(((cv(selected + [c]), c) for c in remaining), key=lambda t: t[0])
            if score - best <= tol:
                break
            selected.append(feat); remaining.remove(feat)
            hist.append(dict(step=len(hist) + 1, feature=feat, action="add", cv_r2=score,
                             delta=score - best if np.isfinite(best) else np.nan,
                             n_selected=len(selected)))
            best = score
        return selected, pd.DataFrame(hist)

    selected = list(pool)
    best = cv(selected)
    hist.append(dict(step=0, feature=None, action="start", cv_r2=best, delta=np.nan,
                     n_selected=len(selected)))
    while len(selected) > 1:
        score, feat = max(((cv([c for c in selected if c != f]), f) for f in selected),
                          key=lambda t: t[0])
        if score < best - tol:
            break
        selected.remove(feat)
        hist.append(dict(step=len(hist), feature=feat, action="drop", cv_r2=score,
                         delta=score - best, n_selected=len(selected)))
        best = score
    return selected, pd.DataFrame(hist)


def select_best_direction(train: pd.DataFrame, pool: list[str], tol: float = 0.001,
                          n_splits: int = 5, seed: int = 0) -> dict:
    """Run forward and backward stepwise; keep the one with the higher final CV r²."""
    runs = {}
    for d in ("forward", "backward"):
        sel, hist = stepwise_select(train, pool, d, tol=tol, n_splits=n_splits, seed=seed)
        runs[d] = dict(direction=d, selected=sel, history=hist, cv_r2=hist["cv_r2"].iloc[-1])
    best = max(runs.values(), key=lambda r: r["cv_r2"])
    return dict(best, runs=runs)


def _nested_one(df, pool, seed, test_size, tol, label):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    pick = select_best_direction(train, pool, tol=tol)
    row = dict(seed=seed, set=label, direction=pick["direction"], cv_r2=pick["cv_r2"],
               cv_r2_forward=pick["runs"]["forward"]["cv_r2"],
               cv_r2_backward=pick["runs"]["backward"]["cv_r2"],
               **evaluate(train, test, pick["selected"]))
    return row, pick["selected"]


def nested_stepwise_scores(df: pd.DataFrame, pool: list[str], seeds: Iterable[int],
                           label: str = "stepwise", test_size: float = 0.25,
                           tol: float = 0.001, n_jobs: int = -1):
    """Re-run the selection inside every split's training agencies, then score on its test.

    Selecting once and re-scoring the subset on other splits would leak (their test agencies
    were in the selection's training data), so the subset is re-chosen per split. Returns
    ``(scores, selections)``: score rows shaped like ``repeated_split_scores`` (plus the
    chosen direction) and a ``{seed: selected features}`` dict.
    """
    from joblib import Parallel, delayed

    seeds = list(seeds)
    out = Parallel(n_jobs=n_jobs)(
        delayed(_nested_one)(df, pool, s, test_size, tol, label) for s in seeds)
    scores = pd.DataFrame([r for r, _ in out])
    return scores, {s: sel for s, (_, sel) in zip(seeds, out)}


def selection_frequency(selections: dict, exclude: Iterable = ()) -> pd.Series:
    """Share of splits in which each feature was selected (excluding ``exclude`` seeds)."""
    lists = [v for k, v in selections.items() if k not in set(exclude)]
    counts = pd.Series([f for sel in lists for f in sel]).value_counts()
    return (counts / len(lists)).rename("share_of_splits")

"""Hyperparameter tuning on LOCO for a frozen predictor set (ADR 0010 §8).

Optuna maximises ``r2_oos`` subject to ``recall@net`` >= the untuned model's, so a tuned
model can never trade top-tier capture for fit. Every trial is a full LOCO run with the
held-out cities fitted in parallel.

``score="pooled"`` judges on all rows together, so New York (~30% of rows) dominates;
``score="city_mean"`` weights each held-out city equally (both r2 and recall).

The folds that judge the trials are the same ones used for selection, so the tuned score
is optimistic; that bias is accepted and disclosed (ADR 0010 §6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import pandas as pd
from joblib import Parallel, delayed

from regression_modelling.models.results import FoldRun
from regression_modelling.models.selection import common_rows, score_run
from regression_modelling.models.training import loco_folds, resolve_predictors, run_cv


def run_loco_parallel(pool: pd.DataFrame, category: str, predictors: Iterable[str], *,
                      estimator: str = "gbm", params: dict | None = None,
                      mode: str = "lograte", n_jobs: int = -1,
                      keep_fits: bool = False) -> FoldRun:
    """``run_cv(split="loco")`` with one process per held-out city. Same output, faster.
    ``keep_fits=True`` returns each held-out city's fitted model (for SHAP / diagnostics);
    off by default because shipping the models back from the workers costs time."""
    preds = resolve_predictors(pool, predictors)
    head = pool.iloc[:0]   # run_cv only reads columns from `pool` when folds are given
    runs = Parallel(n_jobs=n_jobs)(
        delayed(run_cv)(head, category, mode=mode, predictors=preds, estimator=estimator,
                        folds=[fold], gbm_params=params, keep_fits=keep_fits, verbose=False)
        for fold in loco_folds(pool))
    scored = pd.concat([r.scored for r in runs], ignore_index=True)
    fits = {k: v for r in runs for k, v in r.fits.items()}
    return FoldRun(scored=scored, fits=fits, mode=mode, split="loco", category=category,
                   predictors=preds)


def lgbm_space(trial) -> dict:
    """LightGBM search space. Bagging only acts when ``subsample_freq`` > 0, so it is set."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )


def xgb_space(trial) -> dict:
    """XGBoost search space, mirroring ``lgbm_space`` where the knobs line up."""
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 100.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    )


SPACES: dict[str, Callable] = {"gbm": lgbm_space, "xgb": xgb_space}


@dataclass
class TuneResult:
    """``best_params`` is the override merged onto the estimator defaults; it is ``{}``
    when no feasible trial beat the untuned model (keep the defaults)."""
    estimator: str
    predictors: list[str]
    best_params: dict
    scores: pd.DataFrame           # r2_oos / recall under `score`: untuned vs tuned
    untuned_run: FoldRun
    tuned_run: FoldRun
    trials: pd.DataFrame
    study: object = field(repr=False)


def tune(pool: pd.DataFrame, category: str, predictors: Iterable[str], *,
         estimator: str = "gbm", n_trials: int = 50, net: float = 0.10,
         score: str = "pooled", mode: str = "lograte", seed: int = 0, n_jobs: int = -1,
         space: Callable | None = None, log_name: str | None = None,
         verbose: bool = True) -> TuneResult:
    """Tune ``estimator`` on LOCO: max ``r2_oos`` s.t. ``recall@net`` >= untuned, both
    aggregated per ``score`` (``"pooled"`` or ``"city_mean"``).

    All trials score the same rows (``common_rows``). The untuned model (estimator
    defaults) is fit first; its recall is the constraint floor. If ``log_name`` is set the
    untuned and tuned runs are written to the experiment log with predictions.
    """
    import optuna

    space = space or SPACES[estimator]
    preds = list(dict.fromkeys(predictors))
    data = common_rows(pool, category, preds, mode)
    fit = lambda p: run_loco_parallel(data, category, preds, estimator=estimator,
                                      params=p, mode=mode, n_jobs=n_jobs)

    base_params = {"random_state": seed}
    untuned = fit(base_params)
    u = score_run(untuned, net, score)
    floor = float(u.recall)
    if verbose:
        print(f"untuned {estimator} [{score}]: r2={u.r2_oos:.4f}  recall@{net:g}={floor:.3f}  "
              f"({len(data)} rows, {len(preds)} predictors)")

    def objective(trial):
        run = fit({**space(trial), "random_state": seed})
        p = score_run(run, net, score)
        trial.set_user_attr("recall", float(p.recall))
        trial.set_user_attr("constraint", floor - float(p.recall))
        trial.set_constraint("recall_floor", floor - float(p.recall))   # <= 0 is feasible
        return float(p.r2_oos)

    def _report(study, trial):
        if verbose:
            ok = "ok " if trial.user_attrs["constraint"] <= 1e-12 else "low"
            print(f"  trial {trial.number:>3}  r2={trial.value:.4f}  "
                  f"recall={trial.user_attrs['recall']:.3f} [{ok}]")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, callbacks=[_report])

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    feasible = [t for t in study.trials
                if t.value is not None and t.user_attrs["constraint"] <= 1e-12]
    best = max(feasible, key=lambda t: t.value, default=None)
    if best is not None and best.value > u.r2_oos:
        best_params = {**space(optuna.trial.FixedTrial(best.params)), "random_state": seed}
        tuned = fit(best_params)
    else:
        best_params, tuned = {}, untuned

    scores = pd.DataFrame({"untuned": u, "tuned": score_run(tuned, net, score)})
    if verbose:
        t = scores["tuned"]
        print(f"tuned   {estimator} [{score}]: r2={t.r2_oos:.4f}  recall@{net:g}={t.recall:.3f}"
              + ("" if best_params else "  (no feasible trial beat untuned: keep defaults)"))

    if log_name is not None:
        from regression_modelling.logging.experiments import log_run
        log_run(untuned, log_name, estimator=estimator, params=base_params,
                tags={"stage": "untuned", "score": score})
        if best_params:
            log_run(tuned, log_name, estimator=estimator, params=best_params,
                    tags={"stage": "tuned", "n_trials": n_trials, "score": score})

    return TuneResult(estimator=estimator, predictors=preds, best_params=best_params,
                      scores=scores, untuned_run=untuned, tuned_run=tuned, trials=trials,
                      study=study)

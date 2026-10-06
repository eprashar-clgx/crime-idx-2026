"""Synthetic tests for the XGBoost estimator and LOCO tuning (ADR 0010 §8)."""
import pytest

from regression_modelling.models.metrics import city_scores
from regression_modelling.models.training import run_cv
from regression_modelling.models.tuning import run_loco_parallel, tune

from test_predictor_selection import _pool

PREDS = ["anchor", "x1", "x2", "x3"]


@pytest.mark.parametrize("est", ["gbm", "xgb"])
def test_tree_estimators_learn_signal(est):
    run = run_cv(_pool(), "wprop", predictors=PREDS, estimator=est, verbose=False,
                 gbm_params={"n_estimators": 100})
    assert city_scores(run).loc["POOLED", "r2_oos"] > 0.4


def test_parallel_loco_matches_serial():
    df, p = _pool(), {"n_estimators": 50, "random_state": 0}
    serial = run_cv(df, "wprop", predictors=PREDS, estimator="gbm", gbm_params=p,
                    verbose=False)
    par = run_loco_parallel(df, "wprop", PREDS, params=p, n_jobs=2)
    key = ["geoid", "y_pred"]
    a = serial.scored[key].sort_values("geoid").reset_index(drop=True)
    b = par.scored[key].sort_values("geoid").reset_index(drop=True)
    assert a.equals(b)


def test_tune_respects_recall_floor():
    res = tune(_pool(), "wprop", PREDS, n_trials=3, n_jobs=1, verbose=False)
    s = res.scores
    assert s.loc["r2_oos", "tuned"] >= s.loc["r2_oos", "untuned"]
    assert s.loc["recall", "tuned"] >= s.loc["recall", "untuned"] - 1e-12
    assert len(res.trials) == 3
    if res.best_params:
        assert res.best_params["subsample_freq"] == 1


def test_tune_city_mean_score():
    res = tune(_pool(), "wprop", PREDS, n_trials=2, n_jobs=1, score="city_mean",
               verbose=False)
    cs = city_scores(res.untuned_run).drop("POOLED")
    assert res.scores.loc["r2_oos", "untuned"] == pytest.approx(cs["r2_oos"].mean())
    with pytest.raises(ValueError):
        tune(_pool(), "wprop", PREDS, n_trials=1, n_jobs=1, score="nope", verbose=False)

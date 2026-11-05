import numpy as np
import pandas as pd
import pytest

from regression_modelling.models.attribution import (
    FULL, city_feature_table, dropout_deltas, family_direction, family_dropout,
    family_level_push)
from regression_modelling.models.diagnostics import loco_shap
from regression_modelling.models.tuning import run_loco_parallel

PARAMS = dict(n_estimators=60, learning_rate=0.1, num_leaves=8, min_child_samples=10,
              verbose=-1)
# Named so feature_family maps signal -> transit, weak -> roadway, noise -> stores.
PREDS = ["transit_x1", "roadway_x2", "unq_x3", "agency_lag_wprop_log"]


def _pool(n_city=5, n=150, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_city):
        x1, x2, x3 = rng.normal(size=(3, n))
        level = rng.normal(scale=0.3)
        log_rate = 2 + level + 0.8 * x1 + 0.3 * x2 + rng.normal(scale=0.4, size=n)
        parts.append(pd.DataFrame({
            "city": f"c{i}", "geoid": [f"c{i}_{j}" for j in range(n)], "population": 1000,
            "transit_x1": x1, "roadway_x2": x2, "unq_x3": x3,
            "agency_lag_wprop_log": 2 + level, "wprop_rate": np.expm1(log_rate)}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def runs():
    df = _pool()
    df.loc[df.sample(40, random_state=1).index, "unq_x3"] = np.nan
    return family_dropout(df, "wprop", PREDS, params=PARAMS, seeds=(0, 1), n_jobs=1)


def test_dropout_scores_same_rows_and_ranks_signal_first(runs):
    n = {k: r.n_scored for k, r in runs.items()}
    assert len(set(n.values())) == 1                      # dropping stores lets no NaN rows in
    d = dropout_deltas(runs)
    assert d.loc["transit", "d_r2_pooled"] > 0.2
    assert d.loc["transit", "d_r2_pooled"] > d.loc["stores", "d_r2_pooled"] + 0.2
    assert d.loc["transit", "cities_helped"] == 5
    assert {"sd_r2_pooled", "d_spearman_city"} <= set(d.columns)


def test_direction_and_feature_table():
    run = run_loco_parallel(_pool(), "wprop", PREDS, params=PARAMS, n_jobs=1, keep_fits=True)
    shap = loco_shap(run)
    direc = family_direction(shap, run.scored, "wprop")
    assert "agency anchor" not in direc.columns
    assert (direc["transit"] > 0.5).all()

    push = family_level_push(shap, run)
    fams = [c for c in push if c not in ("_base", "obs_mean", "level_err")]
    mean_pred = run.scored.groupby("holdout_city")["y_pred"].mean()
    np.testing.assert_allclose(push[fams].sum(axis=1) + push["_base"], mean_pred, atol=1e-6)

    t = city_feature_table(run, shap, "c0", n_rep=3)
    assert t.index[0] == "transit_x1"
    assert t.loc["transit_x1", "perm_d_r2"] > 0.3
    assert t.loc["transit_x1", "agrees"]
    assert np.isnan(t.loc["agency_lag_wprop_log", "perm_d_r2"])   # city-constant


def test_relevel_moves_level_not_ranking():
    from dataclasses import replace
    from regression_modelling.models import levelling
    from regression_modelling.models.metrics import scorecard
    run = run_loco_parallel(_pool(), "wprop", PREDS[:3], params=PARAMS, n_jobs=1)
    s = run.scored.copy()
    s["y_pred"] += np.where(s["holdout_city"] == "c0", 0.8, 0.0)       # mis-level one city
    bad = replace(run, scored=s)
    stats = levelling.city_stats(bad, pd.Series(dtype=float))
    anchor = np.expm1(stats["obs_log"])                                   # exact on log scale
    fixed = levelling.relevel(bad, anchor, "loglin")
    half = levelling.relevel(bad, anchor, "oracle", weight=0.5)
    before, after = scorecard(bad), scorecard(fixed)
    assert after["r2_pooled"] > before["r2_pooled"] + 0.1
    assert after["spearman_city"] == pytest.approx(before["spearman_city"])
    lvl = lambda r: levelling.city_stats(r, anchor).eval("pred_log - obs_log")["c0"]
    assert abs(lvl(half)) == pytest.approx(abs(lvl(bad)) / 2)

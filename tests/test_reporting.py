"""Synthetic tests for the 02 reporting layer: scorecard, held-out SHAP, figures."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from regression_modelling.constants import PREDICTOR_SETS
from regression_modelling.models import diagnostics, figures
from regression_modelling.models.metrics import city_scores, scorecard, scorecard_by_city
from regression_modelling.models.results import FoldRun
from regression_modelling.models.training import TUNED_GBM_PARAMS
from regression_modelling.models.tuning import run_loco_parallel

from test_predictor_selection import _pool

PREDS = ["anchor", "x1", "x2", "x3"]


@pytest.fixture(scope="module")
def run():
    return run_loco_parallel(_pool(), "wprop", PREDS, n_jobs=2, keep_fits=True,
                             params={"n_estimators": 60, "random_state": 0})


def test_scorecard_aggregations_match_city_scores(run):
    card, cs = scorecard(run), city_scores(run, 0.10)
    assert card["r2_pooled"] == pytest.approx(cs.loc["POOLED", "r2_oos"])
    assert card["r2_city"] == pytest.approx(cs.drop("POOLED")["r2_oos"].mean())
    assert card["recall@10"] == pytest.approx(cs.loc["POOLED", "recall"])
    assert card["recall@10_city"] == pytest.approx(cs.drop("POOLED")["recall"].mean())
    assert card["n_cities"] == 6 and 0 <= card["called_safe"] <= 1


def test_scorecard_accepts_incumbent_rejects_other_modes(run):
    inc = FoldRun(scored=run.scored, fits={}, mode="lograte_within_city",
                  split="existing", category="wprop", predictors=[])
    assert np.isfinite(scorecard(inc)["r2_pooled"])
    with pytest.raises(ValueError):
        scorecard(FoldRun(scored=run.scored, fits={}, mode="rate_within_city",
                          split="loco", category="wprop", predictors=[]))


def test_parallel_keep_fits(run):
    assert sorted(run.fits) == sorted(run.scored["holdout_city"].unique())


def test_loco_shap_is_additive_and_out_of_fold(run):
    sh = diagnostics.loco_shap(run)
    recon = sh.drop(columns="holdout_city").sum(axis=1)
    assert np.allclose(recon, run.scored.loc[sh.index, "y_pred"], atol=1e-8)
    assert (sh["holdout_city"] == run.scored.loc[sh.index, "holdout_city"]).all()
    imp = diagnostics.shap_importance(sh)
    assert imp.index[0] == "x1" and imp["x3"] < imp["x2"]


def test_loco_shap_needs_fits(run):
    with pytest.raises(ValueError):
        diagnostics.loco_shap(FoldRun(scored=run.scored, fits={}, mode="lograte",
                                      split="loco", category="wprop", predictors=PREDS))


def test_grouped_shap_sums_within_family():
    sh = pd.DataFrame({"transit_a": [1.0, -1.0], "transit_b": [-1.0, 2.0],
                       "own_pct": [0.5, 0.5], "_base": 0.0, "holdout_city": ["c", "c"]})
    fam = diagnostics.shap_importance(sh, by_family=True)
    assert fam["transit"] == pytest.approx(0.5)       # |1-1| and |-1+2| -> mean 0.5
    assert fam["demographic"] == pytest.approx(0.5)


@pytest.mark.parametrize("col, fam", [
    ("agency_lag_wprop_log", "agency anchor"), ("own_pct_nbr", "neighbourhood"),
    ("unq_gas_stations_clips_nbr_log", "neighbourhood"), ("unq_gas_stations_clips", "stores"),
    ("transit_has_transit", "transit"), ("roadway_x_ratio", "roadway"),
    ("clip_foreclosure_pct_log", "property distress"), ("vacant_pct_log", "property distress"),
    ("roof_condition_avg", "imagery"), ("lap_pct", "demographic")])
def test_feature_family(col, fam):
    assert diagnostics.feature_family(col) == fam


def test_selected_v1_has_a_tuned_params_entry():
    assert len(PREDICTOR_SETS["selected_v1"]) == len(set(PREDICTOR_SETS["selected_v1"])) == 34
    assert "det_pct" not in PREDICTOR_SETS["selected_v1"]
    assert TUNED_GBM_PARAMS["selected_v1"]["subsample_freq"] == 1


def test_city_frame_outcomes_partition_top_tier(run):
    d = figures.city_frame(run, "c0", net=0.10)
    n = d["top_outcome"].value_counts()
    observed_top = (d["wprop_rate"].rank(pct=True) >= 0.9).sum()    # recall's definition
    assert n.get("hit", 0) + n.get("missed", 0) == observed_top
    assert n.get("hit", 0) + n.get("false alarm", 0) == observed_top
    assert d["obs_decile"].between(1, 10).all()


def test_figures_render(run, tmp_path):
    sh = diagnostics.loco_shap(run)
    figs = {
        "bar": figures.shap_bar(diagnostics.shap_importance(sh)),
        "fam": figures.family_bar(diagnostics.shap_importance(sh, by_family=True)),
        "famcity": figures.family_by_city(
            diagnostics.shap_importance(sh, by_family=True, by_city=True), ["c0", "c1"]),
        "scatter": figures.pred_vs_obs(run, ["c0", "c1", "c2"]),
        "metrics": figures.metric_bars(pd.DataFrame([scorecard(run)], index=["m"])),
    }
    for name, fig in figs.items():
        assert figures.save(fig, name, tmp_path).exists()


def test_scorecard_by_city_matches_city_scores(run):
    by, cs = scorecard_by_city(run), city_scores(run, 0.10).drop("POOLED")
    assert np.allclose(by.loc[cs.index, "r2"], cs["r2_oos"])
    assert np.allclose(by.loc[cs.index, "recall@10"], cs["recall"])
    assert "r2_city" not in by and "recall@10_city" not in by


def test_city_diag_levelled_r2_bounds_r2(run):
    d = diagnostics.city_diag(run, _pool())
    assert (d["r2_levelled"] >= d["r2_oos"] - 1e-9).all()

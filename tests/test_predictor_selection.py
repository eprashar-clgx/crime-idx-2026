"""Synthetic tests for the ADR 0010 selection toolkit: screening, stop rule, experiment
log round-trip and stepwise search. No real data needed."""
import numpy as np
import pandas as pd
import pytest

from regression_modelling.distributions.screening import between_share, screen_predictors
from regression_modelling.logging.experiments import load_run, load_runs, log_run
from regression_modelling.models.metrics import city_scores
from regression_modelling.models.results import FoldRun
from regression_modelling.models.selection import (PairedDelta, StopRule, paired_delta,
                                                   stepwise)


def _pool(n_city=6, n=150, seed=0):
    """y = x1 + x2 + noise per city; x3 pure noise; anchor = city-level constant."""
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(n_city):
        x1, x2, x3 = rng.normal(size=(3, n))
        level = rng.normal(scale=0.3)
        log_rate = 2 + level + 0.8 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n)
        parts.append(pd.DataFrame({
            "city": f"c{i}", "geoid": [f"c{i}_{j}" for j in range(n)],
            "population": 1000, "x1": x1, "x2": x2, "x3": x3,
            "x1_dup": x1 + rng.normal(scale=0.05, size=n),
            "sparse": (rng.random(n) > 0.98).astype(float),
            "cityish": i + rng.normal(scale=0.05, size=n),
            "anchor": 2 + level,
            "wprop_rate": np.expm1(log_rate)}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- screening
def test_between_share_extremes():
    df = _pool()
    s = between_share(df, ["x3", "cityish"])
    assert s["x3"] < 0.1 and s["cityish"] > 0.9


def test_screen_verdicts():
    df = _pool()
    tbl = screen_predictors(df, ["x1", "x2", "x3", "x1_dup", "sparse", "cityish"],
                            "wprop_rate", fixed=["anchor"], artifacts=("cityish",))
    v = tbl["verdict"]
    assert v["anchor"] == "fixed"
    assert v["x1"] == "keep" and v["x2"] == "keep"
    assert v["x3"] == "drop" and "weak" in tbl.loc["x3", "reason"]
    assert v["sparse"] == "drop" and "sparse" in tbl.loc["sparse", "reason"]
    assert v["cityish"] == "drop" and "artifact" in tbl.loc["cityish", "reason"]
    # exactly one of the near-duplicate pair survives, flagged against the other
    assert (v[["x1", "x1_dup"]] == "keep").sum() == 1
    assert "redundant" in tbl.loc["x1_dup", "reason"]


def test_screen_city_identifying_note_not_drop():
    df = _pool()
    tbl = screen_predictors(df, ["x1", "cityish"], "wprop_rate", artifacts=())
    assert "city_identifying" in tbl.loc["cityish", "note"]
    assert "artifact" not in tbl.loc["cityish", "reason"]


def test_screen_missing_column_raises():
    with pytest.raises(KeyError):
        screen_predictors(_pool(), ["nope"], "wprop_rate")


# ---------------------------------------------------------------- stop rule
@pytest.mark.parametrize("d_r2,share,ok", [
    (-0.004, 0.5, True),     # small loss, half not worse
    (-0.006, 1.0, False),    # r2 loss too big
    (0.010, 0.4, False),     # r2 fine but recall worse in most cities
    (0.000, 0.5, True),
])
def test_stop_rule(d_r2, share, ok):
    assert StopRule().drop_ok(PairedDelta(d_r2, share, 10)) is ok


def _run(df, pred):
    s = df.assign(y_pred=pred, holdout_city=df["city"])
    return FoldRun(scored=s, mode="lograte", split="loco", category="wprop",
                   predictors=["x1"])


def test_paired_delta_and_city_scores():
    df = _pool()
    truth = np.log1p(df["wprop_rate"])
    perfect, noisy = _run(df, truth), _run(df, truth + np.random.default_rng(1)
                                                  .normal(scale=2, size=len(df)))
    cs = city_scores(perfect)
    assert cs.loc["POOLED", "r2_oos"] == pytest.approx(1.0)
    assert (cs["recall"] == 1.0).all()
    d = paired_delta(noisy, perfect)
    assert d.d_r2 < -0.5 and d.n_cities == 6
    assert paired_delta(perfect, perfect).share_not_worse == 1.0


# ---------------------------------------------------------------- experiment log
def test_log_round_trip(tmp_path):
    df = _pool()
    run = _run(df, np.log1p(df["wprop_rate"]) * 0.9)
    rid = log_run(run, "unit", estimator="ridge", params={"a": 1}, root=tmp_path)
    log_run(run, "unit", estimator="ridge", save_predictions=False, root=tmp_path)
    runs = load_runs("unit", root=tmp_path)
    assert len(runs) == 2 and runs["predictors"].iloc[0] == ["x1"]
    back = load_run(rid, df, root=tmp_path)
    assert back.n_scored == run.n_scored
    assert city_scores(back).equals(city_scores(run))


# ---------------------------------------------------------------- stepwise
def test_backward_drops_noise_keeps_signal():
    res = stepwise(_pool(), "wprop", ["x1", "x2", "x3"], direction="backward",
                   fixed=["anchor"], n_jobs=1, verbose=False)
    assert set(res.selected) == {"x1", "x2"}
    assert res.predictors[0] == "anchor"
    assert res.path["chosen"].sum() == 1


def test_forward_adds_signal_only(tmp_path):
    res = stepwise(_pool(), "wprop", ["x1", "x2", "x3"], direction="forward",
                   fixed=["anchor"], n_jobs=1, verbose=False,
                   log_name="fwd", log_root=tmp_path)
    assert res.selected == ["x1", "x2"]
    runs = load_runs("fwd", root=tmp_path)
    assert runs["has_predictions"].sum() == len(res.run_ids) == 3   # base + 2 adds
    assert len(runs) == 1 + 3 + 2 + 1   # base, step1 x3, step2 x2, step3 x1


def test_forward_needs_a_base():
    with pytest.raises(ValueError):
        stepwise(_pool(), "wprop", ["x1"], direction="forward", verbose=False)

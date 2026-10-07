"""Synthetic tests for the agency-level Ridge comparison harness."""
import numpy as np
import pandas as pd
import pytest

from regression_modelling.models import agency_comparison as ac


def _agencies(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 5)), columns=["s1", "s2", "n1", "n2", "n3"])
    df[ac.TARGET] = np.expm1(1.5 + 0.6 * df.s1 - 0.4 * df.s2 + rng.normal(0, 0.3, n))
    df[ac.WEIGHT] = rng.integers(ac.POP_CUT, 50_000, n)
    return df


def test_modelling_rows_filters_population_target_and_missing():
    df = _agencies()
    df.loc[0, ac.WEIGHT] = 10
    df.loc[1, ac.TARGET] = np.nan
    df.loc[2, "n3"] = np.nan
    out = ac.modelling_rows(df, [["s1"], ["n3"]])
    assert not {0, 1, 2} & set(out.index)
    assert len(ac.modelling_rows(df, [["s1"]])) == len(df) - 2


def test_signal_beats_noise_on_repeated_splits():
    sets = {"signal": ["s1", "s2"], "noise": ["n1", "n2", "n3"]}
    scores = ac.repeated_split_scores(_agencies(), sets, seeds=[ac.INCUMBENT_SEED, 0, 1, 2])
    summ = ac.summarise_scores(scores)
    assert list(summ.index) == ["signal", "noise"]
    assert summ.loc["signal", "n_splits"] == 3
    assert summ.loc["signal", "adjR2_mean"] > 0.5 > summ.loc["noise", "adjR2_mean"]
    assert f"adjR2_seed{ac.INCUMBENT_SEED}" in summ


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_stepwise_keeps_the_signal(direction):
    sel, hist = ac.stepwise_select(_agencies(), ["n1", "s1", "n2", "s2", "n3"], direction,
                                   tol=0.002)
    assert {"s1", "s2"} <= set(sel)
    assert not {"n1", "n2", "n3"} & set(sel)
    assert hist["cv_r2"].iloc[-1] == pytest.approx(ac.weighted_cv_r2(_agencies(), sel))


def test_nested_stepwise_reselects_per_split():
    scores, sels = ac.nested_stepwise_scores(_agencies(), ["s1", "s2", "n1"], seeds=[0, 1],
                                             tol=0.002, n_jobs=1)
    assert list(scores["seed"]) == [0, 1]
    assert set(scores["direction"]) <= {"forward", "backward"}
    freq = ac.selection_frequency(sels)
    assert freq["s1"] == freq["s2"] == 1.0


def test_dept_plus_ours_drops_redundant_and_disallowed():
    assert not set(ac.DEPT_REDUNDANT) & set(ac.DEPT_PLUS_OURS)
    assert not set(ac.DEPT_DISALLOWED) & set(ac.DEPT_PLUS_OURS)
    assert len(ac.DEPT_APPROVED_PREDICTORS) == 9
    assert "zlg10_jobs_5min" not in ac.DEPT_APPROVED_PREDICTORS

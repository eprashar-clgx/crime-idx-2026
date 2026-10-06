"""Synthetic tests for neighbourhood context and street-morphology features. No real data."""
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from libpysal.weights import Queen
from shapely.geometry import Point, box

from regression_modelling.data_wrangling.neighbourhood import neighbour_aggregate
from regression_modelling.data_wrangling.roadway.build import street_morphology
from regression_modelling.data_wrangling.roadway.tiger import classify_nodes
from regression_modelling.feature_engineering.transforms import apply_transforms


def _grid():
    """3 unit squares in a row (A-B-C) plus a detached island D."""
    return gpd.GeoDataFrame({"geoid": list("ABCD")},
                            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1),
                                      box(10, 10, 11, 11)])


def test_neighbour_mean_skips_nan_and_self():
    g = _grid()
    W = Queen.from_dataframe(g, use_index=False, silence_warnings=True).sparse.tocsr()
    vals = pd.DataFrame({"own_pct": [10.0, np.nan, 30.0, 99.0]})
    out = neighbour_aggregate(W, vals, "mean")["own_pct_nbr"]
    assert np.isnan(out[0])          # A's only neighbour B is NaN -> no valid neighbour
    assert out[1] == 20.0            # B: mean(A, C), self excluded
    assert np.isnan(out[3])          # island


def test_neighbour_sum_treats_nan_as_zero():
    g = _grid()
    W = Queen.from_dataframe(g, use_index=False, silence_warnings=True).sparse.tocsr()
    vals = pd.DataFrame({"stores": [1.0, np.nan, 2.0, 5.0]})
    out = neighbour_aggregate(W, vals, "sum")["stores_nbr"]
    assert out.tolist() == [0.0, 3.0, 0.0, 0.0]


def test_classify_nodes():
    nodes = pd.DataFrame({"deg_street": [1, 1, 2, 3, 4, 5], "deg_all": [1, 2, 2, 3, 4, 6]})
    k = classify_nodes(nodes)
    assert k["is_dead"].tolist() == [True, False, False, False, False, False]  # walkway continues
    assert k["is_t"].tolist() == [False, False, False, True, False, False]
    assert k["is_x"].tolist() == [False, False, False, False, True, True]


def test_street_morphology_ratios_and_nan():
    bg = gpd.GeoDataFrame({"geoid": ["A", "B", "C"]},
                          geometry=[box(0, 0, 10, 10), box(10, 0, 20, 10), box(20, 0, 30, 10)])
    # A: 1 X, 1 T, 2 dead ends. B: only dead end + mid-street node. C: nothing.
    pts = [(1, 1, 4, 4), (2, 2, 3, 3), (3, 3, 1, 1), (4, 4, 1, 1), (12, 2, 1, 1), (13, 3, 2, 2)]
    nodes = gpd.GeoDataFrame({"deg_street": [p[2] for p in pts], "deg_all": [p[3] for p in pts]},
                             geometry=[Point(p[0], p[1]) for p in pts])
    m = street_morphology(bg, nodes)
    assert m.loc["A", "roadway_x_ratio"] == 0.5
    assert m.loc["A", "roadway_deadend_share"] == 0.5
    assert np.isnan(m.loc["B", "roadway_x_ratio"]) and m.loc["B", "roadway_deadend_share"] == 1.0
    assert m.loc["C"].isna().all()


def test_sqrt_transform():
    df = pd.DataFrame({"s": [0.0, 0.25, -0.1]})
    out, cols = apply_transforms(df, spec={"s": "sqrt"}, has_transit_from=None)
    assert cols == ["s_sqrt"]
    assert out["s_sqrt"].tolist() == pytest.approx([0.0, 0.5, 0.0])

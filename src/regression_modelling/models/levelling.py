"""Post-model city levelling: reconcile each held-out city to its agency's reported crime.

The model ranks block groups; this step only moves a city's predictions up or down
together, so within-city order (Spearman, recall) is unchanged and only level-sensitive
scores (r2, MAE) move. It runs after prediction, per city, using one observed number per
city — the lagged UCR agency rate (`lagged_agency.build_lagged_agency_anchor`, the same
number the agency-anchor feature carries, on the target's weighted scale).

Our aggregated BG rate is not the agency rate 1:1 (anchor year precedes the target year,
some incidents are lost in BG geocoding), so every method except ``naive`` learns the
mapping from the OTHER cities only — the held-out city contributes nothing but its anchor,
as it would in a new market.

Methods (`level_targets`):

- ``naive``  — scale rates so the city's population-weighted mean equals the anchor.
- ``ratio``  — the same, with the anchor times the median BG-to-agency ratio of the
  training cities.
- ``loglin`` — shift log predictions so their city mean equals ``a + b * anchor_log``,
  fitted by OLS on the training cities' observed mean log rates.
- ``oracle`` — shift to the city's own observed mean log rate. Not servable (it uses the
  answer); the ceiling for any leveller.

`relevel(..., weight=w)` moves only a fraction ``w`` of the way to the target. On the POC
pools a half-way ``ratio`` blend beats the full move (fewer cities hurt when the anchor
and our incident data disagree, e.g. Oakland and DC).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.results import FoldRun

METHODS = ("naive", "ratio", "loglin", "oracle")


def city_stats(run: FoldRun, anchor_rate: pd.Series) -> pd.DataFrame:
    """Per held-out city: observed population-weighted mean rate, observed mean log rate,
    mean predicted log rate, predicted population-weighted mean rate and the anchor."""
    s = run.scored
    rate = s[_rate_col(run.mode, run.category)].astype(float)
    pop = s["population"].astype(float)
    pred_rate = np.expm1(s["y_pred"].astype(float))
    g = s["holdout_city"]
    out = pd.DataFrame({
        "obs_rate": (rate * pop).groupby(g).sum() / pop.groupby(g).sum(),
        "obs_log": np.log1p(rate).groupby(g).mean(),
        "pred_log": s["y_pred"].astype(float).groupby(g).mean(),
        "pred_rate": (pred_rate * pop).groupby(g).sum() / pop.groupby(g).sum(),
    })
    out["anchor_rate"] = anchor_rate.reindex(out.index).astype(float)
    return out


def level_targets(stats: pd.DataFrame, method: str) -> pd.Series:
    """Per-city level target: a rate (``naive``, ``ratio``) or a mean log rate
    (``loglin``, ``oracle``). Every learned mapping is fitted leaving the city out."""
    if method == "naive":
        return stats["anchor_rate"]
    if method == "oracle":
        return stats["obs_log"]
    out = {}
    for c in stats.index:
        tr = stats.drop(index=c).dropna(subset=["anchor_rate"])
        a = stats.loc[c, "anchor_rate"]
        if method == "ratio":
            out[c] = a * float((tr["obs_rate"] / tr["anchor_rate"]).median())
        elif method == "loglin":
            b1, b0 = np.polyfit(np.log1p(tr["anchor_rate"]), tr["obs_log"], 1)
            out[c] = b0 + b1 * np.log1p(a)
        else:
            raise ValueError(f"method must be one of {METHODS}")
    return pd.Series(out)


def relevel(run: FoldRun, anchor_rate: pd.Series, method: str,
            weight: float = 1.0) -> FoldRun:
    """``run`` with every held-out city's predictions moved to its level target. Rate
    methods multiply the city's predicted rates by one factor; log methods add one shift
    to its log predictions. ``weight`` < 1 moves only that fraction of the way (in log
    space), blending the model's own level with the anchor's. Cities without an anchor
    are left as served."""
    stats = city_stats(run, anchor_rate)
    target = level_targets(stats, method)
    s = run.scored.copy()
    g = s["holdout_city"]
    y = s["y_pred"].astype(float)
    if method in ("naive", "ratio"):
        k = (target / stats["pred_rate"]).reindex(g).to_numpy()
        new = np.log1p(np.expm1(y) * k)
    else:
        new = y + (target - stats["pred_log"]).reindex(g).to_numpy()
    s["y_pred"] = np.where(np.isfinite(new), y + weight * (new - y), y)
    return replace(run, scored=s)

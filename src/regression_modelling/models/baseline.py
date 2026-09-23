"""Incumbent agency-scale model baseline — the benchmark the refresh must beat.

`04_bg_comparison` established the incumbent's scale case: it levels whole cities almost
perfectly (between-city) but fits weakly *within* a city (95% of the variance). This module
reproduces that panel as a **run-like dict** so the prediction notebook can score the
incumbent with the *exact same* helpers (`within_city_recall`, `loco_metrics`) it uses for
the refresh — guaranteeing an apples-to-apples, side-by-side comparison rather than two
hand-rolled metric paths.

The incumbent per-BG prediction ships in ``data/interim/bg_crime/{city}.parquet`` as a
predicted weighted crime rate (``total_pt_ct`` for the total composite, ``property_pt_ct`` for
the property composite) — a linear risk model that down-scales agency crime to the block group,
calibrated so each agency's population-weighted mean matches its agency rate. It is compared,
*within a city*, against the observed weighted relative-risk rate (``compute_weighted_scores``
→ ``wtotal_rate`` / ``wprop_rate``) — the same target the refresh models. Both are the same
kind of weighted rate, but the incumbent carries a per-city level offset (it is agency-anchored,
the observed rate is national-composite-anchored), so comparisons here are rank/level-based
(Pearson/Spearman, within-city percentile recall), which are invariant to that offset.

The incumbent is a single *deployed* model, so it has no LOCO / 80-20 split — one fixed set of
numbers that sits beside every refresh protocol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crime_blockgroup_mapping.config import INTERIM_DIR
from crime_blockgroup_mapping.constants import CITIES
from crime_blockgroup_mapping.scores import compute_weighted_scores, PRIMARY_CRIMES

# Incumbent prediction column per target family (predicted weighted crime rate).
EXISTING_PRED_COL = {"wtotal": "total_pt_ct", "wprop": "property_pt_ct"}
# Drop a handful of tiny-population BGs whose count/pop rates explode (matches 04).
POP_FLOOR = 250


def _default_cities(category: str) -> list[str]:
    """The incumbent is scored on the same pool the refresh is trained on, so the
    comparison is like-for-like. Delegates to the single pool owner (ADR 0008) rather
    than re-deriving eligibility here — this previously kept its own copy of the rule and
    silently included cities with degenerate all-zero crime extracts."""
    from regression_modelling.models.cv import target_pool
    return target_pool(category, verbose=False)


def existing_model_run(category: str = "wtotal", cities: list[str] | None = None,
                       pop_floor: int = POP_FLOOR) -> dict:
    """Assemble a refresh-style ``run`` dict for the INCUMBENT model.

    ``scored`` carries ``city`` / ``holdout_city`` / ``population`` / ``y_pred`` (the incumbent
    predicted weighted rate, ``log1p``-transformed onto the log-rate scale so its within-city
    correlation/recall line up with the refresh's ``lograte`` predictions) / ``{category}_rate``
    (observed weighted rate), one row per within-city BG above ``pop_floor`` — the exact shape
    ``within_city_recall`` / ``loco_metrics`` expect, with ``mode="lograte_within_city"`` so
    ``_rate_col`` resolves to ``{category}_rate``. (``log1p`` is monotonic, so it leaves the
    within-city ranking — hence recall — unchanged; it only puts the R²/level correlations on
    the same log-log footing as ``04_bg_comparison``.)
    """
    pred_col = EXISTING_PRED_COL[category]
    obs_col = f"{category}_rate"
    cities = cities or _default_cities(category)
    need = ["geoid", "within_city", "population", pred_col] + [f"{c}_rate" for c in PRIMARY_CRIMES]

    frames = []
    for city in cities:
        d = pd.read_parquet(INTERIM_DIR / "bg_crime" / f"{city}.parquet", columns=need)
        d = d[(d["within_city"] == 1) & (d["population"] >= pop_floor)].copy()
        d["city"] = city
        frames.append(d)
    panel = compute_weighted_scores(pd.concat(frames, ignore_index=True))
    panel = panel.dropna(subset=[pred_col, obs_col])

    scored = (panel.assign(y_pred=np.log1p(panel[pred_col]))
              [["city", "population", "y_pred", obs_col]]
              .assign(holdout_city=lambda d: d["city"]))
    return {"scored": scored, "fits": {}, "mode": "lograte_within_city",
            "category": category, "predictors": [pred_col], "split": "existing"}


def between_city_level(run: dict) -> pd.Series:
    """Between-city LEVEL skill = correlation of per-city MEAN predicted vs observed rate.
    High for the incumbent (its strength); the refresh's demeaned model discards this by
    construction and must recover it via a leveller (division dummies / Model D anchor)."""
    from scipy.stats import pearsonr, spearmanr
    cat = run["category"]
    cm = (run["scored"].groupby("holdout_city")
          .agg(pred=("y_pred", "mean"), obs=(f"{cat}_rate", "mean")))
    return pd.Series({"n_cities": len(cm),
                      "between_pearson": round(float(pearsonr(cm["pred"], cm["obs"])[0]), 3),
                      "between_spearman": round(float(spearmanr(cm["pred"], cm["obs"])[0]), 3)})

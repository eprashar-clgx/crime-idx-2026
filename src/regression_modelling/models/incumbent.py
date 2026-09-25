"""The incumbent agency-scale model, expressed as a `FoldRun` (ADR 0008).

`04_bg_comparison` established the incumbent's scale case: it levels whole cities almost
perfectly (between-city) but fits weakly *within* a city, which is where ~95% of the
variance lives. This module reproduces that panel as a `FoldRun` so the prediction
notebook can score the incumbent with the *exact same* rank helpers
(`metrics.within_city_recall`, `metrics.within_corr2_pooled`) it uses for the refresh —
an apples-to-apples comparison rather than two hand-rolled metric paths.

That is the payoff of the result contract: the incumbent has no folds, no estimator and
no training loop, yet the rank metrics apply to it unchanged, because all they require
is a `FoldRun`. The concentration metrics (`metrics.fold_metrics`) are the exception —
they need a crime-COUNT column, and the incumbent panel carries only rates. Those are
not reported for the incumbent.

The incumbent per-BG prediction ships in ``data/interim/bg_crime/{city}.parquet`` as a
predicted weighted crime rate (``total_pt_ct`` for the total composite, ``property_pt_ct``
for the property composite) — a linear risk model that down-scales agency crime to the
block group, calibrated so each agency's population-weighted mean matches its agency rate.
It is compared, *within a city*, against the observed weighted relative-risk rate
(``compute_weighted_scores`` → ``wtotal_rate`` / ``wprop_rate``) — the same target the
refresh models. Both are the same kind of weighted rate, but the incumbent carries a
per-city level offset (it is agency-anchored, the observed rate is
national-composite-anchored), so comparisons here are rank/level-based (Pearson/Spearman,
within-city percentile recall), which are invariant to that offset.

The incumbent is a single *deployed* model, so it has no LOCO / 80-20 split — one fixed
set of numbers that sits beside every refresh protocol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crime_blockgroup_mapping.config import INTERIM_DIR
from crime_blockgroup_mapping.scores import compute_weighted_scores, PRIMARY_CRIMES
from regression_modelling.models.dataset import target_pool
from regression_modelling.models.results import FoldRun

#: Incumbent prediction column per target family (predicted weighted crime rate).
EXISTING_PRED_COL = {"wtotal": "total_pt_ct", "wprop": "property_pt_ct"}

#: Drop a handful of tiny-population BGs whose count/pop rates explode (matches 04).
POP_FLOOR = 250


def _default_cities(category: str) -> list[str]:
    """The incumbent is scored on the same pool the refresh is trained on, so the
    comparison is like-for-like. Delegates to the single pool owner (ADR 0008) rather
    than re-deriving eligibility here — this previously kept its own copy of the rule
    and silently included cities with degenerate all-zero crime extracts."""
    return target_pool(category, verbose=False)


def existing_model_run(category: str = "wtotal", cities: list[str] | None = None,
                       pop_floor: int = POP_FLOOR) -> FoldRun:
    """Assemble the INCUMBENT model's predictions as a `FoldRun`.

    ``scored`` carries ``city`` / ``holdout_city`` / ``population`` / ``y_pred`` (the
    incumbent predicted weighted rate, ``log1p``-transformed onto the log-rate scale so
    its within-city correlation/recall line up with the refresh's ``lograte``
    predictions) / ``{category}_rate`` (observed weighted rate), one row per within-city
    BG above ``pop_floor``. ``mode="lograte_within_city"`` so `_rate_col` resolves to
    ``{category}_rate``; ``split="existing"`` marks it as a deployed model rather than a
    cross-validation protocol.

    (``log1p`` is monotonic, so it leaves the within-city ranking — hence recall —
    unchanged; it only puts the R²/level correlations on the same log-log footing as
    ``04_bg_comparison``.)
    """
    pred_col = EXISTING_PRED_COL[category]
    obs_col = f"{category}_rate"
    cities = cities or _default_cities(category)
    need = (["geoid", "within_city", "population", pred_col]
            + [f"{c}_rate" for c in PRIMARY_CRIMES])

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
    return FoldRun(scored=scored, fits={}, mode="lograte_within_city",
                   split="existing", category=category, predictors=[pred_col])


def between_city_level(run: FoldRun) -> pd.Series:
    """Between-city LEVEL skill = correlation of per-city MEAN predicted vs observed rate.

    High for the incumbent (its strength); the refresh's demeaned model discards this by
    construction and must recover it via a leveller (division dummies / Model D anchor).
    """
    from scipy.stats import pearsonr, spearmanr
    cm = (run.scored.groupby("holdout_city")
          .agg(pred=("y_pred", "mean"), obs=(f"{run.category}_rate", "mean")))
    return pd.Series({"n_cities": len(cm),
                      "between_pearson": round(float(pearsonr(cm["pred"], cm["obs"])[0]), 3),
                      "between_spearman": round(float(spearmanr(cm["pred"], cm["obs"])[0]), 3)})

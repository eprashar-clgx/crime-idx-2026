"""The result contract shared by `training` (producer) and `metrics`/`diagnostics`
(consumers), per ADR 0008.

`FoldRun` is a labeled container, not logic: it computes nothing and changes no number.
Its job is to write down the schema that every fold-based run already returned by
convention, so the metrics can stay indifferent to *how* a run was produced.

That indifference is the point. `loco_metrics` works identically on a RidgeCV LOCO run,
a LightGBM LOCO run and a stratified 80/20 run because all three arrive as a `FoldRun`.
Adding a fold protocol (spatially stratified, at scale-up) or an estimator costs zero
metric changes — the new driver just has to produce one of these.

Lives in its own module so the dependency stays one-way: `training` and `metrics` both
import `results`, and `metrics` never imports `training`.
"""
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FoldRun:
    """One completed cross-validation run: every holdout row, scored.

    - ``scored``     : all holdout rows concatenated, each carrying a `y_pred` risk score
                       and a `holdout_city` tag. Under LOCO each city appears exactly once,
                       scored while it was held out; under an 80/20 split every row carries
                       its own city. This is the out-of-sample frame the metrics consume.
    - ``fits``       : {fold tag -> fitted estimator}, for per-fold coefficient or
                       feature-importance inspection. Empty when folds are not retained.
    - ``mode``       : the target mode the run was fit on (e.g. `"lograte"`), needed by
                       metrics to recover the underlying rate column.
    - ``split``      : which fold protocol produced it — `"loco"` or `"holdout"`.
    - ``category``   : the crime target, e.g. `"wtotal"` / `"wprop"`.
    - ``predictors`` : the RESOLVED predictor list actually fit (post column-filtering),
                       not the requested one — so a run always reports what it really used.
    """

    scored: pd.DataFrame
    mode: str
    split: str
    category: str
    predictors: list[str]
    fits: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = {"y_pred", "holdout_city"} - set(self.scored.columns)
        if missing:
            raise ValueError(
                f"FoldRun.scored is missing required column(s) {sorted(missing)}; "
                "every driver must tag its holdout rows before returning."
            )

    @property
    def n_scored(self) -> int:
        """Rows scored out-of-sample."""
        return int(len(self.scored))

    def __repr__(self) -> str:
        return (f"FoldRun(split={self.split!r}, mode={self.mode!r}, "
                f"category={self.category!r}, n_scored={self.n_scored}, "
                f"n_predictors={len(self.predictors)}, n_fits={len(self.fits)})")

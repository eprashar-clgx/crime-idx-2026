"""Predictor functional-form transforms for modeling / EDA.

Right-skewed transit supply features are log1p-compressed into `{col}_log`; the structural
zeros (stopless BGs) are split into a `transit_has_transit` indicator so the model can
separate "no transit at all" from "how much" instead of conflating both at log1p(0)=0.
Bounded features (shares, mode diversity ∈ [0,1]) are left raw.

The mapping of column → functional form is owned by `TRANSIT_MODEL_TRANSFORMS` in
`regression_modelling.constants` (single source of truth); this module just applies it.
Pearson correlation / OLS see these forms directly (so functional form matters here);
Spearman is invariant to any monotonic transform.
"""
import numpy as np
import pandas as pd

from regression_modelling.constants import TRANSIT_MODEL_TRANSFORMS

LOG_SUFFIX = "_log"
CENTERED_SUFFIX = "_logc"           # log1p, mean-centered on the served (has_transit=1) mass
HAS_TRANSIT_COL = "transit_has_transit"
_HAS_TRANSIT_SOURCE = "transit_stop_count"
# The intensive-margin feature paired with the has_transit indicator in the hurdle form.
# Its log1p is 0 exactly when has_transit=0, so centering it on the served mass makes it
# orthogonal to the indicator (see docs/features/transit_stats.md §4).
HURDLE_INTENSIVE_COL = "transit_service_intensity"


def apply_transforms(df, spec=TRANSIT_MODEL_TRANSFORMS, has_transit_from=_HAS_TRANSIT_SOURCE,
                     hurdle=False, hurdle_col=HURDLE_INTENSIVE_COL):
    """Add model-form columns to a copy of `df` per `spec`; return (df, model_cols).

    Parameters
    ----------
    df : DataFrame with the raw predictor columns.
    spec : dict {raw_col: "log1p" | "identity"} — defaults to TRANSIT_MODEL_TRANSFORMS.
    has_transit_from : column whose >0 test defines the `transit_has_transit` indicator;
        pass None to skip the indicator.
    hurdle : if True, emit the intensive-margin feature (`hurdle_col`) as a log1p value
        mean-centered on the served (has_transit=1) mass — `{col}_logc`, 0 for stopless BGs
        — instead of the raw `{col}_log`. This makes it exactly orthogonal to the
        `transit_has_transit` indicator (Cov=0 by construction). Requires the indicator, so
        `has_transit_from` must be set. See docs/features/transit_stats.md.
    hurdle_col : which raw column gets the centered treatment when `hurdle=True`.

    Returns
    -------
    (out_df, model_cols) where model_cols lists the columns to feed a model / correlation
    (the `{col}_log`/`{col}_logc` names for log1p features, the raw name for identity
    features, plus `transit_has_transit`). Missing input columns are skipped with a warning.
    """
    out = df.copy()
    model_cols = []

    served = None
    if has_transit_from and has_transit_from in out.columns:
        served = out[has_transit_from].fillna(0) > 0
        out[HAS_TRANSIT_COL] = served.astype(int)
        model_cols.append(HAS_TRANSIT_COL)

    if hurdle and served is None:
        raise ValueError("apply_transforms: hurdle=True requires has_transit_from (the "
                         "indicator) to be set")

    for col, form in spec.items():
        if col not in out.columns:
            print(f"apply_transforms: '{col}' not in df — skipping")
            continue
        if form == "log1p":
            logv = np.log1p(out[col].clip(lower=0))
            if hurdle and col == hurdle_col:
                # center on the served mass; stopless BGs -> 0 (indicator carries them)
                mean_served = logv[served].mean()
                name = f"{col}{CENTERED_SUFFIX}"
                out[name] = np.where(served, logv - mean_served, 0.0)
            else:
                name = f"{col}{LOG_SUFFIX}"
                out[name] = logv
            model_cols.append(name)
        elif form == "identity":
            model_cols.append(col)
        else:
            raise ValueError(f"apply_transforms: unknown form '{form}' for '{col}'")

    return out, model_cols

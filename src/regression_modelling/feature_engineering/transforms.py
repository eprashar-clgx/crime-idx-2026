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
HAS_TRANSIT_COL = "transit_has_transit"
_HAS_TRANSIT_SOURCE = "transit_stop_count"


def apply_transforms(df, spec=TRANSIT_MODEL_TRANSFORMS, has_transit_from=_HAS_TRANSIT_SOURCE):
    """Add model-form columns to a copy of `df` per `spec`; return (df, model_cols).

    Parameters
    ----------
    df : DataFrame with the raw predictor columns.
    spec : dict {raw_col: "log1p" | "identity"} — defaults to TRANSIT_MODEL_TRANSFORMS.
    has_transit_from : column whose >0 test defines the `transit_has_transit` indicator;
        pass None to skip the indicator.

    Returns
    -------
    (out_df, model_cols) where model_cols lists the columns to feed a model / correlation
    (the `{col}_log` names for log1p features, the raw name for identity features, plus
    `transit_has_transit`). Missing input columns are skipped with a warning.
    """
    out = df.copy()
    model_cols = []

    if has_transit_from and has_transit_from in out.columns:
        out[HAS_TRANSIT_COL] = (out[has_transit_from].fillna(0) > 0).astype(int)
        model_cols.append(HAS_TRANSIT_COL)

    for col, form in spec.items():
        if col not in out.columns:
            print(f"apply_transforms: '{col}' not in df — skipping")
            continue
        if form == "log1p":
            name = f"{col}{LOG_SUFFIX}"
            out[name] = np.log1p(out[col].clip(lower=0))
            model_cols.append(name)
        elif form == "identity":
            model_cols.append(col)
        else:
            raise ValueError(f"apply_transforms: unknown form '{form}' for '{col}'")

    return out, model_cols

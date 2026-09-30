"""Predictor screening — "is this predictor worth modelling?" (ADR 0010).

Model-free rules that cut a candidate list to a reviewed shortlist *before* any
selection fit. Every rule reads the pooled BG frame only; nothing here fits a model.

Rules (a predictor is dropped if any fires; thresholds are arguments so a notebook can
vary them):

- **weak**      — |median within-city Spearman with the target| < ``weak_rho`` AND the
                  sign agrees in fewer than ``sign_agree_min`` of cities.
- **artifact**  — between-city variance share > ``between_share_max`` AND the column is in
                  ``artifacts`` (a known data-collection cause, e.g. liens coverage). A high
                  share alone only earns a ``city_identifying`` note — it may be real level.
- **sparse**    — zero in more than ``sparse_zero_max`` of BGs.
- **redundant** — among survivors, taken strongest-first by |within-city rho|: dropped if
                  its mean within-city |Spearman| with an already-kept predictor exceeds
                  ``redundant_rho``.

``within_city_spearman`` and ``between_share`` are also the §4/§5 EDA quantities, so the
notebook and the screen compute them one way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Predictors whose between-city spread is a known coverage artifact, not crime signal.
DEFAULT_ARTIFACTS = ("clip_liens_pct_log", "clip_liens_pct_lag6_log")


def within_city_spearman(df: pd.DataFrame, cols, target: str,
                         city_col: str = "city") -> pd.DataFrame:
    """Spearman of each predictor with ``target`` computed inside each city.

    Returns a predictor x city frame. City-constant columns (e.g. the agency anchor) are
    NaN — they carry no within-city signal by construction.
    """
    cols = list(cols)
    return pd.DataFrame({city: g[cols].corrwith(g[target], method="spearman")
                         for city, g in df.groupby(city_col)})


def between_share(df: pd.DataFrame, cols, city_col: str = "city") -> pd.Series:
    """Share of each column's variance that lies between cities (eta^2 = SS_between/SS_total).

    1 = the column only identifies the city; 0 = it varies entirely within cities.
    """
    out = {}
    for c in cols:
        x = df[[city_col, c]].dropna()
        dev = x[c] - x[c].mean()
        ss_total = float((dev ** 2).sum())
        g = x.groupby(city_col)[c]
        ss_between = float((g.size() * (g.mean() - x[c].mean()) ** 2).sum())
        out[c] = ss_between / ss_total if ss_total else np.nan
    return pd.Series(out, name="between_share")


def within_city_corr_matrix(df: pd.DataFrame, cols, city_col: str = "city") -> pd.DataFrame:
    """Mean of the per-city Spearman correlation matrices (predictor x predictor).

    Redundancy is judged within cities because that is where the model has to rank; a
    pooled matrix would also pick up two columns that merely share a city level.
    """
    mats = [g[list(cols)].corr(method="spearman") for _, g in df.groupby(city_col)]
    stacked = np.stack([m.to_numpy() for m in mats])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stacked, axis=0)
    return pd.DataFrame(mean, index=list(cols), columns=list(cols))


def vif_table(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Variance inflation factor per predictor (median-imputed, with a constant)."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    X = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    X = add_constant(X.fillna(X.median()))
    vif = pd.DataFrame({"predictor": X.columns,
                        "vif": [variance_inflation_factor(X.values, i)
                                for i in range(X.shape[1])]})
    return (vif[vif["predictor"] != "const"].set_index("predictor")
            .sort_values("vif", ascending=False))


def screen_predictors(df: pd.DataFrame, candidates, target: str, *,
                      fixed=(), city_col: str = "city",
                      weak_rho: float = 0.10, sign_agree_min: float = 0.80,
                      redundant_rho: float = 0.80, between_share_max: float = 0.70,
                      artifacts=DEFAULT_ARTIFACTS, sparse_zero_max: float = 0.95
                      ) -> pd.DataFrame:
    """Apply the screening rules and return one row per predictor with a verdict.

    Parameters
    ----------
    df : pooled BG frame with a ``city`` column, the candidates and ``target``.
    candidates : predictor columns to screen.
    target : outcome column (e.g. ``"wprop_rate"``; Spearman makes the log irrelevant).
    fixed : columns that are always kept (e.g. the agency anchor); reported, never screened.
    weak_rho, sign_agree_min, redundant_rho, between_share_max, sparse_zero_max, artifacts :
        rule thresholds — see the module docstring.

    Returns
    -------
    DataFrame indexed by predictor: ``within_rho`` (median across cities), ``sign_agree``,
    ``between_share``, ``pct_zero``, ``partner`` / ``partner_rho`` (most-correlated kept
    predictor), ``verdict`` (keep / drop / fixed), ``reason`` and ``note``. Sorted with
    kept predictors first, strongest first. Review it before acting on it.
    """
    candidates = [c for c in candidates if c not in set(fixed)]
    missing = [c for c in [*candidates, *fixed, target] if c not in df.columns]
    if missing:
        raise KeyError(f"screen_predictors: columns not in df: {missing}")

    W = within_city_spearman(df, candidates, target, city_col)
    med = W.median(axis=1)
    sign_agree = W.apply(
        lambda r: float((np.sign(r.dropna()) == np.sign(med[r.name])).mean())
        if r.notna().any() else np.nan, axis=1)
    tbl = pd.DataFrame({
        "within_rho": med,
        "sign_agree": sign_agree,
        "between_share": between_share(df, candidates, city_col),
        "pct_zero": (df[candidates] == 0).mean(),
    })
    tbl["partner"], tbl["partner_rho"] = None, np.nan
    tbl["verdict"], tbl["reason"], tbl["note"] = "keep", "", ""

    def _drop(col, reason):
        tbl.loc[col, "verdict"] = "drop"
        tbl.loc[col, "reason"] = "; ".join(r for r in (tbl.loc[col, "reason"], reason) if r)

    artifacts = set(artifacts)
    for c in candidates:
        r = tbl.loc[c]
        if abs(r.within_rho) < weak_rho and r.sign_agree < sign_agree_min:
            _drop(c, f"weak (|rho|={abs(r.within_rho):.2f}, sign_agree={r.sign_agree:.2f})")
        if r.between_share > between_share_max:
            if c in artifacts:
                _drop(c, f"artifact (between_share={r.between_share:.2f})")
            else:
                tbl.loc[c, "note"] = f"city_identifying (between_share={r.between_share:.2f})"
        if r.pct_zero > sparse_zero_max:
            _drop(c, f"sparse ({r.pct_zero:.0%} zero)")

    C = within_city_corr_matrix(df, candidates, city_col).abs()
    survivors = tbl.index[tbl["verdict"] == "keep"]
    kept: list[str] = []
    for c in tbl.loc[survivors, "within_rho"].abs().sort_values(ascending=False).index:
        if kept:
            partner = C.loc[c, kept].idxmax()
            tbl.loc[c, ["partner", "partner_rho"]] = [partner, C.loc[c, partner]]
            if C.loc[c, partner] > redundant_rho:
                _drop(c, f"redundant with {partner} (|rho|={C.loc[c, partner]:.2f})")
                continue
        kept.append(c)

    fixed_rows = pd.DataFrame({"verdict": "fixed", "reason": "", "note": "always kept"},
                              index=list(fixed))
    out = pd.concat([fixed_rows, tbl]) if len(fixed) else tbl
    order = out["verdict"].map({"fixed": 0, "keep": 1, "drop": 2})
    out = out.assign(_o=order, _s=-out["within_rho"].abs().fillna(0))
    return out.sort_values(["_o", "_s"]).drop(columns=["_o", "_s"])

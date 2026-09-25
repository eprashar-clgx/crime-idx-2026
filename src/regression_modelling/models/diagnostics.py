"""Why a model fits — or doesn't (ADR 0008).

Lifted out of `02_regression_prediction` cells 21/27, which is where these lived as
notebook-local functions. They answer a different question from `metrics`: not *how
good is this run* but *what is driving it, and which city is it failing on*.

Everything here reads a `FoldRun` that was produced with `keep_fits=True`. The notebook
version had to hand-rebuild the folds (`_diag_folds`) because the old drivers threw
their fitted models away; `training.run_cv` retains them, so that helper is gone and
the diagnostics operate on the same run object the metrics do.
"""
import numpy as np
import pandas as pd

from regression_modelling.models.results import FoldRun


def varying_predictors(pool: pd.DataFrame, predictors, city_col: str = "city") -> list[str]:
    """Predictors that vary WITHIN a city (block-to-block).

    City-constant LEVEL features (the agency anchor, division dummies) are excluded:
    within-city centroids and permutation cannot see a column that is constant inside
    every city, so including them produces meaningless zeros rather than information.
    """
    return [p for p in predictors
            if pool.groupby(city_col)[p].std().mean() > 1e-9]


def _r2(scored: pd.DataFrame, category: str) -> float:
    """Pooled out-of-sample R² on the log-rate scale."""
    y = np.log1p(scored[f"{category}_rate"].astype(float))
    p = scored["y_pred"].astype(float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def city_diag(run: FoldRun, pool: pd.DataFrame) -> pd.DataFrame:
    """Per-held-out-city fit diagnostics, worst `r2_oos` first.

    Separates the two distinct ways a held-out city fits badly:

    - **Too little internal spread to rank** — `obs_sd` (within-city spread of the
      observed log rate) near zero, with a weak `within_corr`. There is simply not much
      variation to find.
    - **Too atypical or mis-levelled to place** — a large `level_err` (predicted minus
      observed mean) or a large `feat_dist` (how far the city's standardized feature
      centroid sits from the pooled centre). The model can rank the city internally but
      puts the whole city at the wrong height.

    The distinction matters because the fixes differ: the first is a data-coverage
    problem, the second is a levelling problem (ADR 0005's leveller).
    """
    category = run.category
    varying = varying_predictors(pool, run.predictors)
    mu, sd = pool[varying].mean(), pool[varying].std(ddof=0).replace(0, 1.0)
    Z = (pool[varying] - mu) / sd
    Z["city"] = pool["city"].to_numpy()
    centroid = Z.groupby("city")[varying].mean()
    fdist = np.sqrt((centroid ** 2).sum(axis=1) / len(varying))

    rows = []
    for city, h in run.scored.groupby("holdout_city"):
        y = np.log1p(h[f"{category}_rate"].astype(float))
        p = h["y_pred"].astype(float)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rows.append({
            "city": city, "n": len(h),
            "obs_mean": round(float(y.mean()), 2),
            "obs_sd": round(float(y.std(ddof=0)), 2),
            "level_err": round(float(p.mean() - y.mean()), 2),
            "r2_oos": round(float(r2), 3),
            "within_corr": round(float(np.corrcoef(y, p)[0, 1]), 2),
            "feat_dist": round(float(fdist[city]), 2),
        })
    return pd.DataFrame(rows).set_index("city").sort_values("r2_oos")


def perm_importance(run: FoldRun, pool: pd.DataFrame, nrep: int = 3,
                    seed: int = 42) -> tuple[float, pd.Series]:
    """LOCO permutation importance: the drop in pooled out-of-sample R² when a predictor
    is shuffled inside each held-out city, with the fold model left unchanged.

    A model-agnostic measure of a feature's OUT-OF-SAMPLE worth, which is a different
    question from how much the model leaned on it in training (ridge coefficient, GBM
    gain) — a feature can be heavily used and still carry no transferable signal.

    City-constant LEVEL features return NaN: shuffling a column within one held-out city
    cannot move a value that is constant across that city. Their worth shows up as the
    ladder's LEVEL lift instead.

    Returns `(base_r2, drop_per_predictor)`.
    """
    base = _r2(run.scored, run.category)
    varying = set(varying_predictors(pool, run.predictors))
    rng = np.random.default_rng(seed)

    folds = [(city, run.fits[city], h)
             for city, h in run.scored.groupby("holdout_city")
             if city in run.fits]

    out = {}
    for feature in run.predictors:
        if feature not in varying:
            out[feature] = np.nan
            continue
        drops = []
        for _ in range(nrep):
            parts = []
            for _city, model, h in folds:
                d = h.copy()
                d[feature] = rng.permutation(d[feature].to_numpy())
                parts.append(d.assign(y_pred=model.predict(d)))
            drops.append(base - _r2(pd.concat(parts, ignore_index=True), run.category))
        out[feature] = float(np.mean(drops))
    return base, pd.Series(out, name="perm_dr2").round(4)


def ridge_coef_table(run: FoldRun) -> pd.DataFrame:
    """Mean standardized RidgeCV coefficient across folds + sign stability.

    `sign_stab` is the share of folds agreeing with the mean coefficient's sign — a
    coefficient that flips sign across held-out cities is not a finding.
    """
    coefs = np.vstack([m.model.coef_ for m in run.fits.values()])
    df = pd.DataFrame(coefs, columns=run.predictors)
    return pd.DataFrame({
        "coef_mean": df.mean().round(3),
        "sign_stab": (np.sign(df) == np.sign(df.mean())).mean().round(2),
    })


def gbm_gain(run: FoldRun) -> pd.Series:
    """Mean gain-importance share across folds (how much the trees USED each feature).

    Requires the run to have been fit with `gbm_params={"importance_type": "gain"}`;
    LightGBM's default reports split counts, which over-credits low-cardinality columns.
    """
    gains = pd.concat([pd.Series(m.model.feature_importances_, index=run.predictors)
                       for m in run.fits.values()], axis=1).mean(axis=1)
    return (gains / gains.sum()).round(4)

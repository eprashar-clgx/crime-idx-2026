"""Which predictors work, and where (05_predictor_diagnostics).

Three lenses, each answering a different question about a family of predictors
(`diagnostics.feature_family`):

- **Usage** — how far the model moves predictions with it: grouped held-out |SHAP|
  (`diagnostics.shap_importance`). Heavy use is not proof of value.
- **Worth** — what accuracy is lost without it: refit LOCO with the family removed
  (`family_dropout`) and score the gap (`dropout_deltas`, `dropout_by_city`). This is the
  family's *unique* contribution; a family whose signal another family also carries
  scores low here even if it is heavily used.
- **Direction** — whether, inside a given city, the blocks the family pushes up are the
  blocks with more crime (`family_direction`). Negative means the relationship the model
  learned elsewhere runs the wrong way in that city.

`family_level_push` splits a city's level error by family. `city_feature_table` repeats the lenses feature by feature for one held-out city, using
permutation (no refit) for worth.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.diagnostics import feature_family
from regression_modelling.models.metrics import scorecard, scorecard_by_city
from regression_modelling.models.results import FoldRun
from regression_modelling.models.tuning import run_loco_parallel

FULL = "full"
#: Overall metrics compared in `dropout_deltas` (all `scorecard` columns).
DROP_METRICS = ("r2_pooled", "r2_city", "spearman_city", "recall@10_city")


def family_groups(predictors: Iterable[str]) -> dict[str, list[str]]:
    """Predictors grouped by reporting family, in first-seen order."""
    groups: dict[str, list[str]] = {}
    for p in predictors:
        groups.setdefault(feature_family(p), []).append(p)
    return groups


def family_dropout(pool: pd.DataFrame, category: str, predictors: Iterable[str], *,
                   params: dict | None = None, seeds: Iterable[int] = (0,),
                   families: Iterable[str] | None = None,
                   n_jobs: int = -1) -> dict[tuple[str, int], FoldRun]:
    """LOCO runs of the full set and of the set minus each family, for every seed.

    Keys are ``(label, seed)`` with label `FULL` or a family name. The pool is first
    reduced to rows complete in every predictor, so all runs score the same BGs (a
    dropped family would otherwise let rows with gaps in it back in). ``seed`` overrides
    LightGBM's ``random_state``; the bagging it controls is the run-to-run noise.
    """
    preds = list(predictors)
    rows = pool.dropna(subset=[*preds, _rate_col("lograte", category)])
    groups = family_groups(preds)
    fams = list(groups) if families is None else list(families)
    runs = {}
    for seed in seeds:
        p = {**(params or {}), "random_state": seed}
        runs[(FULL, seed)] = run_loco_parallel(rows, category, preds, params=p, n_jobs=n_jobs)
        for fam in fams:
            keep = [c for c in preds if c not in groups[fam]]
            runs[(fam, seed)] = run_loco_parallel(rows, category, keep, params=p,
                                                  n_jobs=n_jobs)
    return runs


def _labels_and_seeds(runs: dict) -> tuple[list[str], list[int]]:
    labels = [lab for lab in dict.fromkeys(k[0] for k in runs) if lab != FULL]
    return labels, sorted({k[1] for k in runs})


def dropout_by_city(runs: dict[tuple[str, int], FoldRun], metric: str = "r2") -> pd.DataFrame:
    """City x family: ``metric`` with the full set minus ``metric`` without the family,
    averaged over seeds. Positive = the family helps that city. ``metric`` is a
    `scorecard_by_city` column (``r2``, ``spearman``, ``recall@10``)."""
    labels, seeds = _labels_and_seeds(runs)
    by = {k: scorecard_by_city(r)[metric].astype(float) for k, r in runs.items()}
    return pd.DataFrame({
        lab: pd.concat([by[(FULL, s)] - by[(lab, s)] for s in seeds], axis=1).mean(axis=1)
        for lab in labels})


def dropout_deltas(runs: dict[tuple[str, int], FoldRun],
                   metrics: Iterable[str] = DROP_METRICS) -> pd.DataFrame:
    """Overall worth per family: ``d_<metric>`` = full minus without-family (mean over
    seeds; positive = the family helps), ``sd_<metric>`` its SD over seeds when there is
    more than one, and ``cities_helped`` = held-out cities whose r2 the family raises."""
    metrics = list(metrics)
    labels, seeds = _labels_and_seeds(runs)
    cards = {k: scorecard(r)[metrics].astype(float) for k, r in runs.items()}
    helped = (dropout_by_city(runs, "r2") > 0).sum()
    rows = {}
    for lab in labels:
        d = pd.DataFrame([cards[(FULL, s)] - cards[(lab, s)] for s in seeds])
        row = d.mean().add_prefix("d_")
        if len(seeds) > 1:
            row = pd.concat([row, d.std().add_prefix("sd_")])
        row["cities_helped"] = int(helped[lab])
        rows[lab] = row
    out = pd.DataFrame(rows).T
    out["cities_helped"] = out["cities_helped"].astype(int)
    return out.sort_values(f"d_{metrics[0]}", ascending=False)


def _spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def _mid_pctile(values: pd.Series, x: float) -> float:
    """Mid-rank percentile of ``x`` in ``values`` (ties count half), so a value equal to a
    common mode such as 0 reads as typical rather than as the minimum."""
    v = values.dropna()
    return float((v < x).mean() + 0.5 * (v == x).mean())


def family_level_push(shap_df: pd.DataFrame, run: FoldRun) -> pd.DataFrame:
    """City x family mean signed SHAP: how far each family moves the held-out city's
    average prediction away from the fold's base (the training mean). With ``_base`` they
    sum to the city's mean prediction; ``obs_mean`` and ``level_err`` (mean predicted minus
    mean observed, log rate) show which families push the level the wrong way."""
    vals = shap_df.drop(columns=["_base", "holdout_city"])
    city = shap_df["holdout_city"]
    out = vals.T.groupby(feature_family).sum().T.groupby(city).mean()
    s = run.scored
    obs = np.log1p(s[_rate_col(run.mode, run.category)].astype(float)).groupby(s["holdout_city"]).mean()
    out["_base"] = shap_df["_base"].groupby(city).mean()
    out["obs_mean"] = obs
    out["level_err"] = s["y_pred"].groupby(s["holdout_city"]).mean() - obs
    return out


def family_direction(shap_df: pd.DataFrame, scored: pd.DataFrame, category: str,
                     exclude: Iterable[str] = ("agency anchor",)) -> pd.DataFrame:
    """City x family: within-city Spearman between the family's grouped SHAP (the push
    it gives each BG) and the observed log rate. Positive = the family raises the blocks
    that really have more crime. The anchor is excluded by default: it is constant inside
    a city, so its job is the level, not the ranking."""
    vals = shap_df.drop(columns=["_base", "holdout_city"])
    fam = vals.T.groupby(feature_family).sum().T
    fam = fam.drop(columns=[c for c in exclude if c in fam.columns])
    y = np.log1p(scored.loc[fam.index, f"{category}_rate"].astype(float))
    return pd.DataFrame({
        city: {f: _spearman(fam.loc[idx, f], y[idx]) for f in fam.columns}
        for city, idx in fam.groupby(shap_df["holdout_city"]).groups.items()}).T


def city_feature_table(run: FoldRun, shap_df: pd.DataFrame, city: str, n_rep: int = 5,
                       seed: int = 0) -> pd.DataFrame:
    """Feature-level lenses for one held-out ``city`` (its LOCO fold model, unchanged).

    - ``shap_share`` — share of the city's mean |SHAP| (usage).
    - ``perm_d_r2`` / ``perm_d_rho`` — drop in the city's r2 / Spearman when the feature
      is shuffled among the city's BGs (worth). Correlated partners can cover for a
      shuffled feature, so this understates shared signal.
    - ``local_rho`` — Spearman of the raw feature with observed log rate in the city;
      ``others_rho`` — the median of that across the other (training) cities.
    - ``model_dir`` — Spearman of the feature with its own SHAP in the city: the sign of
      the relationship the model applies. ``agrees`` — same sign as ``local_rho``.
    - ``city_pctile`` — where the city's median value sits in the other cities' BG
      distribution (0.5 = typical; near 0 or 1 = outside what the model trained on).

    City-constant features (the anchor) have NaN worth and direction.
    """
    s = run.scored
    y_all = np.log1p(s[_rate_col(run.mode, run.category)].astype(float))
    in_city = s["holdout_city"] == city
    h, y = s[in_city], y_all[in_city]
    others = s[~in_city]
    model = run.fits[city]
    rng = np.random.default_rng(seed)

    def _fit(pred):
        return 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()), _spearman(pred, y)

    base_r2, base_rho = _fit(h["y_pred"].to_numpy())
    sh = shap_df.loc[h.index, run.predictors]
    imp = sh.abs().mean()
    rows = {}
    for f in run.predictors:
        varies = h[f].nunique() > 1
        d_r2 = d_rho = np.nan
        if varies:
            drops = []
            for _ in range(n_rep):
                d = h.copy()
                d[f] = rng.permutation(d[f].to_numpy())
                r2, rho = _fit(model.predict(d))
                drops.append((base_r2 - r2, base_rho - rho))
            d_r2, d_rho = np.mean(drops, axis=0)
        other_rho = [_spearman(g[f], y_all[g.index]) for _, g in others.groupby("holdout_city")]
        local, direction = _spearman(h[f], y), _spearman(h[f], sh[f])
        rows[f] = {
            "family": feature_family(f), "shap_share": imp[f] / imp.sum(),
            "perm_d_r2": d_r2, "perm_d_rho": d_rho,
            "local_rho": local, "others_rho": float(np.nanmedian(other_rho)) if varies else np.nan,
            "model_dir": direction,
            "agrees": (bool(np.sign(local) == np.sign(direction))
                       if np.isfinite(local) and np.isfinite(direction) else None),
            "city_pctile": _mid_pctile(others[f], h[f].median()),
        }
    out = pd.DataFrame(rows).T
    num = [c for c in out if c not in ("family", "agrees")]
    out[num] = out[num].astype(float)
    return out.sort_values("shap_share", ascending=False)

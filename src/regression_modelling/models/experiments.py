"""POC experiment grid — the four differentiation axes, made explicit for leadership.

Leadership evaluates the model along FOUR axes. This module makes each one a first-class,
named field so a single results table reads them straight off — no hidden defaults:

  AXIS 1 · PREDICTOR SET  — which features enter the model
      "ours"                  our new BG features (constants.PREDICTOR_COLS)
      "ours+approved"         + the incumbent national model's governance-APPROVED,
                              non-duplicate features: in_household_pct, det_pct (within-city
                              varying) and the Census-division LEVEL dummies
      "ours+approved+agency"  + the observed lagged UCR agency crime anchor (Model D, ADR 0007)
  AXIS 2 · TRANSFORM      — how predictors/target are treated
      "absolute"     raw predictors + absolute log-rate target → carries between-city LEVEL
      "within_city"  per-city demeaned predictors + within-city target → RANK inside a city
      transit sub-variant: "gtfs" (per-city feed, best-fit) vs "acs" (national, deployable)
  AXIS 3 · TRAINING       — evaluation protocol
      "loco"   leave-one-CITY-out — extrapolate to an unseen city (the honest deployment test)
      "80_20"  stratified 80/20   — interpolate to unseen blocks in cities we have seen
  AXIS 4 · MODEL          — estimator family
      "ridge"  RidgeCV (linear, shrunk)   "ols" (linear, unshrunk — inference)
      "gbm"    LightGBM (non-linear trees)

`run_grid` returns one tidy row per (spec × target), with the four axes as leading columns
followed by the metrics that matter for that transform (ranking recall for within_city,
absolute R² for absolute). The between-city LEVEL features (division dummies, agency anchor)
are constant within a city, so they only bind in the "absolute" transform — they demean to
zero under "within_city". That is the whole point of the level-vs-rank split.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

from regression_modelling.constants import (
    PREDICTOR_SETS, AGENCY_ANCHOR_COL, TRANSIT_MODEL_PREDICTORS,
    ACS_TRANSIT_MODEL_PREDICTORS,
)
from regression_modelling.models.cv import (
    run_loco, run_holdout, loco_metrics, within_city_recall,
    loco_folds, stratified_split, make_target,
)

# AXIS 2 · transform name -> (target mode, demean predictors within city?)
TRANSFORM_SPECS = {
    "absolute":    ("lograte", False),
    "within_city": ("lograte_within_city", True),
}


@dataclass(frozen=True)
class ExperimentSpec:
    """One cell of the POC grid — the four differentiation axes plus the transit sub-variant."""
    predictor_set: str            # AXIS 1: "ours" | "ours+approved" | "ours+approved+agency"
    transform: str                # AXIS 2: "absolute" | "within_city"
    training: str                 # AXIS 3: "loco" | "80_20"
    model: str                    # AXIS 4: "ridge" | "ols" | "gbm"
    transit: str = "gtfs"         # AXIS 2 sub-variant: "gtfs" | "acs"
    gbm_params: dict | None = None  # optional LightGBM hyperparameter override (else defaults)

    def label(self) -> str:
        return (f"{self.predictor_set} · {self.transform} · {self.training} · "
                f"{self.model} · transit={self.transit}")


def resolve_predictors(spec: ExperimentSpec, category: str, pool: pd.DataFrame) -> list[str]:
    """Materialize a spec's predictor list for one target, filtered to columns present.

    Handles the transit sub-variant swap (GTFS block ↔ ACS block) and appends the
    target-paired agency anchor column for "+agency" sets.
    """
    add_agency = spec.predictor_set.endswith("+agency")
    base_name = "ours+approved" if add_agency else spec.predictor_set
    preds = list(PREDICTOR_SETS[base_name])
    if spec.transit == "acs":
        preds = ([p for p in preds if p not in TRANSIT_MODEL_PREDICTORS]
                 + list(ACS_TRANSIT_MODEL_PREDICTORS))
    if add_agency:
        preds = preds + [AGENCY_ANCHOR_COL[category]]
    return [p for p in preds if p in pool.columns]


def _run_gbm(pool, mode, predictors, category, training, gbm_params=None):
    """LightGBM LOCO / 80-20 scorer mirroring run_loco's return schema. Trees are level- and
    scale-invariant, so predictors enter raw (no demean / standardize); the target still
    follows the transform's mode so within-city runs predict within-city rank.

    ``gbm_params`` optionally overrides the default LightGBM hyperparameters (e.g. an
    Optuna-tuned set); anything not supplied falls back to the deployed defaults."""
    import lightgbm as lgb

    # Deployed defaults; a tuned override is merged on top so callers can pass a partial set.
    params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                  subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                  n_jobs=1, verbose=-1)
    if gbm_params:
        params.update(gbm_params)

    def fit_score(train, test, city_tag):
        tr = train.dropna(subset=predictors).copy()
        tr["_y"] = make_target(tr, mode, category)
        tr = tr.dropna(subset=["_y"])
        m = lgb.LGBMRegressor(**params)
        m.fit(tr[predictors], tr["_y"].to_numpy())
        te = test.dropna(subset=predictors).copy()
        te["y_pred"] = m.predict(te[predictors])
        return te.assign(holdout_city=city_tag if city_tag is not None else te["city"])

    parts = []
    if training == "loco":
        for city, train, hold in loco_folds(pool):
            parts.append(fit_score(train, hold, city))
    else:
        train, test = stratified_split(pool)
        parts.append(fit_score(train, test, None))
    scored = pd.concat(parts, ignore_index=True)
    return {"scored": scored, "fits": {}, "mode": mode, "split": training,
            "category": category, "predictors": list(predictors)}


def run_spec(spec: ExperimentSpec, pool: pd.DataFrame, category: str) -> dict:
    """Materialize and run one spec against one target's pool. Public entry point used both
    by ``run_grid`` and by notebooks that compose focused, single-axis comparisons."""
    preds = resolve_predictors(spec, category, pool)
    mode, demean = TRANSFORM_SPECS[spec.transform]
    if spec.model in ("ridge", "ols"):
        runner = run_loco if spec.training == "loco" else run_holdout
        return runner(pool, mode=mode, predictors=preds, category=category,
                      demean_by_city=demean, estimator=spec.model)
    if spec.model == "gbm":
        return _run_gbm(pool, mode, preds, category, spec.training, spec.gbm_params)
    raise ValueError(f"unknown model {spec.model!r}")


def _within_corr2_pooled(run: dict) -> float:
    """Mean per-city within-city **squared correlation** (Pearson² of y_pred vs
    log1p(rate)) — within-city variance explained.

    Deliberately not called "R²": it is affine-invariant, so a model that ranks a city
    correctly but levels it badly scores identically to one that gets both right. The
    level-sensitive counterpart is `r2_oos`. See the metric vocabulary in CONTEXT.md.
    """
    from regression_modelling.models.cv import _rate_col
    rate_col = _rate_col(run["mode"], run["category"])
    r2s = []
    for _, g in run["scored"].groupby("holdout_city"):
        if len(g) <= 2:
            continue
        y = np.log1p(g[rate_col].astype(float)); p = g["y_pred"].astype(float)
        r = np.corrcoef(y, p)[0, 1]
        if np.isfinite(r):
            r2s.append(r * r)
    return round(float(np.mean(r2s)), 3) if r2s else float("nan")


def between_city_level_r(run: dict) -> float:
    """Between-city Pearson of city MEANS (predicted vs observed log-rate) — the cross-city
    LEVEL skill an absolute model must have. Pooled BG r2_oos is dominated by within-city
    scatter (~95% of variance), so this isolates the leveller's actual job."""
    from regression_modelling.models.cv import _rate_col
    rate_col = _rate_col(run["mode"], run["category"])
    g = run["scored"].copy()
    g["_obs"] = np.log1p(g[rate_col].astype(float))
    means = g.groupby("holdout_city").agg(pred=("y_pred", "mean"), obs=("_obs", "mean"))
    if len(means) < 3:
        return float("nan")
    r = np.corrcoef(means["pred"], means["obs"])[0, 1]
    return round(float(r), 3) if np.isfinite(r) else float("nan")


def _metrics_row(spec: ExperimentSpec, run: dict, category: str) -> dict:
    m = loco_metrics(run).loc["POOLED"]
    row = {"category": category, **asdict(spec),
           "n": int(m["n"]), "skill": m.get("skill"), "spearman": m.get("spearman")}
    if spec.transform == "within_city":
        # Legacy 4-axis grid: keep its original worst-DECILE → top-10%/30% recall
        # (pinned explicitly; the cv default is now the worst-QUARTILE 25/25 recall the
        # 02/04 notebooks report).
        rec = within_city_recall(run, category, danger_top=0.10, nets=(0.10, 0.30)).loc["POOLED"]
        row["recall@top10"] = rec["recall@top10"]
        row["recall@top30"] = rec["recall@top30"]
        row["called_safe"] = rec["called_safe"]
        row["within_corr2"] = _within_corr2_pooled(run)
    else:  # absolute → the between-city LEVEL metrics (r2_oos over all BGs + city-mean level_r)
        row["r2_oos"] = m.get("r2_oos")
        row["level_r"] = between_city_level_r(run)
    return row


def run_grid(specs: list[ExperimentSpec], pools: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run every (spec × target) and return one tidy, axis-labeled results row each.

    `pools` maps a target category to its pooled table, e.g.
    ``{"wtotal": pool5, "wprop": pool10}``. Leading columns are the four axes (+ transit
    sub-variant); trailing columns are the transform-appropriate metrics.
    """
    rows = []
    for spec in specs:
        for category, pool in pools.items():
            print(f"\n▶ {category} · {spec.label()}")
            rows.append(_metrics_row(spec, run_spec(spec, pool, category), category))
    axis_cols = ["category", "predictor_set", "transform", "transit", "training", "model"]
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in axis_cols]
    return df[axis_cols + metric_cols]


# ── Curated POC grid: each block isolates ONE axis so leadership can read the effect ───────
def poc_specs() -> list[ExperimentSpec]:
    """A focused, non-exploding grid — every block varies exactly one axis off a common base
    (within_city · loco · ridge · gtfs), so each contrast is attributable to a single axis."""
    S = ExperimentSpec
    specs = [
        # AXIS 1 — predictor set ladder (within-city ranking): does adding the incumbent's
        # approved features, then the agency anchor, move within-city recall?
        S("ours",                 "within_city", "loco",  "ridge"),
        S("ours+approved",        "within_city", "loco",  "ridge"),
        S("ours+approved+agency", "within_city", "loco",  "ridge"),
        # AXIS 2 — LEVEL question: in the ABSOLUTE model, which cross-city leveller restores
        # between-city level best — the incumbent's division dummies (in "ours+approved") or
        # the observed agency anchor on top?
        S("ours+approved",        "absolute",    "loco",  "ridge"),
        S("ours+approved+agency", "absolute",    "loco",  "ridge"),
        # AXIS 2 (transit sub-variant) — GTFS best-fit vs ACS deployable
        S("ours+approved",        "within_city", "loco",  "ridge", transit="acs"),
        # AXIS 3 — training protocol: extrapolation (loco) vs interpolation (80/20)
        S("ours+approved",        "within_city", "80_20", "ridge"),
        # AXIS 4 — model family: linear shrinkage vs non-linear trees
        S("ours+approved",        "within_city", "loco",  "gbm"),
    ]
    return specs

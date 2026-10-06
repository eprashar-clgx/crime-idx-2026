"""The cross-validation harness: fold protocols, estimators, and the one loop that
pairs them (ADR 0008).

Replaces `cv.run_loco`, `cv.run_holdout` and `experiments._run_gbm`, which were three
copies of the same fold loop — the third existing only because trees don't need a
scaler. Two seams remove the duplication:

**Fold protocol** — any iterable of ``(tag, train, test)``. LOCO and stratified 80/20
are the two shipped; a spatially stratified protocol at scale-up is a third generator
and nothing else.

**Estimator** — any object with ``fit(train, predictors, y)`` and ``predict(test)``.
Each estimator owns its OWN preprocessing, which is why "trees skip the scaler" is no
longer a reason for a separate driver.

That last point carries the leakage guarantee. The predictor scaler is **fold-local**:
its mean/sd pool across the TRAIN cities and are then applied unchanged to the holdout.
Because it is computed inside `LinearEstimator.fit`, which only ever receives a train
frame, the holdout is not in scope at the moment the scaler is built — the rule holds
by construction rather than by convention. Contrast `dataset.make_target`, which is
fold-INVARIANT and so may be computed once up front.
"""
from typing import Any, Callable, Iterable, Iterator, Protocol

import numpy as np
import pandas as pd
import statsmodels.api as sm

from regression_modelling.constants import PREDICTOR_COLS
from regression_modelling.models import dataset
from regression_modelling.models.results import FoldRun

Fold = tuple[str | None, pd.DataFrame, pd.DataFrame]

#: Alpha grid for the prediction-side RidgeCV (leave-one-out CV picks alpha per fold).
#: OLS stays the inference estimator (honest HC3 coefficients); Ridge is the prediction
#: estimator — it shrinks the correlated spatial-lag / transit / demographic block so
#: LOCO extrapolation is not thrown by unstable OLS slopes (ADR 0003/0005).
DEFAULT_RIDGE_ALPHAS = np.logspace(-2, 4, 25)

#: Deployed LightGBM defaults; a tuned override is merged on top.
#: `importance_type="gain"` is a reporting flag only — it leaves predictions bit-identical
#: but makes `diagnostics.gbm_gain` readable off any run, so diagnostics never refit.
DEFAULT_GBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
                          n_jobs=1, verbose=-1, importance_type="gain")

#: Optuna-tuned LightGBM overrides per predictor set (ADR 0010 §7). Objective: city-mean
#: LOCO r2_oos on wprop / 20 cities, subject to recall@10 >= untuned; 50 TPE trials, seed 0.
#: City-mean r2 0.342 -> 0.358, recall@10 0.432 -> 0.438; wtotal city-mean r2 0.299 -> 0.323.
TUNED_GBM_PARAMS = {
    "selected_v1": dict(n_estimators=1000, learning_rate=0.013640227848836065, num_leaves=28,
                        min_child_samples=16, subsample=0.52952221807425, subsample_freq=1,
                        colsample_bytree=0.7459236178141202, reg_lambda=18.09648700746888,
                        reg_alpha=0.049630776400581805, random_state=0),
}

#: XGBoost comparison defaults (ADR 0010 §8), matched to the LightGBM ones where the knobs
#: line up: same trees/learning rate/sampling, depth-wise growth capped at 6.
DEFAULT_XGB_PARAMS = dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          tree_method="hist", n_jobs=1, importance_type="gain")


# =========================================================================== #
# Fold protocols                                                              #
# =========================================================================== #
# Each yields (tag, train, test). `tag` names the fold for the `fits` dict and for the
# `holdout_city` column; None means "every test row belongs to its own city", which is
# how the 80/20 split is labeled.

def loco_folds(pooled: pd.DataFrame, city_col: str = "city") -> Iterator[Fold]:
    """Rotating leave-one-city-out: (held_out_city, train, holdout).

    Every city holds out exactly once. Train is all other cities pooled; holdout is the
    single left-out city. This is the EXTRAPOLATION protocol — "score a city we have
    never seen" — and the generalization target ADR 0003 cares about.
    """
    for city in sorted(pooled[city_col].unique()):
        yield city, pooled[pooled[city_col] != city], pooled[pooled[city_col] == city]


def stratified_folds(pooled: pd.DataFrame, test_size: float = 0.20, seed: int = 0,
                     city_col: str = "city", verbose: bool = True) -> Iterator[Fold]:
    """A single random split stratified by city, yielded as one fold: (None, train, test).

    This is the INTERPOLATION protocol — "predict unseen BGs in cities we have partly
    seen". It leaks spatially (a BG's neighbour can sit across the split) and knows each
    city's level, so its scores are OPTIMISTIC. The gap between it and LOCO is exactly
    the value of having seen a city before.
    """
    test = pooled.groupby(city_col, group_keys=False).sample(frac=test_size,
                                                             random_state=seed)
    train = pooled.drop(test.index)
    if verbose:
        print(f"  stratified {int((1-test_size)*100)}/{int(test_size*100)} split "
              f"-> train={len(train)} test={len(test)} across "
              f"{pooled[city_col].nunique()} cities (seed={seed})")
    yield None, train, test


def stratified_split(pooled: pd.DataFrame, test_size: float = 0.20, seed: int = 0,
                     city_col: str = "city") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for callers that want the split itself rather than a fold stream."""
    _, train, test = next(stratified_folds(pooled, test_size, seed, city_col,
                                           verbose=False))
    return train, test


FOLD_PROTOCOLS: dict[str, Callable[..., Iterable[Fold]]] = {
    "loco": loco_folds,
    "holdout": stratified_folds,
}


# =========================================================================== #
# Estimators                                                                  #
# =========================================================================== #
# An estimator owns its own preprocessing. The orchestrator never mentions
# standardization or demeaning, so it cannot leak them across a fold boundary.

class Estimator(Protocol):
    """What the orchestrator requires of a model.

    `fit` receives a TRAIN frame only — never the holdout — so any fold-local state
    (a scaler, an encoder) computed inside it is leakage-safe by construction.
    `predict` then scores a test frame using that retained state.
    """

    predictors: list[str]

    def fit(self, train: pd.DataFrame, predictors: list[str], y: np.ndarray) -> "Estimator": ...

    def predict(self, test: pd.DataFrame) -> np.ndarray: ...


class LinearEstimator:
    """OLS or RidgeCV on a per-city demeaned (optional) and standardized design matrix.

    Owns the **fold-local scaler**. `fit` computes mean/sd on the train frame it is
    handed and stores them; `predict` reuses them unchanged, never re-fitting. Moving
    that computation anywhere the holdout is visible would inflate every LOCO number,
    which is precisely why it lives in here.

    Standardization is not cosmetic for ridge: predictor sds span ~350x across the
    block, and the ridge penalty is scale-dependent, so on raw columns a single alpha
    would shrink a small-sd predictor hundreds of times harder than a large-sd one —
    the penalty would be an artifact of units. For OLS it is a pure reparametrization
    (identical predictions) that exists so `coef_table` reports comparable effect sizes.

    `kind`:
      - ``"ols"``   — statsmodels OLS + HC3 robust wrapper. The INFERENCE estimator
        (honest coefficients + SEs). Exposes `result` / `robust`.
      - ``"ridge"`` — sklearn `RidgeCV`, alpha chosen per fold by leave-one-out CV
        *within the train frame*. The PREDICTION estimator. Exposes `alpha_`.

    Set `demean=True` to pair per-city demeaned predictors with a `*_within_city`
    target (the within/fixed-effects estimator).
    """

    def __init__(self, kind: str = "ridge", demean: bool = False,
                 ridge_alphas=None, robust: str = "HC3", city_col: str = "city"):
        if kind not in ("ols", "ridge"):
            raise ValueError(f"unknown linear estimator {kind!r} (expected 'ols'/'ridge')")
        self.kind = kind
        self.demean = demean
        self.ridge_alphas = DEFAULT_RIDGE_ALPHAS if ridge_alphas is None else ridge_alphas
        self.robust = robust
        self.city_col = city_col
        self.predictors: list[str] = []
        self.scaler: tuple[pd.Series, pd.Series] | None = None
        self.n_train = 0

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demean (optional) then standardize with the retained train scaler."""
        block = (dataset.demean_by_city(df, self.predictors, self.city_col)
                 if self.demean else df[self.predictors])
        mu, sd = self.scaler
        return (block - mu) / sd

    def fit(self, train: pd.DataFrame, predictors: list[str], y: np.ndarray):
        self.predictors = list(predictors)
        block = (dataset.demean_by_city(train, self.predictors, self.city_col)
                 if self.demean else train[self.predictors])
        # Fold-local: these moments come from the train frame and are never re-fit.
        self.scaler = (block.mean(), block.std(ddof=0).replace(0, 1.0))
        Xstd = (block - self.scaler[0]) / self.scaler[1]
        self.n_train = int(len(train))

        if self.kind == "ridge":
            from sklearn.linear_model import RidgeCV
            self.model = RidgeCV(alphas=self.ridge_alphas).fit(Xstd.to_numpy(), y)
            self.alpha_ = float(self.model.alpha_)
            return self

        X = sm.add_constant(Xstd, has_constant="add")
        self.result = sm.OLS(y, X).fit()
        self.robust_result = self.result.get_robustcov_results(cov_type=self.robust)
        self.n_train = int(self.result.nobs)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        Xstd = self._design(test)
        if self.kind == "ridge":
            return np.asarray(self.model.predict(Xstd.to_numpy()))
        return np.asarray(self.result.predict(sm.add_constant(Xstd, has_constant="add")))

    @property
    def summary_tail(self) -> str:
        """One-line per-fold status for the driver's progress print."""
        if self.kind == "ridge":
            return f"alpha={self.alpha_:.3g}"
        return f"adjR2_in={self.result.rsquared_adj:6.3f}"


class GbmEstimator:
    """LightGBM on RAW predictors.

    Trees are level- and scale-invariant, so there is deliberately no scaler and no
    demean here — and because preprocessing is an estimator concern, that difference
    costs nothing structurally. The target still follows the run's mode, so a
    within-city run predicts within-city rank.
    """

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_GBM_PARAMS, **(params or {})}
        self.predictors: list[str] = []
        self.n_train = 0

    def fit(self, train: pd.DataFrame, predictors: list[str], y: np.ndarray):
        import lightgbm as lgb
        self.predictors = list(predictors)
        self.n_train = int(len(train))
        self.model = lgb.LGBMRegressor(**self.params).fit(train[self.predictors], y)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(test[self.predictors]))

    @property
    def summary_tail(self) -> str:
        return f"trees={self.params['n_estimators']}"


class XgbEstimator(GbmEstimator):
    """XGBoost on RAW predictors — the ADR 0010 comparison estimator. Same contract as
    ``GbmEstimator``; ``gbm_params`` are XGBoost params when ``estimator="xgb"``."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_XGB_PARAMS, **(params or {})}
        self.predictors = []
        self.n_train = 0

    def fit(self, train: pd.DataFrame, predictors: list[str], y: np.ndarray):
        import xgboost as xgb
        self.predictors = list(predictors)
        self.n_train = int(len(train))
        self.model = xgb.XGBRegressor(**self.params).fit(train[self.predictors], y)
        return self


def make_estimator(kind: str, demean: bool = False, ridge_alphas=None,
                   gbm_params: dict | None = None) -> Estimator:
    """Build a fresh estimator from a name. The orchestrator calls this per fold, so no
    fitted state is ever shared between folds."""
    if kind in ("ols", "ridge"):
        return LinearEstimator(kind, demean=demean, ridge_alphas=ridge_alphas)
    if kind == "gbm":
        return GbmEstimator(gbm_params)
    if kind == "xgb":
        return XgbEstimator(gbm_params)
    raise ValueError(f"unknown estimator {kind!r} (expected 'ols', 'ridge', 'gbm' or 'xgb')")


# =========================================================================== #
# The orchestrator                                                            #
# =========================================================================== #

def resolve_predictors(pool: pd.DataFrame, predictors: Iterable[str]) -> list[str]:
    """Keep only the requested predictors the frame actually carries.

    A run reports what it really fit, not what it was asked for — so a silently absent
    column shows up in `FoldRun.predictors` rather than as a KeyError three folds in.
    """
    return [p for p in predictors if p in pool.columns]


def run_cv(pool: pd.DataFrame, category: str, mode: str = "lograte",
           predictors: Iterable[str] = PREDICTOR_COLS,
           estimator: str | Callable[[], Estimator] = "ridge",
           split: str = "loco", *,
           folds: Iterable[Fold] | None = None,
           demean_by_city: bool = False, winsor_upper: float | None = None,
           ridge_alphas=None, gbm_params: dict | None = None,
           test_size: float = 0.20, seed: int = 0,
           keep_fits: bool = True, verbose: bool = True) -> FoldRun:
    """Fit one estimator across one fold protocol and return every holdout row, scored.

    This is the single driver. `split` picks a shipped fold protocol (`"loco"` or
    `"holdout"`); pass `folds` directly to supply your own — a spatially stratified
    protocol plugs in here without touching this function or any metric.

    `estimator` is a name (`"ols"`/`"ridge"`/`"gbm"`) or a zero-arg factory returning an
    `Estimator`. A FRESH estimator is built for every fold, so no fitted state leaks
    between them.

    The target is built per fold from the TRAIN frame via `dataset.make_target`. That is
    belt-and-braces rather than necessity — the target is fold-invariant — but it keeps
    `winsor_upper` honest and means a caller cannot smuggle in a pre-computed column
    built under different settings.
    """
    preds = resolve_predictors(pool, predictors)
    if folds is None:
        if split not in FOLD_PROTOCOLS:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(FOLD_PROTOCOLS)}")
        kwargs = {"verbose": verbose} if split == "holdout" else {}
        if split == "holdout":
            kwargs.update(test_size=test_size, seed=seed)
        folds = FOLD_PROTOCOLS[split](pool, **kwargs)

    build: Callable[[], Estimator] = (
        estimator if callable(estimator)
        else lambda: make_estimator(estimator, demean=demean_by_city,
                                    ridge_alphas=ridge_alphas, gbm_params=gbm_params)
    )

    if verbose:
        name = estimator if isinstance(estimator, str) else "custom"
        print(f"{split.upper()} — target mode = {mode!r} ({category}, {name})"
              + (", demeaned-X" if demean_by_city else "")
              + (f", winsor@{winsor_upper:g}" if winsor_upper is not None else ""))

    fits: dict[str, Any] = {}
    parts: list[pd.DataFrame] = []
    for tag, train, test in folds:
        # Target BEFORE the predictor dropna: the `*_within_city` modes z-score against
        # per-city moments taken over the rows present, so building it on the reduced
        # frame would silently shift them. Order matters here; see `dataset.make_target`.
        tr = train.copy()
        tr[dataset.TARGET_COL] = dataset.make_target(tr, mode, category,
                                                     winsor_upper=winsor_upper)
        tr = tr.dropna(subset=list(preds) + [dataset.TARGET_COL])

        model = build().fit(tr, preds, tr[dataset.TARGET_COL].to_numpy())

        te = test.dropna(subset=preds).copy()
        te["y_pred"] = model.predict(te)
        te["holdout_city"] = tag if tag is not None else te["city"]
        parts.append(te)
        if keep_fits and tag is not None:
            fits[tag] = model
        if verbose and tag is not None:
            print(f"  holdout={tag:14} n_train={model.n_train:>5} "
                  f"scored={len(te):>5} {model.summary_tail}")
        elif verbose:
            print(f"  n_train={model.n_train:>5} scored={len(te):>5} {model.summary_tail}")

    scored = pd.concat(parts, ignore_index=True)
    if verbose and len(parts) > 1:
        print(f"  -> {len(scored)} BGs scored out-of-sample across {len(parts)} folds")

    return FoldRun(scored=scored, fits=fits, mode=mode, split=split,
                   category=category, predictors=preds)


def run_loco(pool: pd.DataFrame, category: str, mode: str = "lograte", **kwargs) -> FoldRun:
    """LOCO convenience wrapper over `run_cv` (the extrapolation protocol)."""
    return run_cv(pool, category, mode=mode, split="loco", **kwargs)


def run_holdout(pool: pd.DataFrame, category: str, mode: str = "lograte", **kwargs) -> FoldRun:
    """Stratified 80/20 convenience wrapper over `run_cv` (the interpolation protocol)."""
    return run_cv(pool, category, mode=mode, split="holdout", **kwargs)

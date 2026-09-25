# ADR 0008 — Module boundaries for the modelling work

- **Status:** Accepted (2026-09)
- **Date:** 2026-09-23
- **Related:** ADR 0001 (task-oriented module restructure), ADR 0003 (pooled fit +
  grouped/LOCO CV), ADR 0005 (weighted-score math promoted to the foundation),
  `src/regression_modelling/models/{cv,experiments,model,baseline}.py`,
  `notebooks/regression_modelling/models/02_regression_prediction.ipynb`,
  `04_bg_comparison.ipynb`

## Context

A week of fast iteration (prediction rework, incumbent baseline, 4-axis experiments,
recall metrics) left `regression_modelling/models/` carrying most of the project's
complexity in four files and 32 notebook-local functions. An architecture review
surfaced the friction:

- **`cv.py` spans seven responsibilities** in 609 lines: city/table loading, target
  construction, split protocol, fold fitting, metric computation, table assembly and
  plotting.
- **32 inline notebook functions**, several of which are library code. `rank_metrics`
  is pasted in two notebooks; `04` re-derives loading, fitting and scoring that `cv.py`
  already provides.
- **Four producers of the "run" dict** agree on keys only by convention.
  `baseline.existing_model_run` hand-builds a cv-shaped dict without importing `cv`.
- **Anti-drift helpers are already being bypassed.** `cv.full_coverage_cities()` exists
  so the city set "can never drift from the per-city config"; both notebooks ignore it
  and inline the same comprehension. `04` additionally hardcodes `c != "columbus"`,
  while `02` detects degenerate cities dynamically.
- **11 functions are dead** (no callers anywhere). Two of them are competing Moran's I
  implementations, and the dead one computes KNN weights on unprojected degrees.

Separately, the POC is scaling from 5/9 cities to 15-20, and ultimately to as many
cities as have usable incident data.

## Decision

### 1. The outer `src/` structure stands; `regression_modelling` keeps its name

ADR 0001's "one folder per statistical task" still holds. The friction is *inside*
`models/`, not in the top-level split, so the fix belongs there.

### 2. `models/` splits four ways, by the question each module answers

| module | purpose |
|---|---|
| `dataset` | **what we model**: eligible cities, eligible block groups, and the target column on its stated scale |
| `harness` | **how we fit**: split protocol, fold fitting, estimator choice. Produces a `FoldRun` |
| `scorecard` | **what we report**: every metric and the comparison table, one name per concept |
| `diagnostics` | **why it behaves this way**: per-city and per-feature explanation of a `FoldRun` |

`model.py` becomes `inference` — the explanatory path (coefficients, HC3, Moran's I),
which answers a different question from the prediction path and keeps its own seam.

A finer split (separate `pool`, `target`, `protocol`, `fit` modules) was rejected:
those four would be shallow, with interfaces nearly as complex as their bodies, and
understanding a single run would mean bouncing between eight files.

### 3. The incumbent is promoted out of `models/`, as `incumbent_eval`

Scoring the deployed model against observed crime involves no fitting and no split
protocol; it is a different task, which is precisely why `baseline.py` ended up
*imitating* the run schema instead of using it. It moves to `src/incumbent_eval/`.

Named for its subject rather than its role: "baseline" is relative (it collides with
the archived per-city OLS baseline, a different concept in the same glossary), and
`incumbent_eval` parallels `carrier_eval` — same verb, different ground truth.

`incumbent_eval` and `carrier_eval` both score the deployed model and may eventually
want a shared scoring interface. **We are not building that seam now.** One adapter is
a hypothetical seam; two is a real one. The overlap is recorded, not implemented.

### 4. Scale-up is a city pool, not a module

Going to 15-20 cities is a *statistical* move — LOCO with 5 cities estimates the
between-city level from 4 points per fold — not an engineering one. The same pulls,
the same fit and the same metrics apply; only the pool changes. A `scale_up` module
would either be empty or a fork of the POC, and, worse, would give city-set drift a
second home to live in.

The city pool gets one owner, `target_pool(target)`, returning the cities that both
**can** support a target (source capability, from `CityConfig.property_only`) and
**did** produce usable crime (observed usability: degenerate all-zero extracts
excluded, each exclusion logged with its reason). This replaces
`full_coverage_cities()`, the proposed `property_cities()`, and both notebook-local
pool definitions.

Docs name the concept; they do not assert a city count. Counts in prose are what
drifted.

### 5. `harness` takes a fold generator, not a protocol string

The split protocol is the thing we expect to vary. Today: LOCO (extrapolation) and
stratified 80/20 (interpolation). Plausibly next: spatially-blocked folds, if LOCO
still underperforms at 15-20 cities. `run_loco` currently hardcodes `loco_folds`.

Passing a protocol in means a new protocol is one new implementation, and every metric
and diagnostic keeps working unchanged.

**This does not pre-approve spatial stratification.** ADR 0003 rejected random k-fold
because it leaks spatial signal between neighbouring block groups; spatially-blocked
folds are consistent with that reasoning, but they change the estimand — "predict an
unseen neighbourhood" is a weaker claim than "predict an unseen city," and unseen-city
extrapolation is what ADR 0003 exists to measure. Adopting it would amend ADR 0003 and
needs its own decision.

### 6. Notebooks are drivers

A notebook may define a function only if it **takes already-computed frames and returns
a figure or a styled frame**. Anything computing a number that reaches a deck belongs in
`src/`. The rule is deliberately checkable from a signature.

### 7. Sequencing: vocabulary first

Three landings, in order:

1. **Vocabulary and docs** — metric vocabulary in `CONTEXT.md`, the misleading metric
   names corrected, the phantom `within_city_r2` removed, `target_pool` introduced,
   `architecture.md` corrected to describe what exists *today*.
2. **Module split** — the boundaries above; notebook logic moves behind them.
3. **`architecture.md` rewritten** to describe the new structure.

Landing 1 is a prerequisite, not a warm-up: `scorecard` cannot be named while seven
names mean R², and `dataset` cannot own "which cities" while the pool is defined three
different ways.

## Consequences

- One place to look per question, and `cv.py` stops being the file everything lands in.
- The `FoldRun` contract makes the incumbent just another run, so any diagnostic written
  for the refresh works on it for free.
- `scorecard` is the project's first cheaply testable surface: it needs neither BigQuery
  nor a model fit, so two synthetic runs with known ranks can assert the comparison
  table. The repo currently has zero tests, largely because nothing else is this cheap.
- Import churn across both notebooks, and a re-execution to confirm numbers are
  unchanged.
- Dead code is removed *after* the boundaries are set, not before: whether something is
  dead depends on which module claims it. Moran's I is the worked example — it survives
  in `inference`, but only `residual_spatial`; `spatial_moran` is deleted as a
  near-duplicate that runs KNN on unprojected degrees.
- Scale-up work will feel like configuration rather than development. If that turns out
  to be false — if 20 cities forces chunked loading or BigQuery-side aggregation — that
  is a genuine second task and this decision should be revisited.

---

## Landing 2 — implemented

The split is done. Final module names differ slightly from the sketch above (`training`
rather than `harness`, `metrics` rather than `scorecard`), and three design questions
were settled during implementation.

### Final layout

| module | question it answers | lines |
|---|---|---|
| `dataset.py` | What are we modelling? Eligibility, pooled frame, target, predictor sets. | 326 |
| `training.py` | How well does it score a city we have not seen? Fold protocols, estimators, orchestrator. | 338 |
| `results.py` | What does a run look like? The `FoldRun` contract. | 63 |
| `metrics.py` | How good is it? | 268 |
| `diagnostics.py` | Why does it fit, or not? | 139 |
| `inference.py` | What does the relationship say? | 112 |
| `incumbent.py` | What must we beat? | 104 |

`cv.py`, `experiments.py`, `model.py` and `baseline.py` are deleted. `src` grew ~180
lines net while absorbing ~130 lines of function definitions **out of** notebook 02
(cells 21/27) and dropping 11 dead functions.

### Decision: an estimator owns its own preprocessing

Rejected: the orchestrator standardizes and asks each estimator whether it wants it
(`needs_scaling`). That flag is a shallow interface — an estimator detail leaking
through the seam — and every later preprocessing need adds another branch.

Chosen: `fit(train, predictors, y)` / `predict(test)`. `LinearEstimator` computes the
**fold-local scaler** inside `fit`; `GbmEstimator` has no scaler at all, because trees
are scale-invariant.

This is what removed the third driver. `experiments._run_gbm` existed **only** to skip
the scaler, and in doing so re-implemented fold rotation. Once preprocessing became an
estimator concern, `run_loco`, `run_holdout` and `_run_gbm` collapsed into one
`run_cv` loop. It also makes the leakage rule structural rather than conventional: the
holdout frame is not in scope at the moment the scaler is computed, so there is no line
a future contributor can move to break LOCO.

Standardization is not cosmetic for ridge — predictor sds span ~350x, and the ridge
penalty is scale-dependent, so on raw columns one alpha would shrink a small-sd
predictor hundreds of times harder than a large-sd one. (Checked: the chosen alphas sit
at grid indices 14-16 of 24, so the penalty is not railing at either bound.)

### Decision: the target is fold-invariant and may be precomputed

`make_target` groups by city, so a city's target depends only on its own rows. Verified
exhaustively: `max |precomputed - per-fold| = 0.000e+00` for both `lograte` and
`lograte_within_city`. `dataset.load_pool(target, mode=...)` therefore attaches `_y` up
front, and `make_target` stays public for one-off variants.

This is the **opposite** of the scaler, and the contrast is the module boundary: the
scaler pools across train cities (fold-local, lives in `training`), the target does not
(fold-invariant, lives in `dataset`).

One caveat found while verifying: the `*_within_city` modes are invariant to dropping
other *cities* but not to dropping *rows within* a city, since the z-score moments are
taken over the rows present. The target must therefore be built **before** the
null-predictor `dropna`. The retired `_run_gbm` did the reverse, so a `within_city`
spec silently produced a different target for GBM than for ridge. No reported number
was affected — no live notebook ran that combination — and `run_cv` now uses one order.

### Decision: `FoldRun` is a frozen dataclass in its own module

The three old drivers already returned the same dict shape by hand-matched convention;
`_run_gbm`'s docstring said so explicitly. That convention was load-bearing — it is why
one metric function serves ridge, GBM, LOCO and 80/20 alike — but nothing enforced it,
and `run_holdout` had already drifted an unused `"fit"` key.

`FoldRun` writes the contract down. It computes nothing. Its value is that a new fold
protocol (spatial stratification, at scale-up) gets **every existing metric for free**
provided it returns one.

It lives in `results.py`, not `training.py`, so the dependency stays one-way:
`training` and `metrics` both import `results`, and `metrics` never imports `training`.

### Verification

Old and new drivers were compared directly before the old files were deleted: **all 7
linear configurations** (ridge/ols x loco/holdout, demeaned within-city, winsorized) and
**both GBM `lograte` configurations** reproduce bit-for-bit, `max|Δy_pred| = 0.000e+00`.
Metric tables compare `.equals()`-identical. All three notebooks re-executed clean and
every headline number is unchanged.

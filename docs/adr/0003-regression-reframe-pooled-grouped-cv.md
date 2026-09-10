# ADR 0003 — Regression reframe: pooled grouped-CV, archived per-city baseline, rate target

- **Status:** Accepted (amended 2026-09-10 — see Update below)
- **Date:** 2026-08-27
- **Related:** ADR 0001 (supersedes its "no train/test split by design" clause and the
  `log(count+1)`-primary target), ADR 0005 (weighted-score promotion), `docs/eda_plan.md`

## Context

ADR 0001 fixed the model as an **inferential OLS fit per city** — no train/test split by
design, predictors z-standardized, HC3 robust SEs, Moran's I on residuals. That was right
for a coefficient-reading POC on two cities. We are now scaling to 6 POC cities with a
national roll-up as the endgame, and the goal shifts from "read coefficients in one city"
to "**predict block-group crime in cities we have not fit**." That is a generalization
claim the per-city protocol cannot measure.

Two things become important:

1. **Prediction vs inference is not the real fork.** Both report the same diagnostics —
   coefficients, robust SEs, residuals, Moran's I. What actually changes is the
   **fitting/evaluation protocol**: separate per-city fits → a pooled fit with
   cross-validation.
2. **Spatial autocorrelation makes naive k-fold dishonest.** Neighboring block groups are
   spatially autocorrelated (the very thing Moran's I flags), so a random fold puts a BG
   in test whose neighbor is in train and inflates apparent performance. The city split is
   also the generalization we actually care about.

## Decision

**Pooled model with grouped cross-validation becomes the primary protocol; the per-city
fit is archived as a heterogeneity baseline.**

- **Primary — pooled + grouped CV by city.** Fit on the pooled BG table across POC cities;
  cross-validate with **leave-one-city-out (or leave-k-cities-out)** folds. This matches
  the POC vision ("fit on train cities, check on test cities") and the national scale-up,
  and avoids both cross-city and spatial leakage. Random k-fold is **not** used — it leaks
  spatial signal.
- **Archived — per-city fits, kept runnable.** The per-city OLS is retained (not deleted)
  as a **cross-city coefficient-heterogeneity** diagnostic: does the convenience-store or
  transit effect differ Chicago vs Houston? That is a substantive RTM finding pooling
  loses. It is a documented baseline, not a headline output.
- **Shared diagnostics on both layers.** Coefficient tables, clustered/HC3 robust SEs,
  residuals, and **Moran's I** are reported for both the pooled and per-city fits.

### Target column

- **Primary target: `crime_rate` (per 1,000 population), modeled directly in OLS.**
  Zero/NaN-population BGs are **dropped before the fit** (rates are already set to NaN,
  never inf — see the domain rule); we drop those rows rather than impute them.
- **Comparators in the same run:** `log(count + 1)` (the former ADR-0001 primary) and the
  **weighted crime rate** (relative-risk composite; math promoted to the shared foundation
  per ADR 0005). All three are reported; `crime_rate` is the headline.

### Exposure

Exposure (population at risk) is handled by the **rate denominator**, not a GLM offset.
The current machinery is OLS on `log(count+1)`, which has **no offset mechanism** — the log
transform only variance-stabilizes. Modeling `crime_rate` puts population in the
denominator directly. Residual heteroskedasticity from small-population BGs is absorbed by
the **HC3 robust SEs** already in the ADR-0001 machinery; WLS-by-population is a later
option if residuals demand it. (A Poisson/NB GLM with a true `log(pop)` offset was
considered and deferred — it is a larger departure than this POC needs.)

### Filtered geoid set is load-bearing

Because zero/NaN-pop BGs are dropped, the fit runs on a **filtered geoid set**. Everything
spatial must be rebuilt on that same set:

- **Moran's I and spatial CV folds** are computed on the post-drop geoids, not the full
  boundary set.
- **Bias-testing** (ADR 0004) joins its protected-attribute table to **this same filtered
  geoid set** at test time, so the correlation checks run on the fitted population.

### Multicollinearity (Step 3)

VIF and the feature correlation matrix are **diagnostics only — no automatic pruning.** A
human decides every drop with a written rationale (mirroring the bias-test philosophy in
ADR 0004). Rationale: the transit hurdle-form pair is *engineered* to be orthogonal and
must not be "fixed," and exposure-carrying terms may be intentionally retained despite
collinearity. High VIF (>10 severe, >5 flag) and `|r| > 0.8` pairs surface for review, not
deletion.

## Consequences

- ADR 0001's "no train/test split by design" and `log(count+1)`-primary decisions are
  **superseded** for `regression_modelling`. The inferential per-city fit survives as the
  archived baseline, so no capability is lost (deletion test passes).
- `models/` gains a pooled-fit + grouped-CV path alongside the existing per-city
  `fit_and_report`; the per-city function stays.
- `build_model_table` / the fit entry point must **drop zero/NaN-pop rows** and expose the
  surviving `geoid` set so spatial weights and the bias table align to it.
- Cross-city coefficient comparison becomes a first-class (if secondary) artifact.

## Update (2026-09-10) — within-city objective, dual target, target-paired predictors, dual protocol

Refines this ADR after clarifying the product intent with the prior data scientist and a
grilling session on the standardization design. Nothing above is reversed; the target and
protocol are made precise.

### Objective (made explicit)

The endgame is to **predict within-city block-group crime variation** and have it
**generalize to cities we have not fit** — *not* to reproduce cross-city level differences.
This reframes the "predict BG crime" goal of the original ADR: the transferable signal is
the *within-city gradient*, not each city's baseline level.

### Target — run BOTH forms (per weighted category)

The single modeled quantity is the **weighted (relative-risk) rate** promoted in ADR 0005,
on a **resident-population denominator** (coherent with the national `*_pt_u` benchmarks;
daytime denominator would break that coherence). Two target forms are run side by side:

- **Absolute — `log(weighted rate)`** (`lograte`, one intercept): the level/reported form.
  Under LOCO its absolute error is expected to be *poor* (a held-out city's level does not
  transfer) — that is a finding, not a bug.
- **Within-city — per-city z-score of `log(weighted rate)`** (`lograte_within_city`): the
  headline. Each city standardized to its own mean/sd, so cross-city level+scale is removed
  and the fit learns *relative within-city* risk. The earlier hesitation to make this the
  headline is resolved: within-city variation **is** the objective.

### Target-paired predictor treatment (the key refinement)

Standardizing the target per city but leaving predictors pooled is internally inconsistent
(it attenuates the within-city slope). The fix is **target-paired**:

- **Absolute target → pooled/raw predictors** (z-standardized on the train fold). Keeps the
  between-city predictor variation that legitimately explains between-city *level*.
- **Within-city z-score target → per-city demeaned predictors** (subtract each city's own
  predictor mean, then scale). This is the textbook **within / fixed-effects estimator**
  (Frisch–Waugh: demean both sides per city), giving the un-attenuated within-city slope.

**Why this survives test time (and city fixed effects do not).** City *dummies* fail
out-of-sample: a held-out city has no learned intercept. Per-city *demeaning of predictors*
is a different operation — it uses the held-out city's **own observed X** (predictors are
not the outcome), so at LOCO predict time the held-out city is demeaned by its own X-means
with no leakage and no dummy required. The constraint "we can't account for a city fixed
effect at test time" applies to the target's *level*, not to demeaning the predictors.
(Implemented as `cv.fit_fold(..., demean_by_city=True)`.)

### Protocol — run BOTH splits

- **Stratified 80/20 (interpolation).** Random split stratified by city (every city ~80/20).
  "Predict unseen BGs in cities we have partly seen." Spatially leaky and city-level-aware →
  **optimistic** R². Implemented as `cv.run_holdout`.
- **Leave-one-city-out (extrapolation).** The honest generalization number (unchanged from
  the Decision above). Implemented as `cv.run_loco`.

The **gap between the two** quantifies the value of having seen a city before (and the
spatial-leakage optimism the original ADR warned about).

### City sets

- **Property composite (`wprop`) over all 10 POC cities** (burglary+larceny+mvt; the 5
  `property_only` cities are usable here since violent coords are not required).
- **Total composite (`wtotal`) over the 5 full-coverage cities.**

### Metric mapping (targets are on different scorecards — never cross-compare)

| Target | Stratified 80/20 | LOCO |
| --- | --- | --- |
| `log(weighted rate)` (absolute) | R² + MAE + Spearman | R² + MAE + Spearman |
| within-city z-score | within-city R²/MAE + rank/concentration | **rank/concentration only** |

R²/MAE on the z-score target is legitimate only under 80/20 (train-city moments known); a
genuinely held-out city has no moments to form the z-score without leakage, so LOCO uses the
affine-invariant **rank/concentration** metrics (Spearman, capture@top-20%-pop, Gini). R² on
the z-score target reads as **within-city variance explained** and is not comparable to the
absolute target's R².

### Deferred

City fixed effects / an explicit spatial control (spatial-error or spatial-lag-of-outcome
model) are **deferred**. For now residual choropleths + Moran's I on both targets surface
any leftover between-city / spatial pattern and motivate the next iteration.

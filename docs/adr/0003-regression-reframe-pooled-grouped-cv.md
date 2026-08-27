# ADR 0003 — Regression reframe: pooled grouped-CV, archived per-city baseline, rate target

- **Status:** Accepted
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

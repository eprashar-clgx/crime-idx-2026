# Transit predictors — statistical treatment

Companion to `transit_eda_plan.md`. Covers the functional form and collinearity handling
for the transit block-group (BG) predictors before they enter the regression. Read
`transit_eda_plan.md` §5 for the raw feature definitions and the distribution EDA that
motivates the log transforms.

## 1. Context

Distribution EDA on the 10-city artifact (`data/interim/sources/transit.parquet`, 7,863 BGs)
established two things about the supply features:

1. **Right skew** — `stop_density`, `nearest_stop_m`, `service_intensity` (and the raw
   counts) are heavily right-skewed. `log1p` compresses them to ~Gaussian and preserves the
   city ordering (e.g. SF denser/closer than Jacksonville). Handled by the
   `TRANSIT_MODEL_TRANSFORMS` spec (`constants.py`), applied by
   `feature_engineering.transforms.apply_transforms`.
2. **Structural zeros** — ~15% of BGs have no stop at all (8.7% in SF, 29.6% in
   Jacksonville). These are not "a little transit"; they are a different population. We
   split them out into a binary `transit_has_transit = 1[stop_count > 0]` indicator.

Bounded features (`overnight_stop_share`, `route_mode_diversity` ∈ [0,1]) are left raw.

## 2. The correlation problem

Pearson correlation on the **model-form** columns (what OLS and VIF actually see; Spearman
is invariant to the log so it can't surface this) shows the supply features collapse to
essentially one latent "transit supply" factor:

| Pair | Pearson |
|---|---|
| `has_transit` ~ `service_intensity(log)` | **0.89** |
| `stop_count(log)` ~ `service_intensity(log)` | 0.86 |
| `stop_density(log)` ~ `service_intensity(log)` | 0.84 |
| `overnight_count(log)` ~ `overnight_share` | 0.77 |
| `stop_density(log)` ~ `nearest_stop(log)` | −0.76 |

After redundancy pruning we retain four features on three dimensions:

- `transit_service_intensity_log` — intensive supply magnitude (represents the cluster)
- `transit_nearest_stop_m_log` — access / proximity (defined even for stopless BGs)
- `transit_overnight_stop_share` — H2 nighttime exposure (bounded, city-comparable)
- `transit_has_transit` — extensive margin (any transit at all)

The remaining problem: **`has_transit` and `service_intensity(log)` correlate 0.89**,
because `log1p(service) = 0` *exactly* when `has_transit = 0` (a stopless BG has zero
trips). Keeping both raw in a linear model inflates VIF and makes the two coefficients
unstable and hard to interpret.

## 3. Possible fixes

**A. Drop `has_transit`; keep `service_intensity(log)` only.**
Simplest, one parameter. But it forces a *single slope* through both the point-mass of
stopless BGs (at `log = 0`) and the continuous served mass. That assumes the effect of
*none → a little* transit equals the per-unit effect of *a little → a lot*. Since stopless
BGs are a structurally different population (suburban, low-density), the point-mass leverages
the fit and the slope conflates the **extensive** margin (transit vs none) with the
**intensive** margin (more vs less) — biased if the two margins differ, which they plausibly
do.

**B. Keep both raw.** Coefficients remain jointly unbiased but VIF is high (~0.89
collinearity), standard errors inflate, and the individual coefficients are hard to read.

**C. Hurdle form (chosen).** Keep the indicator and *center the continuous term within the
served mass*:

- `z = 1[stop_count > 0]` (`transit_has_transit`)
- `w = log1p(service) − mean(log1p(service) | stop_count > 0)` for served BGs; `w = 0` for
  stopless BGs.

## 4. Chosen fix and rationale

We use the **hurdle form (C)**. The decisive property is that centering `w` on the served
mass makes `z` and `w` **exactly orthogonal**, not merely less correlated:

```
E[w] = 0                    (w is centered on the positive mass, zero elsewhere)
E[z·w] = E[w | z=1]·P(z=1) = 0·p = 0
⇒ Cov(z, w) = E[z·w] − E[z]E[w] = 0
```

So VIF between the indicator and the intensive term is ≈ 1 by construction. Benefits:

- **Separates the two margins.** `β_z` = crime shift from having *any* transit (a node of
  activity/opportunity); `β_w` = marginal effect of *more* service *among served BGs*.
- **Robust to the zero point-mass.** The intensive slope is estimated only on served BGs, so
  stopless BGs no longer leverage it.
- **Cheap.** One extra parameter at n = 7,863.

Caveats / discipline:

- `w` must be centered on the **served** subset, not the full sample, or the orthogonality
  is lost.
- `β_w` is interpreted "among served BGs"; remember the centering when back-transforming or
  when adding interactions.
- The orthogonality is only between `z` and its own `w`. `w` still carries moderate
  correlation with `nearest_stop(log)` (−0.61) and `overnight_share` — those are kept
  deliberately as separate dimensions, to be checked with VIF in the regression step.

## 5. Where this lives in code

- `regression_modelling/constants.py` — `TRANSIT_MODEL_TRANSFORMS` (functional-form spec),
  the predictor family groups, derived `PREDICTOR_COLS`.
- `regression_modelling/feature_engineering/transforms.py` — `apply_transforms` emits the
  `{col}_log` columns, the `transit_has_transit` indicator, and (hurdle) the centered `w`.
- Retention / correlation evidence: `context/transit_eda/corr_pearson.png` (scratch,
  gitignored).

Open follow-up: when the BigQuery POI pull populates the `transit_risky_*` co-location
features (currently zero offline), add them to `TRANSIT_PREDICTORS` +
`TRANSIT_MODEL_TRANSFORMS` (a two-line registry edit) and re-run this correlation. The
analysis helpers skip zero-variance columns, so they stay out of the matrix until populated.

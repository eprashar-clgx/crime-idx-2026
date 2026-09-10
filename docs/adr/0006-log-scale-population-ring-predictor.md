# ADR 0006 — Log-scale the `pop_est_5mile` population-ring predictor

- **Status:** Implemented (2026-09)
- **Date:** 2026-09-10
- **Related:** ADR 0003 (regression reframe, log1p target), `constants.py`
  (`DEMOGRAPHIC_MODEL_TRANSFORMS`), `data_wrangling/dataset.py`,
  `data_wrangling/agency.py`, `notebooks/regression_modelling/models/03_agency_comparison.ipynb`

## Context

`pop_est_5mile` (estimated population within a 5-mile ring of the block-group centroid)
entered the fit-set **raw**. It is a population **count**, not a share: it spans ~46 →
2.3M with skew ≈ 5.7. Every other right-skewed count/distance predictor in the pipeline is
log-compressed (`PROPERTY_MODEL_TRANSFORMS`, `TRANSIT_MODEL_TRANSFORMS`); this one was the
lone raw-scale magnitude feature.

Under the model's `log1p` target with a ridge/robust fit, a raw feature that spans six
orders of magnitude is dangerous: when it lands in a sparse model with a non-shrunk
coefficient, `expm1` back-transform explodes on the largest-population units. The
2026-09-10 **agency-level ablation** (`03_agency_comparison.ipynb`) caught it directly —
forward selection's held-out adj R² *collapsed to −0.76* the moment raw `pop_est_5mile`
was added (CV had nudged *up*, masking the instability). The full our-set was also
depressed: test adj R² **0.219**.

The existing national baseline model never had this problem because it log-scales the same
feature (`zlg10_pop_est_5mile`).

## Decision

**Log1p-transform `pop_est_5mile` into `pop_est_5mile_log` as its standard model form**,
centralized in `regression_modelling.constants`:

- New `DEMOGRAPHIC_MODEL_TRANSFORMS = {"pop_est_5mile": "log1p"}` (single source of truth),
  and a derived `DEMOGRAPHIC_MODEL_PREDICTORS` that renames the transformed column to the
  `apply_transforms` `{col}_log` convention. `PREDICTOR_COLS` is built from the model-form
  list, so the fit-set carries `pop_est_5mile_log`, never the raw count.
- `DEMOGRAPHIC_PREDICTORS` keeps the **raw** `pop_est_5mile` — it is the transform input and
  the imputation target (`MEDIAN_FILL`), and distribution EDA (`01_eda`) still profiles the
  raw column. This mirrors the raw-vs-model split already used for property predictors.
- Both consumers apply the same spec so the change "flows everywhere": the BG pipeline
  (`dataset.build_model_table`, feeding the inference + prediction notebooks via
  `models/cv.load_pooled_table`) and the agency roll-up (`agency.rollup_predictors_to_agency`).
- `pop_ch_1mile` (bounded % change, −83 → +94) is left raw — it is not a skewed magnitude.

`log1p` (not `log10`) is used for consistency with the rest of the pipeline and because
predictors are z-standardized at fit time, so the two differ only by a constant that washes
out after standardization.

## Consequences

- **Stability + lift.** The −0.76 collapse disappears; the standalone our-set agency adj R²
  rises **0.219 → 0.358** from this one change. Downstream `*_model_table.parquet` caches
  and `agency_predictors.parquet` must be rebuilt (`refresh=True`) to add the `_log` column.
- Any code selecting `pop_est_5mile` from the fit-set must now use `pop_est_5mile_log`
  (EDA/imputation still reference the raw name — intentionally).
- Reinforces a pipeline rule: **raw magnitude counts/distances are log-compressed before a
  `log1p`-target fit**; only bounded shares/ratios stay raw.

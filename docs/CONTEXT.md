# CONTEXT — glossary & module vocabulary

Shared vocabulary for `crime-idx-2026`. Names here are load-bearing: use them
exactly in code, docs, and discussion. Update this file when a concept changes.

## Guiding structure

- **`src/` = one folder per statistical task.** Each top-level module has a single
  goal. No ambiguous names (`core`, `prediction`, `utils`) — name a module for what
  it does.
- **`config.py` holds paths only.** All constants, dataclasses, and registries live
  in a per-module **`constants.py`** (never a single global constants file — that
  becomes the next `core`).
- **Notebooks live outside `src/`**, in `notebooks/` mirroring the module tree.
  Notebooks are drivers; importable logic lives in `src/`.
- **Data lives outside `src/`**, catalogued in `data/README.md`.

## Modules

### `crime_blockgroup_mapping` (shared foundation)
Goal: **assemble a city's block-group crime table — counts, rates, population —**
the substrate both `regression_modelling` and `carrier_eval` build on. The one
place both tasks import from; replaces the old `core`. Holds:
- **city registry + crime taxonomy** — `CityConfig`/`CITIES`, `NIBRS_TO_CATEGORY`,
  `CHICAGO_FBI_TO_CATEGORY`, `CRIME_CATEGORIES` (in `constants.py`); paths in
  `config.py`.
- **BG-crime assembly** — load boundaries/block groups, load crime, spatial-join
  crimes to block groups, map crime categories, aggregate to block-group counts.
- **population + rate normalization** — population source, LODES jobs, per-1,000
  rate normalization (incl. daytime-adjusted), and the **weighted-score / relative-risk
  math** (`compute_weighted_scores`, `extract_national_rates`) shared by both tasks
  (ADR 0005).
- **general plot helpers** — reusable maps/plots (base folium/choropleth/basemap).

### `regression_modelling` (POC crime-risk model rebuild)
Goal: predict block-group crime for POC cities. Sub-modules:
- **`data_wrangling`** — BigQuery/GCS ingestion (`load_sql`, BQ/GCS clients, cached
  pulls), feature assembly, and the features-⋈-target join. Owns all task SQL under
  `sql/{build,pull,explore}` (one `SQL_DIR`).
- **`feature_engineering`** — transforms on raw predictors: winsorize, scale, log,
  spatial terms.
- **`distributions`** — exploratory analysis: counts, distributions, correlations,
  VIF; POI store-EDA (`eda.py`) + task-specific EDA plots (`plots.py`). Its explore SQL
  templates live under `data_wrangling/sql/explore` (co-located with the loader).
- **`models`** — model fitting and diagnostics. Primary path: **pooled fit with
  grouped (leave-one/-k-cities-out) cross-validation** on `crime_rate`; comparators
  `log(count+1)` and weighted rate (ADR 0003). Archived **per-city** OLS baseline for
  cross-city coefficient heterogeneity. Coefficient tables, HC3 SEs, and **Moran's I**
  (spatial autocorrelation of residuals) on both.
- **`bias_testing`** — checks that predictors correlate with crime and **not** with
  protected attributes (e.g. race). Owns a **separate protected-attribute table**
  (`PROTECTED_ATTRIBUTES`, keyed by `geoid`, never a predictor) and reports **conditional
  association** as a human-review diagnostic (ADR 0004). Distinct from Moran's I.
- **`logging`** — run/experiment logging.

### `carrier_eval` (side-quest)
Goal: evaluate how well the **existing** model performs against carrier insurance
data. Dedicated, clearly-separated space (side-quests recur and must not leak into
the main POC). Holds carrier-evals ingestion/aggregation and its own maps. The **score
reconstruction math** (`compute_weighted_scores`, `extract_national_rates`) now lives in
`crime_blockgroup_mapping` (ADR 0005) and is imported from there.

## Design vocabulary (from improve-codebase-architecture)

- **module** — a folder/file with one goal.
- **interface** — the small surface a module exposes; the test surface.
- **depth** — a deep module hides a lot behind a small interface; shallow modules
  leak internals.
- **seam** — a clean import boundary between modules.
- **leverage** — one interface serving many cases (e.g. one `FacilitySpec`, N POIs).
- **locality** — related logic lives in one place, so a change touches one module.
- **deletion test** — if removing a module/function loses no capability, delete it.

## Domain nouns (do not rename)

- **`geoid`** — the 12-char block-group join key. Everything normalizes to it before
  merging (`bg_key` → `geoid`).
- **7 primary crimes** — violent: murder, rape, robbery, assault; property:
  burglary, larceny, mvt. **Vandalism is excluded** from composite totals.
- **weighted score** — equal-representation average of *relative risks* (local rate /
  national `*_pt_u` rate), not raw sums. The math (`compute_weighted_scores`,
  `extract_national_rates`) lives in **`crime_blockgroup_mapping`** (shared), imported by
  both tasks (ADR 0005). See `docs/weightage_methodology.md`.
- **prediction target** — primary is the **weighted (relative-risk) rate**, run in TWO
  forms (ADR 0003 §Update 2026-09-10): **`log(weighted rate)`** (absolute level, one
  intercept) and its **per-city z-score** (`*_within_city`, the within-city "risk index"
  headline). Modeled for `wprop` (property composite, 10 cities) and `wtotal` (total, 5
  cities); resident-pop denominator. `crime_rate` / `log(count+1)` are now comparators.
  Zero/NaN-pop BGs **dropped** before the fit. Report HC3 SEs; check Moran's I.
- **target-paired predictors** — pair the predictor treatment to the target (ADR 0003
  §Update): **pooled/raw predictors** with the absolute `log`-rate target; **per-city
  demeaned predictors** with the within-city z-score target (the within / fixed-effects
  estimator). Demeaning uses each city's OWN observed predictors, so it survives test time
  — unlike city dummies, which a held-out city has no intercept for. (`cv.fit_fold(...,
  demean_by_city=True)`.)
- **exposure** — population at risk. A big-population BG mechanically has more crimes.
  Handled by the **rate denominator** (`crime_rate = count/pop × 1000`), not a GLM offset —
  plain OLS on log-count has no offset, so log alone does **not** handle exposure (ADR 0003).
- **grouped CV / leave-one-city-out (LOCO)** — the **extrapolation** protocol: pool BGs
  across cities, cross-validate with folds **held out by city** ("predict an unseen city").
  Random k-fold is not used — it leaks spatial autocorrelation across neighboring BGs
  (ADR 0003). `cv.run_loco`.
- **stratified 80/20 split** — the **interpolation** protocol: a random split stratified by
  city (every city ~80/20), "predict unseen BGs in cities partly seen." Spatially leaky and
  city-level-aware → **optimistic** R². The gap vs LOCO measures the value of having seen a
  city before. `cv.run_holdout` (ADR 0003 §Update).
- **per-city baseline** — the archived per-city OLS fit, kept runnable to inspect
  **cross-city coefficient heterogeneity** (does an effect differ Chicago vs Houston?). A
  documented baseline, not a headline output (ADR 0003).
- **protected attribute / flagged variable** — a variable (e.g. race) used **only** for
  bias testing, never a predictor. Lives in a **separate bias-testing-only table** keyed by
  `geoid`; never in `FEATURE_SOURCES`/`PREDICTOR_COLS` (ADR 0004). Owned by
  `bias_testing.PROTECTED_ATTRIBUTES`.
- **conditional association** — the bias test: whether a predictor's crime signal
  **survives conditioning on** a protected attribute (proxy check), reported alongside raw
  correlation. Soft-flags for human review; never an auto-drop (ADR 0004).
- **imagery predictor** — a **structure-level** IDAP source (e.g. structure density,
  roof/condition) re-aggregated to BG `geoid` via the build/pull pattern (not a tract
  broadcast).
- **predictor functional form / log-scaled predictor** — the modeled form of a raw
  predictor, owned by the `*_MODEL_TRANSFORMS` specs in `regression_modelling.constants`
  and applied by `feature_engineering.transforms.apply_transforms` (`{col}_log`). Rule:
  **right-skewed magnitude counts/distances are log1p-compressed before the `log1p`-target
  fit; only bounded shares/ratios stay raw.** `pop_est_5mile` (5-mile population ring,
  ~46→2.3M) is log-scaled to **`pop_est_5mile_log`** — raw scale destabilised the fit
  (`expm1` blow-up; agency adj R² 0.22→0.36), ADR 0006. Raw names persist for EDA/imputation
  (`DEMOGRAPHIC_PREDICTORS`, `MEDIAN_FILL`); the model-form list feeds `PREDICTOR_COLS`.
- **filtered geoid set** — the post-drop geoids the fit actually runs on. Spatial weights
  (Moran's I), CV folds, and the bias-testing join must all align to it, not the full
  boundary set (ADR 0003).
- **`total_pt_ct`** — the **existing (agency-scale) model's** per-BG predicted weighted
  *current* total crime rate (`ct` = "current", **not** census tract). Present per-BG in
  `data/interim/bg_crime/{city}.parquet` (all categories carry a `*_pt_ct`); the national
  source is `gs://…/ns4/2025q4/block_group_data.sav` keyed by `bg_key`. This is the
  agency-scale prediction we **benchmark against** in the scale-case (`04_bg_comparison`).
  It varies within-city (it is agency-*trained* but BG-*scored*, not a flat agency broadcast).
- **between-city vs within-city skill (scale case)** — the diagnostic motivating the
  city-incident BG target: the agency-scale model tracks crime **level between cities**
  (correct city-mean ordering) but fits **within-city magnitude** weakly (near-zero
  city-demeaned Pearson, only moderate within-city rank). Measured by decomposing
  `total_pt_ct` ↔ observed weighted relative-risk rate into a between-city (city-mean) vs
  within-city (city-demeaned) component. Part 1 of the two-part rebuild case (predictors =
  `03_agency_comparison`; target/scale = `04_bg_comparison`). Pending ADR 0007.

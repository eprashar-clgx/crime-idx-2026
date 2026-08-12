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
  rate normalization (incl. daytime-adjusted).
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
- **`models`** — model fitting (OLS, coefficient tables) and model diagnostics,
  including **Moran's I** (spatial autocorrelation of residuals).
- **`bias_testing`** — checks that predictors correlate with crime and **not** with
  protected attributes (e.g. race). Distinct from Moran's I.
- **`logging`** — run/experiment logging.

### `carrier_eval` (side-quest)
Goal: evaluate how well the **existing** model performs against carrier insurance
data. Dedicated, clearly-separated space (side-quests recur and must not leak into
the main POC). Holds carrier-evals ingestion/aggregation and the **score
reconstruction math** — `compute_weighted_scores`, `extract_national_rates` — plus
its own maps.

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
  national `*_pt_u` rate), not raw sums. See `docs/weightage_methodology.md`.
- **prediction target** — `log(count + 1)` (`*_logcount`); `*_rate` kept as
  validators. Predictors z-standardized at fit; report HC3 SEs; check Moran's I.

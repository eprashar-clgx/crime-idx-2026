# EDA & Feature Feasibility Plan

Living plan for exploring candidate predictors, testing SQL against BQ, checking
distributions, assessing data-gathering feasibility, and iterating engineered
features in city crime regressions. Update this file as work progresses.

> Fits the existing pipeline (`docs/architecture.md`): registry -> cache ->
> model table -> regression, with EDA->features and diagnostics->spec loops.

## Phases

Phases run in dependency order; phases 1-3 loop per variable.

| # | Phase | Depends on | Summary |
|---|-------|-----------|---------|
| 0 | Variable inventory & triage | — | List every candidate variable in the table below; tag `bq_ready` vs `feasibility_unknown` + a one-line crime hypothesis. |
| 1 | SQL prototyping + distribution EDA (`bq_ready`) | 0 | `regression_modelling/data_wrangling/sql/explore/{var}.sql` run via `sources.load_sql(name, kind="explore")` + `run_bq_explore()`; BG-level counts, null rate, cardinality, ranges; plot distributions in a `notebooks/regression_modelling/distributions/` notebook via `regression_modelling.distributions.plots`. Decide keep/transform/drop. |
| 2 | Feasibility assessment (`feasibility_unknown`) | 0 | Locate source (BQ/GCS/external), check BG coverage, `geoid` joinability, refresh cadence; write go/no-go. No source code until it passes. |
| 3 | Feature engineering + regression iteration (validated vars) | 1, 2 | Add `FeatureSource` to `FEATURE_SOURCES` (`regression_modelling/constants.py`) + `data_wrangling/sql/build` + `data_wrangling/sql/pull`; engineer features in `feature_engineering`; add to `PREDICTOR_COLS` (+ `ZERO_FILL`/`MEDIAN_FILL`); rebuild model table; fit in `models/model.py` (coef sig, VIF, residuals, Moran's I); iterate. |

## Variable tracker

`bucket`: `bq_ready` | `feasibility_unknown`.
`stage`: inventory -> sql_eda -> feasibility -> engineered -> in_regression -> kept | dropped.

| Variable | Bucket | Hypothesis (relationship to crime) | Stage | Decision | Notes |
|----------|--------|------------------------------------|-------|----------|-------|
| convenience_stores | bq_ready | Presence invites/attracts criminal activity | sql_eda | — | Broad dual-vintage NAICS (`445131`/`445120` convenience + `457110`/`447110` gas-w-mart) via generic `stores.sql`; replaces the old 7-Eleven-only `seven_eleven` proxy. |
| liquor_stores | bq_ready | Presence invites/attracts criminal activity | sql_eda | — | Existing `liquor_stores` source (`business_name/brand LIKE %liquor%`). |
| gas_stations | bq_ready | Presence invites/attracts criminal activity | sql_eda | — | Existing `gas_stations` source (`business_name LIKE %gas station%`). |
| transit_stations | feasibility_unknown | Presence invites/attracts criminal activity (co-location, overnight exposure, convergence-node effects) | engineered | — | **Resolved to external GTFS** (Mobility Database), not BQ firmographics. Built + cached for 8 cities via `regression_modelling/data_wrangling/transit/`; 10 BG features in the `transit` `FeatureSource` (`backend="file"`). See `docs/features/transit_eda_plan.md`. Co-location H1/H3 columns BQ-gated (need POI points). Next: distribution EDA, then promote to `PREDICTOR_COLS`. |
| structure_density | feasibility_unknown | Denser -> less crime; low density may = dilapidated blocks -> more crime | feasibility | — | Explore interaction of vacant parcels x structure density x roof/age/condition (IDAP property data). |

## Open feature-engineering questions

- **Store spatial form:** express each store variable as in-BG count vs count within a radius
  (e.g. 5 miles) of block centroid vs distance to nearest store. Compare in EDA before promoting.
- **Structure density:** whether a single density measure or an interaction term
  (vacant parcels x density x roof/age/condition) carries the crime signal.

## Working notes / decisions

- 2026-08-04: Triaged 5 variables. Stores are `bq_ready` (user has EDA SQL); transit &
  structure density need feasibility work. Next: templatize the store EDA SQL + build a
  BQ runner + folium visualization.
- 2026-08-04: Built reusable store EDA repo — `regression_modelling/data_wrangling/sql/explore/*.sql`
  (NAICS mix, SIC mix, national counts, per-state counts, parcel points),
  `regression_modelling/distributions/eda.py` (`StoreQuery` registry + runners via `run_bq_explore`),
  `regression_modelling.distributions.plots.store_points_map` (folium), and
  `notebooks/regression_modelling/distributions/store_eda.ipynb`.
  Fixed an AND/OR precedence bug in the original filter. Next: run in BQ, confirm
  NAICS codes per store, then decide the BG spatial feature form (radius vs nearest).
- 2026-08-25: Transit feasibility **resolved** — chose external GTFS (Mobility Database) over
  BQ firmographics/OSM. Built the `transit/` ingestion module (gtfs-kit) and cached 10 BG
  features for 8 cities to `data/interim/sources/transit.parquet`; wired the file-backed
  `transit` `FeatureSource`. Risky-facility co-location (H1/H3) is implemented but emits zeros
  offline (POI points are BQ-gated). Next: distribution EDA, then promote validated features to
  `PREDICTOR_COLS`. Full plan + status: `docs/features/transit_eda_plan.md`.

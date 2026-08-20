# ADR 0002 — GTFS transit predictors: ingestion architecture

- **Status:** Accepted
- **Date:** 2026-08-20
- **Related:** `docs/transit_eda_plan.md`, `docs/hypothesis.md`, ADR 0001

## Context

We are adding **transit** predictors (GTFS-derived) to the block-group (BG) crime model
for the five POC cities (Chicago, Houston, Atlanta, San Francisco, Pittsburgh). The
hypotheses (see `docs/hypothesis.md`) are: H1 stops co-located with risky facilities
(convenience/liquor/ATM), H2 overnight/24-7 service, and H3 their interaction.

Every existing `FeatureSource` is **BigQuery/GCS-backed and already aggregated to
`geoid`** — BigQuery does the point→BG mapping and `pull_source` just caches a
geoid-keyed parquet. Transit breaks three of those assumptions:

1. **Local zip files**, not BQ/GCS. GTFS feeds are downloaded `.zip` archives.
2. **Point data needing a spatial join** to BGs (like crime incidents, unlike the
   pre-aggregated store sources).
3. **A per-stop derivation step** (service span, overnight flag, facility co-location,
   the H3 flag) that must happen *before* aggregation to BG — a grain that has no
   equivalent anywhere in the current pipeline.

It is also **per-city and point-based**, whereas existing sources are national and
pre-aggregated. And H1/H3 require **point-to-point** distance from each stop to the
nearest facility, so transit consumes a POI **point** layer (lat/lon), not the existing
BG-level store counts.

## Decision

Keep the registry as the single plug-in seam; add one new backend and one task-specific
submodule. No new top-level module (transit is a `regression_modelling` predictor, not
shared with `carrier_eval`).

- **New backend `backend="file"`** on `FeatureSource`. `pull_source` reads a pre-built
  parquet and does not re-fetch — building is out-of-band, exactly mirroring the existing
  BQ **build/pull split** (`run_bq_build_store` materializes; `pull_source` reads).
- **New submodule `regression_modelling/data_wrangling/transit/`:**
  - `feeds.py` — read city zip(s) via `gtfs-kit`, pick the representative 2025 service
    date, compute per-stop features (span, overnight, trips/day, route types). Unions
    multi-feed cities (SF = Muni + BART) and dedups physically shared stations.
  - `colocation.py` — per-stop nearest-facility distance / within-radius (H1) and the
    stop-level H3 AND flag; plus BG-aggregation helpers and the Shannon equitability
    index for route-mode diversity.
  - `build.py` — `build_transit(city)`: stops → sjoin to `geoid` (reusing
    `crime_blockgroup_mapping.boundaries` + the crime sjoin pattern) → aggregate to BG.
    Stacks all cities → `data/interim/sources/transit.parquet`.
- **`TRANSIT_FEEDS` registry** in `regression_modelling/constants.py` (city → list of
  feed specs) — config over hardcoding, same spirit as `CITIES`. A `transit`
  `FeatureSource` (`backend="file"`) makes it appear in `FEATURE_SOURCES` like any other.
- **Data tiering** (matches ADR 0001 / `data/README.md`):
  - `data/raw/transit/{city}/*.zip` — immutable downloaded feeds.
  - `data/interim/transit/stops/{city}.parquet` — per-stop intermediate (reusable).
  - `data/interim/sources/transit.parquet` — geoid-keyed BG feature table (registry cache).
- **New dependency:** `gtfs-kit` (Poetry).

### Decisions locked
- **One representative mid-2025 snapshot per city** (feed active ~June 2025), aligned
  across cities; record each `feed_version`. GTFS has no whole-year feed; supply is stable
  quarter-to-quarter. Multi-feed (median-across-feeds) only if we later model within-year
  change.
- **SF = Muni + BART** (two feeds unioned; shared stations deduped by proximity).
- **POI co-location uses point coordinates** (true stop→facility distance), not the
  existing BG-level store counts.
- **Supply, not ridership.** GTFS Static only; realtime/APC/AFC are out of scope and the
  crowding channel behind H3 is a documented approximation.

## Consequences

**Positive**
- One documented new code path (the `"file"` backend); everything downstream
  (`assemble_features`, model table) is unchanged.
- Reuses the foundation's boundary + sjoin machinery — no duplication.
- Transit stays out of `carrier_eval`; the seam (both tasks depend only on
  `crime_blockgroup_mapping`) is preserved.
- The per-stop intermediate is cached and reusable for future transit features.

**Negative / trade-offs**
- First source with a stop-level intermediate grain — more moving parts than a BQ pull.
- `backend="file"` sources are materialized out-of-band, so a missing artifact is a
  hard error directing the user to run `build_transit` (no silent auto-fetch).
- Transit covers only the five POC cities, so its columns are null for the rest of the
  national BG spine (handled by the existing left-join + Phase-3 imputation).

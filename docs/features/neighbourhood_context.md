# Neighbourhood context features

Built by `data_wrangling/neighbourhood.py::build_neighbourhood()` into
`data/interim/sources/neighbourhood.parquet` (`neighbourhood` FeatureSource, backend `file`).
National: every BG in the 2025 cartographic-boundary layer (`cb_2025_us_bg_500k`).

## 1. Neighbour definition

Queen contiguity: two BGs are neighbours if their polygons share an edge or a vertex. Self
is excluded. Using the national layer means neighbours across state lines count. Mean
6.3 neighbours; 63 BGs have none (islands → NaN → city median).

Queen contiguity is used here instead of the KNN(6) pooled lag that the property-distress
`*_lag6` columns use (`property_distress.md` §2). The ACS inputs are published as shares
without numerators and denominators, so they can't be pooled.

## 2. Features

| Column | Definition | Model form |
|---|---|---|
| `{own,det,lap,moved1yr}_pct_nbr` | unweighted mean of neighbours' raw shares (NaN neighbours skipped) | raw |
| `unq_{convenience_stores,gas_stations,liquor_stores}_clips_nbr` | sum of neighbours' store counts | `log1p` |
| `unq_*_clips_per_km2` | BG's own store count / land area (ALAND), area floored at 0.05 km² | `log1p` |
| `bg_area_km2` | ALAND / 1e6 (size/exposure column, not a default candidate) | `log1p` |

Not re-averaged: vacancy/liens/foreclosures/transactions (already have `*_lag6`), and the
population rings (`pop_est_5mile`, `pop_ch_1mile`). Transit, roadway and imagery cover
in-city BGs only, so their neighbour means would be biased at the city edge.

## 3. Results (2026-10-01)

LOCO over the final-10 + anchor (Ridge base r2 0.390; deterministic GBM base r2 0.410):

| Added | Ridge Δr2 (cities up) | GBM Δr2 (cities up) | GBM recall |
|---|---|---|---|
| neighbour means + store sums (7) | +0.001 (15) | +0.014 (14) | 0.399 → 0.410 |
| store densities (3) | +0.001 (13) | +0.007 (15) | 0.399 → 0.400 |

Per-resident store ratios were tested and dropped: they mostly proxy population.
City-, county- and CBSA-relative ranks were also tested and dropped: they hurt in every
geography.

## 4. Caveats

- Neighbour means are unweighted. Population-weighting is a possible refinement.
- Inputs are raw (pre-imputation) values, so a NaN neighbour is skipped rather than
  median-filled.
- Rebuild after the demographic or store sources are refreshed: the builder reads their
  cached parquets.

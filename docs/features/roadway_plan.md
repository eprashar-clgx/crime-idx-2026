# Roadway / Road-Network — Feature & Data Design Plan

Design for adding **roadway / road-network** predictors to the block-group (BG) crime model,
as a complementary "crime opportunity / feeding ground" signal to the transit features —
especially for **suburban/rural BGs where transit is sparse or absent**. Companion to
`docs/features/transit_eda_plan.md`, `docs/features/transit_stats.md` (the functional-form /
hurdle machinery this reuses), and `docs/hypothesis.md` (theory mapping).

> **Design constraint (same as transit):** no demographic predictors. Roadway features are
> justified by *environmental / situational* theory only — Crime Pattern Theory (nodes /
> paths / **edges**), Routine Activity Theory (target/offender flow, guardianship), and Risk
> Terrain Modeling. They sit *net of* the existing population / land-use controls
> (`pop_est_5mile_log`, density) because the literature repeatedly warns road-network effects
> confound with population and land use.

> **Status: PLANNING — no code yet.** This doc fixes the target so a build can start from a
> settled feature list, data source, and licensing decision.

---

## 1. Motivation & hypothesis

Transit stops and risky POIs act as **crime generators / activity nodes** (Brantingham &
Brantingham pattern theory): places where offenders and targets converge and impulsive /
property crime concentrates, with a route in and a quick route out. In low-density areas that
lack transit, the **road network is the analog**: **highway on/off ramps and interchanges**
create the same accessibility + convergence + fast-escape structure ("edges" and "paths"),
and **major arterials** carry the through-movement that concentrates outdoor crime.

Core hypotheses (all environmental):

- **R1 — Highway-access / edge (primary).** BGs closer to a highway ramp / interstate have
  elevated crime (accessibility + escape-route). This is the best-supported analog and the
  natural **extensive-margin partner to `transit_has_transit`** for no-transit BGs.
- **R2 — Arterial through-movement.** Higher arterial-road density → more crime (main roads
  carry the movement that generates outdoor offences).
- **R3 — Permeability / connectivity.** Higher street permeability / intersection density
  relates to crime — **direction is contested** (see §3), so this enters as an exploratory
  control, not a headline feature.

---

## 2. Structural advantage over transit

Transit features were **city-feed-limited** (one GTFS feed per city; BGs outside a feed's
service area are structurally null). Roadway features come from **national** road data
(Census TIGER/Line), so they **populate every BG on the national spine — including the
rural/suburban BGs where transit = 0.** That is exactly the gap this work targets:
`has_highway_access` is defined everywhere, whereas `has_transit` is not.

---

## 3. Evidence base (2015+, US-first; verified citations)

What the empirical literature supports — and, importantly, what it does *not*.

| Concept | Key study (verified DOI) | Unit / place | Direction | Confidence |
|---|---|---|---|---|
| Proximity/adjacency to **interstate as "edge"** | ⭐ Kim & Hipp 2017, *Crime & Delinquency*, 10.1177/0011128716687756 | street segments, S. California | closer → more crime | **Strong (direct US)** |
| **Street-network morphology** (community/ring roads +, **intersections −**) | Mao et al. 2025, *Humanities & Social Sci. Comms.*, 10.1057/s41599-025-05362-1 | neighborhood, Detroit | mixed (see below) | Medium (US) |
| **Main arteries / high-choice** carry violence | Summers & Johnson 2016, *J. Quant. Criminol.*, 10.1007/s10940-016-9306-9 | segments, **London (non-US)** | high-choice → more | Medium (non-US) |
| **Permeability** → burglary; cul-de-sacs safer | Johnson & Bowers 2010, *J. Quant. Criminol.*, 10.1007/s10940-009-9084-8 | segments, UK (foundational) | more permeable → more | Foundational |
| Crime concentrates at **micro-places** | Weisburd 2015, *Criminology*, 10.1111/1745-9125.12070 | street segments (foundational) | — | Foundational |
| Pattern theory: nodes / paths / **edges** | Brantingham & Brantingham 1993/1995 (foundational) | theory | — | Foundational |

**Contested sign — intersection density / permeability.** Mao 2025 (Detroit) finds
**intersections protective (−)**, while the permeability tradition (Johnson & Bowers 2010;
Summers & Johnson 2016) finds connectivity **criminogenic (+)**. Likely because "number of
intersections" (a density/urbanity proxy) and "through-movement permeability" (space-syntax
choice/betweenness) capture different things. → We treat R3 as **exploratory**, check its sign
empirically per-city, and do not lean on it.

**Two genuine gaps (important for scoping):**

1. **AADT / traffic volume as crime exposure:** **no** verified US peer-reviewed study
   operationalizes it. Novel but unproven → **deferred** (see §6). Not in the first build.
2. **Ramp / interchange / exit density as a discrete predictor:** no study isolates it;
   Kim & Hipp 2017 interstate **adjacency/distance** is the closest verified proxy. We include
   ramp proximity under R1 but label the "getaway via ramp" mechanism as theory-motivated,
   not directly estimated.

**Confounding caveat (all studies).** Road-network features co-vary strongly with population,
land use, and ambient foot traffic; Hipp et al. 2022 (10.1177/17488958221132764) shows
neighborhood disadvantage **moderates** built-environment → crime effects. Keep the existing
population/density controls; consider a road × context interaction later.

---

## 4. Proposed feature family (mirrors the transit hurdle structure)

One BG-level `roadway_*` family, engineered to slot into the exact machinery built for
transit (`feature_engineering.transforms.apply_transforms`, the hurdle form, and the
`correlation_matrix` variance guard). Candidate raw features and their intended model form:

| # | Raw feature | Definition | Source | Hypothesis | Model form |
|---|---|---|---|---|---|
| 1 | `roadway_nearest_ramp_m` | meters, BG centroid → nearest ramp | TIGER ROADS `S1630` | R1 | `has_highway_access` (extensive) + `nearest_ramp_m_logc` (log1p, centered on served — hurdle) |
| 2 | `roadway_nearest_interstate_m` | meters → nearest interstate/limited-access | TIGER ROADS `S1100` | R1 (edge) | `nearest_interstate_m_log` |
| 3 | `roadway_ramp_count` | # ramp features intersecting the BG | TIGER ROADS `S1630` | R1 | `log1p` |
| 4 | `roadway_arterial_density` | km of principal + minor arterial per km², clipped to BG (interstates/freeways excluded — already in #2) | FHWA HPMS `f_system` 3–4 | R2 | `log1p` |
| 5 | `roadway_intersection_density` | street junctions (≥ 3 incident `S1200`/`S1400` edges) per km² | TIGER EDGES topology | R3 (contested) | `log1p` — exploratory, verify sign |

**Why HPMS for #4, not TIGER `S1200`.** TIGER's `S1200` ("secondary road") only covers
*numbered* US/state/county highways; most big-city arterials (e.g. Chicago's grid main
streets) are coded `S1400` alongside every residential street, with no hierarchy inside it.
An `S1200`-only arterial density was 76% zeros and near-null in Chicago/SF/KC. Adding
`S1400` doesn't help — it's 74–92% of road length, measures grid density (ρ ≈ 0.6 with #5),
and still can't tell a main street from a cul-de-sac. HPMS `f_system` is FHWA's official
functional classification (1 Interstate, 2 Other freeway/expressway, 3 Other principal
arterial, 4 Minor arterial, 5–6 collectors, 7 local), assigned by state DOTs.

**Why TIGER EDGES for #5.** ROADS has one record per named road, crossing others at interior
vertices, so endpoint degree is meaningless. EDGES is the topological layer (every edge split
at every node, with from/to node ids `TNIDF`/`TNIDT`), so degree is a direct count. Only
at-grade public streets (`S1200`, `S1400`) count toward degree — TIGER topology is planar,
so including `S1100`/`S1630` would register overpasses as false junctions.

- **`has_highway_access`** = `1[nearest_ramp_m ≤ threshold]` (or ramp_count > 0) — the
  extensive margin, defined for **every** BG (unlike transit). Threshold to be set in EDA.
- Hurdle centering on feature 1 (`nearest_ramp_m_logc` centered on the served mass) reuses the
  exact trick that made `transit_has_transit` ⟂ `service_intensity` — see `transit_stats.md`.
- **Deferred (not in first build):** `roadway_aadt_near` (traffic exposure) — no evidence base
  (§3 gap 1). The HPMS pull already carries `aadt` on every section, so this needs no new
  source when revisited.

Expected collinearity (to resolve with `correlation_matrix` + hurdle, as with transit): the
distance features (ramp vs interstate) and the two density features will correlate; ramp
proximity will also correlate with `has_highway_access`. Prune to a non-redundant retained set
the same way — likely keep one distance (edge), one density (arterial), plus the indicator.

---

## 5. Data sources & licensing decision

**Chosen stack: TIGER/Line + FHWA HPMS — public domain only. No OSM, no Overture.**

| Source | Role | License | Commercial? |
|---|---|---|---|
| **US Census TIGER/Line ROADS** | ramps (`S1630`), interstate (`S1100`); nearest-distance, ramp count | Public domain | ✅ clean |
| **US Census TIGER/Line EDGES** | street-junction topology (intersection density) | Public domain | ✅ clean |
| **FHWA HPMS (2024)** | functional class (arterial density); carries AADT for the deferred feature | US federal publication; no licence restrictions stated | ✅ clean (confirm with legal) |
| ~~Overture Maps — transportation~~ | (dropped) was the intersection-density source | **ODbL 1.0** | ⚠️ share-alike on redistribution |
| ~~OpenStreetMap via `osmnx`~~ | (rejected) richer `basic_stats` permeability | **ODbL 1.0** | ⚠️ share-alike on redistribution |

**Why no OSM-lineage data (the ODbL caveat).** ODbL 1.0 requires (1) **attribution** and
(2) **share-alike** — publicly distributing a *derivative database* built from it (or a
produced work made from one) can force you to release that derivative database under ODbL.
Internal feature computation is generally fine, but once derived roadway columns ship inside a
**distributed commercial data product**, the share-alike clause can attach, and the
"insubstantial extract vs. derivative database" line is legally fuzzy.

**Correction (Overture).** An earlier draft of this doc chose Overture for permeability on the
belief that it is relicensed CDLA-Permissive 2.0. That is wrong for the **transportation**
theme: every segment's `sources[].license` field reads `ODbL-1.0` (OpenStreetMap, plus a
small TomTom contribution also under ODbL — checked on the 2026-09-23.0 release). CDLA applies
to other Overture themes, not roads. Overture was therefore dropped and intersection density
rebuilt from TIGER EDGES; the cached Overture parquets were deleted.

**Technical notes for the build:**

- TIGER ROADS / EDGES: per-county zips `tl_<YYYY>_<county FIPS>_{roads,edges}.zip` under
  `https://www2.census.gov/geo/tiger/TIGER<YYYY>/{ROADS,EDGES}/`. Vintage 2024.
- MTFCC road codes: `S1100` primary/interstate, `S1200` secondary (numbered highways only),
  `S1400` local/city street, **`S1630` ramp**, `S1640` service/frontage. **No explicit
  "interchange" feature** — an interchange is a *cluster of `S1630` features* off an `S1100`;
  derive interchange count by clustering `S1630` centroids if needed.
- HPMS: one public ArcGIS FeatureServer per state,
  `https://geo.dot.gov/server/rest/services/Hosted/HPMS_FULL_<ST>_2024/FeatureServer/0`
  (all 16 model-city states present). Queried per county, filtered server-side to
  `f_system IN (1,2,3,4) AND facility_type IN (1,2)` and paged (2,000-record limit).
  `facility_type = 6` ("non-inventory direction") is the mirrored opposite carriageway of a
  divided road — excluded to avoid double-counting length. `county_id` is the integer county
  FIPS. The line work's geodetic accuracy is not evaluated by FHWA (fine at BG scale).
  Ramps (`facility_type = 4`) are also pulled, **spatially** (county mainline bbox + ~2 km),
  because IL, TX, OH, MO and WA leave `county_id` null on ramps. Fields cached per section:
  `aadt`, `aadt_single_unit`, `aadt_combination`, `through_lanes`, `speed_limit`,
  `access_control`, `nhs`.
- **TIGER vs HPMS for R1 (10-city check, 2026-09).** TIGER `S1100` ≈ HPMS `f_system` 1–2
  (freeways incl. non-Interstate; per-city ρ 0.89–1.00), so `nearest_interstate_m` is really
  "nearest limited-access freeway". HPMS ramp inventory is **incomplete** in some states
  (BGs with a ramp: Houston 11% vs TIGER 32%, SF 2% vs 12%, Pittsburgh 16% vs 25%), and crime
  ρ is equal or slightly weaker than TIGER → R1 stays on TIGER. HPMS traffic metrics:
  arterial lane-km density ≈ arterial density (ρ 0.96, same crime ρ); arterial VMT density
  (ρ 0.90) is weaker (median 0.27 vs 0.31); freeway VMT / truck VMT / max AADT track ramp
  count (ρ 0.55–0.73) and are weaker (0.15–0.20). None promoted.
- **CRS:** TIGER native is EPSG:4269; HPMS is requested in EPSG:4326. **Reproject to
  EPSG:5070 (CONUS Albers, meters)** before any distance / length / density math — same
  equal-area CRS used for transit density (`_EQUAL_AREA_CRS`).
- BG polygons: reuse the shared `load_state_block_groups` foundation (same as transit/crime).

---

## 6. Repo integration (same discipline as transit)

- **New build module** `src/regression_modelling/data_wrangling/roadway/` mirroring
  `transit/`: load road layers → reproject to 5070 → sjoin / nearest to BG polygons →
  aggregate to `geoid`. `build_all_roadway()` materializes
  `data/interim/sources/roadway.parquet`, exposed by a new
  `FeatureSource(name="roadway", backend="file")`. Out-of-band build (nothing in the normal
  pull path triggers it), like transit.
- **Constants** (`regression_modelling/constants.py`): add `ROADWAY_PREDICTORS` (raw),
  `ROADWAY_MODEL_TRANSFORMS`, and `ROADWAY_MODEL_PREDICTORS` (retained set); extend
  `PREDICTOR_COLS` (stays derived) and the `ZERO_FILL` / `MEDIAN_FILL` lists. Distances →
  `MEDIAN_FILL`; counts / densities / indicator → `ZERO_FILL`.
- **Reuse, don't rebuild:** `apply_transforms(hurdle=True)` (add `roadway`'s hurdle pair),
  `correlation_matrix` (variance guard already handles any degenerate/all-zero column), and
  the distribution-EDA workflow (raw vs log, contrasting cities).
- **Print diagnostics** on build (BGs with/without ramp access, nearest-distance summary) per
  the repo's ingestion-logging convention.

---

## 7. Execution status

**Planned (this doc):**

- [x] Scaffold `data_wrangling/roadway/` (`tiger.py`: ROADS `S1630`/`S1100` + EDGES junctions;
      `hpms.py`: FHWA functional class; `build.py`: BG aggregation) + `FeatureSource` +
      constants group.
- [x] Build the 5 candidate raw features for the original 10 model cities; cache to
      `data/interim/sources/roadway.parquet`.
- [ ] Extend the build to the 10 cities added in the 20-city expansion.
- [ ] Distribution EDA (raw vs log; a good vs bad city) → confirm transforms + the
      `has_highway_access` threshold.
- [ ] Correlation / redundancy pruning (`correlation_matrix`) → retained
      `ROADWAY_MODEL_PREDICTORS`; hurdle-decorrelate the ramp indicator vs distance.
- [ ] **Verify the intersection-density sign** empirically (R3 is contested) before trusting
      it; drop or keep based on stability.
- [ ] Wire retained roadway features into `build_model_table` / `PREDICTOR_COLS`.

**Deferred / backlog:**

- [ ] **AADT / traffic-exposure feature** (`roadway_aadt_near`) — no evidence base; revisit
      after the current features validate. Data is already in the HPMS pull (`aadt`).
- [ ] Interchange-cluster count (DBSCAN over `S1630` centroids) if raw ramp density proves
      noisy.
- [ ] Road × neighborhood-context interaction (Hipp 2022 moderation finding).

---

## 8. Key references

- Kim & Hipp (2017), *Crime & Delinquency* — interstate highways as crime edges. `10.1177/0011128716687756`
- Mao et al. (2025), *Humanities & Social Sciences Communications* — Detroit street-network morphology. `10.1057/s41599-025-05362-1`
- Summers & Johnson (2016), *J. Quantitative Criminology* — space syntax, main arteries (London). `10.1007/s10940-016-9306-9`
- Johnson & Bowers (2010), *J. Quantitative Criminology* — permeability & burglary. `10.1007/s10940-009-9084-8`
- Weisburd (2015), *Criminology* — law of crime concentration. `10.1111/1745-9125.12070`
- Hipp et al. (2022), *Criminology & Criminal Justice* — disadvantage moderation. `10.1177/17488958221132764`
- Brantingham & Brantingham (1993/1995) — crime pattern theory (nodes / paths / edges), foundational.

Data: US Census TIGER/Line ROADS (MTFCC `S1630`/`S1100`) and EDGES (junction topology), public
domain; FHWA HPMS 2024 (functional class `f_system`, AADT), US federal publication.

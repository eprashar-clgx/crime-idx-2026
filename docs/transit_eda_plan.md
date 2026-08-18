# Transit (GTFS) — EDA & Feature Plan

Plan for adding **transit** predictors to the block-group (BG) crime model. Scopes the
hypotheses, their theory roots, and exactly what to pull from GTFS and how — so execution
can start from a fixed target. Companion to `docs/hypothesis.md` (full theory mapping) and
`docs/eda_plan.md` (pipeline phases). **Planning only — no code until this is signed off.**

> **Design constraint:** no demographic predictors (race, sex, age, income-as-group).
> Transit features are justified by *environmental/situational* theory only — Routine Activity,
> Crime-as-Opportunity, Crime Pattern Theory, and Risk Terrain Modeling (RTM).

> **Modeling context:** target `*_logcount` = log(count+1); `*_rate` (per 1,000 pop + LODES
> daytime variant) as validators. Inferential standardized OLS, HC3 SEs, Moran's I check.

---

## 1. Key hypotheses & research roots

| # | Hypothesis | Theory root | Expected offense concentration |
|---|-----------|-------------|-------------------------------|
| **H1** | Transit stops co-located with a convenience store (7-Eleven), liquor/alcohol outlet, or ATM are more dangerous; BGs with more such co-located stops have higher property-crime `logcount`, net of stop density. | **RTM "risky facilities"** (Eck/Clarke/Guerette); **Crime Pattern** attractors (Brantingham & Brantingham 1995); **RAT** suitable-target = cash/alcohol disinhibition. | robbery, larceny, MVT |
| **H2** | Stops with overnight / near-24-7 service carry a low-guardianship exposure window and are more dangerous per unit of activity; BGs with more all-night stops have higher property + robbery `logcount`. | **RAT** (Cohen & Felson 1979) — guardianship collapses at night, extended exposure window; **Crime-as-Opportunity** (Mayhew et al. 1976). | robbery, larceny, MVT |
| **H3** | **Interaction of H1 × H2:** a stop that is *both* facility-adjacent *and* all-night is disproportionately criminogenic — beyond either main effect alone. | RAT convergence (target + offender + absent guardian align in space **and** time); reinforced by Díaz et al. 2026 (crowding→theft, offense-specific). | robbery, larceny, MVT; interaction strongest |

**Supporting evidence (2015–2026):** Phillips & Sandler 2015 (station closures → crime falls,
causal); *Bus Stops and Violence* 2018 (RTM; facilities predict counts not per-person risk — the
ambient-population confound); Chicago station-areas 2022 (BG-level U-shape, guardianship);
Kadar & Pletikosa 2018 (ambient features beat census baseline; larceny gains most); Díaz et al.
2026 (micro-temporal causal DiD — **crowding**, not mean flow, selectively raises **theft**).

**Guardrails carried from `hypothesis.md`:** (1) count-vs-per-person confound → also test `*_rate`
+ population/jobs offset; (2) non-monotonic at BG scale → allow non-linear form; (3) city
heterogeneity → interact with city / transit-share (Chicago is high-transit; Atlanta, Sacramento,
Pittsburgh auto-oriented); (4) prefer GTFS **supply** over endogenous ridership; (5) opportunity ≠
volume → crowding proxy over mean flow; (6) offense specificity → property/theft, not violent.

---

## 2. What to source from GTFS — files, columns, why

Only what H1–H3 require: a **stop location** + a **per-stop service profile**. Five files.

| File | Columns used | Why (which hypothesis) |
|------|-------------|------------------------|
| **stops.txt** | `stop_id`, `stop_lat`, `stop_lon`, `location_type`, `parent_station` | Spatial anchor: `sjoin` to `geoid`; co-locate with 7-11/liquor/ATM POIs (**H1**). |
| **stop_times.txt** | `trip_id`, `stop_id`, `departure_time` | Per-stop **service span** (first→last), **overnight** trips, **trips/day** (**H2**). |
| **trips.txt** | `trip_id`, `route_id`, `service_id` | Bridges `stop_times` → `calendar` (which days) and → `routes` (mode). |
| **calendar.txt** / **calendar_dates.txt** | `service_id`, `monday`…`sunday`, `start_date`, `end_date`; exception `date`/`exception_type` | Pick a representative weekday; **days/week** of service; apply exceptions (**H2**). |
| **routes.txt** | `route_id`, `route_type` | Rail vs bus weighting (`route_type` 0/1/2 = rail); mode diversity. |

Optional: **frequencies.txt** (`headway_secs`) for frequency-based feeds; **feed_info.txt**
(`feed_version`, `feed_start_date`, `feed_end_date`) to **pin each feed to the 2025 crime year**.

**Per-stop variables to derive:**
- `stop_lat`, `stop_lon` — geometry for `sjoin` and POI co-location.
- `span_hours` = last − first `departure_time` on a representative day.
- `overnight_flag` = any departure in ~00:00–05:00.
- `n_trips_day`, `days_per_week`, `route_types` (set of modes served).

**Four gotchas (these bite):**
1. **Times exceed `24:00:00`** (e.g. `25:30:00` = 1:30 AM next service day). Parse as seconds —
   this is exactly how overnight service is detected; do **not** drop them.
2. **`parent_station`** — collapse child platforms to the station so a rail station ≠ 6 stops.
3. **`location_type`** — keep boardable stops (`0`/blank); drop stations (`1`) and entrances (`2`).
4. **One representative service date** (a typical Wednesday via `calendar`), not the union of all
   `service_id`s — otherwise trips/day is overcounted.

---

## 3. Best way to source it

**Primary tool: `gtfs-kit`.** `feed.compute_stop_stats([date])` returns per-stop `num_trips`,
`start_time`, `end_time`, `min/mean/max_headway`, `num_routes` for a service date — the span +
frequency table in one call. Read `departure_time` from `stop_times` for the overnight flag.

```python
import gtfs_kit as gk
feed  = gk.read_feed("cta_gtfs.zip", dist_units="km")
week  = feed.get_first_week()                 # representative service week
stats = feed.compute_stop_stats([week[2]])    # a Wednesday: span + frequency per stop
stops = feed.get_stops(as_gdf=True)           # EPSG:4326 geometry -> sjoin to geoid
```

`start_time`/`end_time` are `HH:MM:SS` strings that **can exceed 24:00**, so `span_hours` and
`overnight_flag` fall straight out.

**Alternative: `partridge`** — `get_representative_feed()` auto-picks the busiest service date and
returns filtered pandas frames; faster/memory-light, slightly more manual for span. Default to
gtfs-kit for built-in stop stats; fall back to partridge on performance limits.

**Feed acquisition:** pull **stable archived** zips from the **Mobility Database** (API +
versioned history) so each city pins to a 2025 feed — not agency "latest" URLs that get overwritten.
Cities → agencies: Chicago = CTA (+Metra/Pace optional); Atlanta = MARTA; Pittsburgh = PRT;
Sacramento = SacRT; SF = SFMTA/Muni + BART (consider 511 SF Bay regional feed).

---

## 4. Interaction construction (H3) — the one thing not to get wrong

Build the interaction at **stop level first, then aggregate to BG**. Both conditions are properties
of the *same physical stop*; multiplying two BG-level averages would falsely light up a BG that has a
liquor-adjacent *daytime* stop and a separate all-night stop elsewhere.

- **Route A (recommended):** stop-level `risky_allnight(s) = near_risky(s) × overnight(s) ∈ {0,1}`,
  then BG `Σ` → `n_risky_allnight_stops`, `share_risky_allnight`, density/km².
- **Route B:** continuous `fac(s) × night(s)` (inverse-distance facility exposure × overnight-trip
  count), aggregate then z-score — keeps gradient info.

**Stats rules (inferential OLS, HC3):** (1) always include **both main effects** (marginality);
(2) **center components before multiplying** (kills collinearity, keeps mains interpretable) — the
pipeline z-standardizes at fit; (3) report VIF; (4) re-check Moran's I. **Support for H3** = the
interaction is significant while both mains are weak.

---

## 5. Feature shapes to test (per variable)

| Variable | Shapes | Enters as |
|----------|--------|-----------|
| Risky-facility co-location (H1) | stop-level flag / distance / kernel per category (ATM, liquor, convenience kept separate) → BG count, share, density; offering-advantage on categories | main |
| Overnight / 24-7 service (H2) | `span_hours`, `overnight_flag`, `days_per_week`, overnight-trip count → BG max/mean span, count/share overnight stops, Σ overnight trips | main + offset |
| Risky × overnight (H3) | stop-level AND → BG count/share (Route A); centered product (Route B) | interaction + both mains |
| (context) stop density | stops/km², kernel-weighted | main + quadratic |
| (context) route-mode diversity | Shannon equitability over `route_type` | main |

---

## 6. Execution checklist (next steps → maps to `eda_plan.md` phases)

1. **Feasibility (Phase 2):** confirm Mobility Database has 2025-vintage feeds for all five cities;
   verify stop `geoid` join coverage; confirm POI layer (7-11 / liquor / ATM) is already in hand.
2. **Prototype ingestion:** one city (Chicago/CTA) — gtfs-kit → per-stop `stats` + geometry;
   apply the four gotchas; `sjoin` to `geoid`; print matched/unmatched stop counts.
3. **Stop-level features:** derive `span_hours`, `overnight_flag`, `near_risky` (POI co-location),
   and the H3 AND flag at stop level.
4. **Aggregate to BG** and cache to `data/interim/sources/transit_<city>.parquet` (`refresh: bool`
   pattern); reproject to `EPSG:4326` for joins.
5. **Distribution EDA (Phase 1):** null rate, cardinality, ranges, distributions per feature;
   decide keep/transform/drop.
6. **Registry + regression (Phase 3):** add `FeatureSource` (local-file backend, not BQ/GCS) to
   `FEATURE_SOURCES`; add validated features to `PREDICTOR_COLS` (+ `ZERO_FILL`/`MEDIAN_FILL`);
   rebuild model table; fit with both-mains-plus-interaction; check sig, VIF, residuals, Moran's I;
   validate transit coefficients against `*_rate`; test city-heterogeneity interaction.

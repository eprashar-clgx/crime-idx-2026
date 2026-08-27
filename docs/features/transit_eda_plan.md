# Transit (GTFS) — EDA & Feature Plan

Plan for adding **transit** predictors to the block-group (BG) crime model. Scopes the
hypotheses, their theory roots, and exactly what to pull from GTFS and how — so execution
can start from a fixed target. Companion to `docs/hypothesis.md` (full theory mapping) and
`docs/eda_plan.md` (pipeline phases). The ingestion module is built and cached; see
**Where We Are Right Now** and **Execution status** below for current state.

> **Design constraint:** no demographic predictors (race, sex, age, income-as-group).
> Transit features are justified by *environmental/situational* theory only — Routine Activity,
> Crime-as-Opportunity, Crime Pattern Theory, and Risk Terrain Modeling (RTM).

> **Modeling context:** target `*_logcount` = log(count+1); `*_rate` (per 1,000 pop + LODES
> daytime variant) as validators. Inferential standardized OLS, HC3 SEs, Moran's I check.

---

## Where We Are Right Now: Feature Evaluation

Ten BG-level transit features are built and cached (`data/interim/sources/transit.parquet`),
grouped by the hypothesis each one tests. Theory roots are environmental only — Routine
Activity Theory (RAT), Risk Terrain Modeling (RTM), Crime Pattern Theory. No demographics.

**Supply / context (control + baseline exposure).** Not a single hypothesis — these quantify
how much transit a BG has, so H1–H3 effects can be measured *net of* sheer transit presence
(and to test the BG-scale U-shape).

| Feature | What it is | Why |
|---------|-----------|-----|
| `transit_stop_count` | # boardable stops in the BG | Base transit exposure; the density confound control. |
| `transit_stop_density` | stops per km² (equal-area) | Normalizes count by BG size — the real "how transit-saturated" measure. |
| `transit_nearest_stop_m` | distance from BG centroid to nearest stop | Proximity gradient; lets stopless BGs still carry a signal (distance-decay). |
| `transit_service_intensity` | sum of trips/day across the BG's stops | Activity/throughput proxy — more service = more ambient people converging (RAT). |
| `transit_route_mode_diversity` | Shannon equitability over `route_type` (bus/rail/streetcar) | Multi-modal interchange = convergence node (Crime Pattern "nodes"); SF scores highest. |

**H2 — Overnight / low-guardianship window (RAT).** Stops running overnight (~00:00–05:00)
have collapsed guardianship → higher property + robbery.

| Feature | What it is |
|---------|-----------|
| `transit_overnight_stop_count` | # stops with any overnight service |
| `transit_overnight_stop_share` | fraction of the BG's stops that run overnight |

**H1 — Risky-facility co-location (RTM "risky facilities").** Stops within ~150m of a
convenience store / liquor outlet / ATM are riskier (cash + alcohol disinhibition), net of
stop density.

| Feature | What it is |
|---------|-----------|
| `transit_risky_stop_count` | # stops near a risky facility |
| `transit_risky_stop_share` | fraction of the BG's stops that are facility-adjacent |

**H3 — Interaction (RAT convergence in space *and* time).** A stop that is *both*
facility-adjacent *and* all-night is disproportionately criminogenic — beyond either main
effect alone.

| Feature | What it is |
|---------|-----------|
| `transit_risky_allnight_count` | # stops that are *both* near a risky facility *and* overnight |

**Key design note:** H3 is built at the **stop level first** (a single stop flagged as
risky-AND-overnight), then counted per BG — not by multiplying two BG averages, which would
falsely fire when a BG has a daytime-risky stop and a *separate* all-night stop.

⚠️ The H1/H3 columns (`transit_risky_*`) are currently **zeros** — they need the BigQuery POI
point pull (see In-Progress below). Supply + H2 features are fully populated. Expected offense
concentration for all three hypotheses: **robbery, larceny, MVT** (property/theft, not violent).

---

## Execution status

Maps to `eda_plan.md` phases. Detail on gotchas and code locations follows in sections 1–5.

### Done

- **Feeds downloaded (Phase 2):** 2025 feeds for all ten cities from the Mobility Database,
  keyed by stable `feed_id` in `TRANSIT_FEEDS`. Stop→`geoid` join coverage confirmed (Chicago
  9,935/10,792; Houston 7,972/8,918; SF 3,155/3,212; etc. — remainder are suburban stops
  outside city limits).
- **Prototype ingestion:** implemented for all cities in `transit/feeds.py`
  (`read_stop_features` / `load_city_stops`) via gtfs-kit `compute_stop_stats` + per-stop
  geometry; the four gotchas applied; `sjoin` to `geoid` in `transit/build.py` with
  matched/unmatched diagnostics. A **service-density-aware date picker** selects a
  peak-service weekday nearest the target anchor (avoids near-empty special-service dates —
  MARTA's real service is confined to a narrow window).
- **Stop-level service features:** `span_hours`, `overnight_flag`, route-mode set done in
  `feeds.py`.
- **Aggregate to BG + cache:** `build_all_transit()` writes
  `data/interim/sources/transit.parquet` (registry cache; `refresh: bool` pattern), with
  per-stop intermediates at `data/interim/transit/stops/{city}.parquet`. All joins in
  `EPSG:4326`; area/centroid via equal-area `EPSG:5070`.
- **Registry wired:** the `transit` `FeatureSource` (`backend="file"`) is in `FEATURE_SOURCES`
  and read by `assemble_features`.

### In-Progress

- **Risky-facility co-location (H1/H3):** `near_risky` + `risky_allnight` logic implemented in
  `colocation.py` (`add_risky_flags`), but the POI point layer (7-11 / liquor / ATM) is pulled
  from the firmographics CLIP source in BigQuery (`build.load_risky_facilities`) — needs
  credentials, so H1/H3 columns are currently emitted as 0 offline. Run `build_all_transit`
  with BQ access to populate them.

### Planned

- **Distribution EDA (Phase 1):** null rate, cardinality, ranges, distributions per feature;
  decide keep/transform/drop. Re-run once BQ POI points populate H1/H3.
- **Registry + regression (Phase 3):** add EDA-validated features to `PREDICTOR_COLS`
  (+ `ZERO_FILL`/`MEDIAN_FILL`); rebuild model table; fit with both-mains-plus-interaction;
  check significance, VIF, residuals, Moran's I; validate transit coefficients against
  `*_rate`; test city-heterogeneity interaction.
- **(Optional)** drop a fresher mid-2025 SacRT feed zip and re-run (current Sacramento feed is
  a stale Jan–Apr 2025 snapshot).
- **Revisit feed coverage / expansion candidates (§3.2):** several cities run rail or streetcar
  under separate agency feeds not yet ingested (notably Detroit QLINE + People Mover, Chicago
  Metra/Pace, SF Caltrain). Decide whether to add them before finalizing
  `transit_route_mode_diversity` and rail-proximity — Detroit is bus-only in-feed today, which
  understates its rail nodes.
- **Harden the representative-date picker (`_resolve_service_date`):** the current single fixed
  anchor (`2025-06-04`, peak-service weekday) is a fragility point. It already self-corrects for
  holidays (the ≥80%-of-peak filter) and tolerates different feed windows (the anchor is a soft
  tiebreaker), but three residual risks remain: (1) **seasonality** — an absolute June anchor can
  bias toward summer schedules that drop school/seasonal routes; (2) **weekend-only overnight
  service** — sampling one weekday understates `overnight_flag`/H2 where night-owl service runs
  only Fri/Sat; (3) **absolute-year anchoring** — for a non-2025 feed, proximity-to-anchor decays
  to "closest feed boundary." Options to implement (recommend A+B+D, C optional): **A** feed-
  relative anchor (feed midpoint / median active date, or align to the crime-data year) instead
  of an absolute date; **B** multi-day-type sampling (weekday + Saturday + Sunday) combined via
  `overnight = OR`, `span = max`, trips = weekday or 5:1:1 weekly average — closes the H2 gap;
  **C** median trips/day over all typical weekdays in the feed instead of a single date; **D**
  explicit deterministic fallback ladder (explicit date → peak weekday ≥ threshold → best any-day
  with warn → hard error) plus per-feed telemetry (chosen date, day-of-week, trips, % of peak).
  Not urgent — current output is fine for the first EDA/regression pass.
- **Resolve the robbery offense-classification tension:** `hypothesis.md` guardrail 6 groups
  robbery with violent crime ("weaker/opposite for assault/robbery"), but the H1/H2/H3
  (catalog H7/H8/H7×H8) offense expectation lists robbery alongside larceny/MVT because it is
  acquisitive (cash + low guardianship). Decide how robbery should be treated in the transit
  offense-specificity tests and reconcile the wording across both docs.

---

## 1. Key hypotheses & research roots

> **Numbering note:** this plan uses a compact local scheme focused on the three hypotheses
> being executed now. They map to the full catalog in `docs/hypothesis.md` as: **H1** →
> H7 (risky-facility co-location, §3.7); **H2** → H8 (overnight/24-7 service, §3.8); **H3**
> → the H7×H8 interaction (§4.3). The supply/context features (stop density, proximity,
> service intensity, route-mode diversity) are catalog H1–H5 there and enter here as
> controls.

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
heterogeneity → interact with city / transit-share (Chicago is high-transit; Houston, Atlanta,
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
Cities → agencies: Chicago = CTA (+Metra/Pace optional); Houston = METRO; Atlanta = MARTA;
Pittsburgh = PRT; SF = SFMTA/Muni + BART (consider 511 SF Bay regional feed).

> **One snapshot, not the whole year.** GTFS has no "whole-year" feed — each zip is a snapshot valid
> only for its service date range. Transit *supply* (stop locations, spans, overnight service) is
> highly stable quarter to quarter, so for an annual, cross-sectional 2025 model we pull **one
> representative mid-2025 snapshot per city** (e.g. the feed active in June 2025), temporally aligned
> across all five cities, and record each `feed_version` for reproducibility. Only go multi-feed if we
> later model within-year change (e.g. a new rail line opening); in that case derive per-stop features
> per feed and take the **median across feeds** as the annual value, rather than concatenating raw stops.

### 3.1 Demand-side data — what exists, and why real-time is overkill here

Standard GTFS is **supply only**: stops, routes, trips, and scheduled times. It contains **no
passenger counts, no boardings, and no occupancy**. Ridership and real-time data live in separate
feeds. Here's the landscape and why we deliberately stay on the static side:

- **GTFS Realtime** — live vehicle positions, delays, and service alerts, refreshed every few
  seconds. It can carry an occupancy label, but only if the agency bothers to fill it in, and many
  don't. It is a live stream with no history, so it cannot be pulled retroactively for 2025.
- **GTFS-Ride** — an open standard built specifically for ridership, but almost no agency publishes
  it. Likely a dead end for our five cities.
- **APC / AFC (passenger counters and fare-card taps)** — the true ridership sources, and what the
  crowding research actually used. Usually not public, and access is a per-agency data request.
- **National Transit Database (NTD)** — public and does cover 2025, but only at the agency and route
  level, not stop or block group. Too coarse to be a block-group feature. We use it only as a
  **city-level moderator** for how transit-dependent each city is (Chicago high; Houston, Atlanta,
  Pittsburgh lower).

**Why real-time is overkill for this project:**

- Our crime target is **annual and cross-sectional** for 2025. Real-time feeds describe second-by-
  second conditions we would then have to average away back into a yearly number.
- Real-time data has **no 2025 history** to pull — we'd have to have been collecting it live all year.
- Ridership is **endogenous**: crime lowers ridership, so using it as a predictor muddies cause and
  effect. Stable **supply** features are the cleaner choice for an inferential model.
- We can **approximate demand from supply** we already plan to use: service intensity (trips per day),
  overnight span, and seat-capacity weighting (rail counts for more than bus) as a static stand-in for
  crowding pressure.

**Honest limitation to record:** static GTFS cannot measure *realized* crowding, so the crowding
channel behind H3 is approximated, not observed. If APC or fare-card data ever becomes available for a
city, revisit that channel then.

### 3.2 Feed inventory per city (and expansion candidates)

One primary agency feed is ingested per city (SF uses two). Modes below are the GTFS
`route_type`s actually present in each feed. **Most feeds are bus-only or bus-dominant; the
`transit_route_mode_diversity` feature only lights up where a city's own feed carries rail.**
Several metros run rail/streetcar under *separate* agency feeds we have not pulled — flagged
as expansion candidates. **Come back to this** before finalizing the mode-diversity feature:
adding the missing rail feeds would materially change Detroit's and Chicago's profiles.

| City | Feed (agency) | Modes in feed | Expansion candidates (separate feeds, not yet ingested) |
|------|---------------|---------------|----------------------------------------------------------|
| Chicago | CTA (`mdb-389`) | heavy rail ('L') + bus | **Metra** commuter rail, **Pace** suburban bus |
| Houston | METRO (`mdb-2060`) | light rail + bus | — (METRO covers the city) |
| Atlanta | MARTA (`mdb-368`) | heavy rail + streetcar + bus | CobbLinc / GRTA Xpress suburban bus |
| San Francisco | Muni (`mdb-2886`) + BART (`mdb-53`) | light rail, streetcar, cable car, heavy rail (BART) + bus | **Caltrain** (downtown terminus), SamTrans, Golden Gate |
| Pittsburgh | PRT (`mdb-409`) | light rail (the T) + funicular inclines + bus | — (PRT covers bus + rail + inclines) |
| Jacksonville | JTA (`tld-764`) | Skyway automated people-mover + ferry + bus | — (JTA covers all city modes) |
| Kansas City | KCATA / RideKC (`mdb-187`) | KC Streetcar (light rail) + bus | IRIS / Unified Gov (KCK) suburban bus |
| Sacramento | SacRT (`mdb-2137`) | light rail + bus | — (SacRT covers the city) |
| Detroit | DDOT (`mdb-464`) | **bus only** | **QLINE** streetcar, **Detroit People Mover** (monorail), **SMART** suburban bus |
| Columbus | COTA (`mdb-404`) | **bus only** | — (Columbus has no rail; CBUS circulator is a COTA bus) |

**Takeaways to revisit:**
- **Detroit** is bus-only in-feed today; its QLINE + People Mover are real rail nodes missing
  from the mode-diversity signal. Adding them would raise Detroit's diversity and rail-proximity
  scores (relevant to the Detroit-vs-KC comparison, which currently understates Detroit's rail).
- **Chicago** ingests CTA rail+bus but not Metra/Pace, so far-out BGs served only by commuter
  rail read as transit-poor. Consider Metra if edge coverage matters.
- Where "—" is listed, the single agency feed already spans the city's modes; no action needed.

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

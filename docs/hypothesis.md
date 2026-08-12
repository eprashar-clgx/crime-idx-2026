# Transit & Crime — Hypotheses, Theory Mapping, and Feature Shapes

Working document for the **block-group crime prediction** workstream. It maps each
candidate transit/environment variable to the criminological theory that licenses it,
states a falsifiable hypothesis, and enumerates the concrete feature shapes the variable
can take in the national BG matrix.

> **Design constraint (non-negotiable):** this project deliberately excludes demographic
> predictors (race, sex, age composition, income as a group trait). Theories that *require*
> demographic composition — **Social Disorganization** and **Lifestyle-Exposure** — are
> therefore out of scope as *justifications*. We lean on theories whose mechanism is
> **environmental / situational**: Routine Activity, Crime-as-Opportunity / Rational Choice,
> Crime Pattern Theory, and Risk Terrain Modeling (RTM). Note RTM is *not* demographic — it
> models crime from environmental risk factors (transit, bars, vacant lots) and is arguably
> the most license-appropriate framework we have.

> **Modeling context:** target is `*_logcount` = log(count + 1); `*_rate` (per 1,000 pop,
> plus a daytime-adjusted variant using LODES `c000` jobs) are kept as validators. Fit is
> inferential standardized OLS with HC3 robust SEs and a Moran's I spatial-autocorrelation
> check — no train/test split. Predictors are z-standardized at fit time.

---

## 1. Theories in play

| Theory | Core mechanism | Uses demographics? | What it licenses here |
|---|---|---|---|
| **Routine Activity Theory (RAT)** — Cohen & Felson 1979 | Crime = motivated offender + suitable target + absent capable guardian, converging in space/time | No | Transit as a **convergence node** that assembles offenders and targets; guardianship proxies |
| **Crime-as-Opportunity / Rational Choice** — Mayhew et al. 1976; Cornish & Clarke | Crime follows the momentary supply of easy opportunities | No | Opportunity-sensitive crimes (theft/larceny) respond to **crowding/exposure**, not volume per se |
| **Crime Pattern Theory** — Brantingham & Brantingham 1995 | Crime concentrates at **nodes**, **paths**, **edges**; places are *generators* vs *attractors* | No | Stops/stations as generators (bring people) & attractors (known opportunity); land-use mix |
| **Risk Terrain Modeling (RTM)** — Caplan & Kennedy | Model risk from co-located environmental risk factors | No | The *method*: transit is one spatial risk layer among many; combine layers |
| **CPTED** — natural surveillance | "Eyes on the street" / guardianship from design & activity mix | No | Guardianship & activity-density moderators |
| ~~Social Disorganization~~ | Collective efficacy from residential composition | **Yes** | Excluded as justification |
| ~~Lifestyle-Exposure~~ | Victim demographics drive exposure | **Yes** | Excluded as justification |

---

## 2. Cross-cutting methodological guardrails

These emerged from the literature review (2015–2026) and constrain *how* transit features
should enter the model. Every hypothesis below must be read against them.

1. **Count vs. per-person risk (the ambient-population confound).** *Bus Stops and Violence*
   (Solymosi/others, 2018, RTM + passenger offset) shows facilities are risk factors for
   crime **counts** but often **not** for **per-person victimization** — because they move
   people around. Since our target is `logcount`, a transit feature can load simply by
   proxying ambient population. **Mitigation:** always test transit features (a) against the
   `*_rate` validators and (b) with a population/jobs **exposure offset**, not just as raw
   count predictors.
2. **Non-monotonicity at BG scale.** Chicago transit-station-area study (2022, block-group,
   propensity-matched) finds a **U-shape**: dense CBD-TOD safest, mid-density TOD least safe,
   low-density TAD also high. A bare "closer/more stops → more crime" linear term is
   mis-specified. **Mitigation:** allow non-linear form (splines / quadratic) and
   land-use interactions.
3. **City heterogeneity (big-city artifact).** Most positive evidence is from heavy-transit
   cities (LA, NYC, Chicago, London, São Paulo, Montevideo). Our panel includes auto-oriented
   metros (Atlanta, Sacramento, Pittsburgh). **Mitigation:** interact transit with city /
   transit-mode-share; expect attenuation where transit carries few people.
4. **Endogeneity of ridership.** Crime depresses ridership (Texas-Triangle 2024 studies), so
   *ridership* is a poor exogenous predictor. **GTFS-derived supply** (stop density, service
   frequency) is cleaner than demand. Prefer supply.
5. **Opportunity ≠ volume.** *Opportunity in transit* (Díaz et al. 2026) shows **average
   passenger flow** predicts even household crimes (i.e. it is confounded by place), whereas
   **crowding/occupancy pressure** selectively predicts theft. Prefer crowding/exposure
   proxies over mean flow.
6. **Offense specificity.** Property/theft is opportunity-sensitive; violent crime is not (and
   often occurs in *emptier* settings). Expect transit effects concentrated in
   `larceny`/`mvt`/`burglary`, weaker/opposite for `assault`/`robbery`.

---

## 3. Variable → hypothesis → feature-shape catalog

For each variable: **H** (directional, falsifiable), the theory anchor, and candidate feature
shapes. "Offset" = enter with ambient-population exposure. Shapes marked ★ are the
recommended first cut.

### 3.1 Transit stop density
- **Theory:** Crime Pattern (nodes), RAT (convergence).
- **H1:** Net of ambient/daytime population, BG theft/larceny `logcount` rises with stop
  density; the effect on `*_rate` is weaker or null (consistent with the count-vs-risk
  confound). Effect attenuates in low-transit-share cities.
- **Feature shapes:**
  - ★ Count of stops in BG.
  - ★ Density = stops / BG land area (km²).
  - Kernel/decay-weighted stop density (Gaussian around centroid) to reduce MAUP edge effects.
  - Offset variant: stops with `log(pop + jobs)` as exposure.
  - Non-linear: quadratic or spline (test the U-shape).

### 3.2 Transit proximity
- **Theory:** Crime Pattern (paths/edges), RAT.
- **H2:** Shorter distance from BG centroid to nearest high-service stop/station is associated
  with higher property-crime `logcount`; relationship is non-linear (threshold within
  ~400–800 m walk shed), not a smooth linear gradient.
- **Feature shapes:**
  - ★ Distance from BG centroid to nearest stop (m).
  - Distance to nearest **rail/subway** station specifically (`route_type` 0/1/2).
  - Mean distance to k-nearest stops (k = 3–5) — smoother than single-nearest.
  - Binary "within X m of a station" walk-shed indicator (thresholded).
  - Inverse-distance / negative-exponential decay weight.

### 3.3 Service intensity (GTFS supply)
- **Theory:** RAT (guardianship fluctuates with schedule), Opportunity (exposure windows).
- **H3:** Higher scheduled service intensity (weekday trips/day, shorter headways) increases
  property-crime exposure and thus `logcount`, **more strongly than static stop counts**,
  because it proxies the *flow* of targets — but this is the channel most confounded by
  ambient population (guardrail 1), so its `*_rate` coefficient should shrink most.
- **Feature shapes:**
  - ★ Sum of weekday trips/day across stops in BG (from `stop_times` × `calendar`).
  - Mean headway (min) of routes serving the BG; peak vs off-peak split.
  - Seat-capacity-weighted service (weight rail > bus) as a crowding-pressure proxy
    (closest static analogue to Díaz et al.'s occupancy metric).
  - Offset variant strongly recommended here.

### 3.4 Crowding / occupancy pressure (proxy)
- **Theory:** Crime-as-Opportunity (Díaz et al. 2026).
- **H4:** Crowding pressure (riders per vehicle / stop dwell demand), **not** average flow,
  selectively elevates **theft/larceny** with little effect on robbery or violent crime.
- **Feature shapes:**
  - Scheduled arrivals concentrated in peak windows (peak trips ÷ span) as a static crowding
    proxy (true occupancy needs AFC/ridership data we may not have).
  - Ratio of service to nearby residential+job capacity.
  - **Caveat:** static GTFS cannot measure realized crowding; flag as a weaker proxy and,
    if ridership/AFC becomes available, revisit. Do **not** substitute mean flow (guardrail 5).

### 3.5 Route / mode diversity
- **Theory:** Crime Pattern (hub connectivity → more offender/target inflow).
- **H5:** BGs served by more distinct routes and more transit **modes** (`route_type`
  diversity) act as stronger convergence hubs → higher property-crime `logcount`.
- **Feature shapes:**
  - Count of distinct `route_id` serving the BG.
  - Count of distinct `route_type` (mode) present.
  - **Shannon equitability index** over route-type mix (see §4.1) — captures multi-modal hubs
    vs single-mode coverage.

### 3.6 Land-use / venue functional mix (guardianship & attractors)
- **Theory:** CPTED (natural surveillance), Crime Pattern (generators/attractors), RTM.
- **H6:** Functional **diversity** of the BG moderates transit effects: mixed-use, high-amenity
  areas supply guardianship (dampening), whereas single-function or amenity-rich-but-low-density
  areas around transit are most criminogenic (the Chicago-2022 U-shape). Enter as an
  **interaction** with transit, not just a main effect.
- **Feature shapes:**
  - **Venue diversity index** (Shannon equitability, §4.1) over POI/land-use categories.
  - **Offering advantage** (location quotient, §4.2) per key category (e.g. bars, ATMs,
    convenience retail) — flags locally over-represented attractors.
  - Activity density (jobs + pop per area) as a guardianship proxy.
  - ★ Interaction terms: `stop_density × venue_diversity`, `stop_density × activity_density`.

---

## 4. "Fancier" feature definitions

Adapted from Kadar & Pletikosa (2018), *Mining large-scale human mobility data for long-term
crime prediction*, EPJ Data Science — which predicts NYC census-tract yearly crime counts and
finds **ambient-population features beat census-only baselines** (R² up to ~65% geo-out-of-sample),
with the biggest gains for **grand larceny** and little gain for assault (mirrors our offense-
specificity guardrail).

### 4.1 Diversity index — Shannon equitability
Normalized Shannon entropy of category shares within a BG. For BG $t_i$ with $V_c(t_i)$ items
(venues, or transit route-types) of category $c$ and total $V(t_i)$ over category set $C$
(+1 smoothing to avoid zero-division):

$$
D(t_i) = -\sum_{c \in C}\left(\frac{1 + V_c(t_i)}{1 + V(t_i)}\;\ln\frac{1 + V_c(t_i)}{1 + V(t_i)}\right)\Big/\ln |C|
$$

- **Range** ≈ 0 (single dominant function) → 1 (maximally heterogeneous).
- **Use:** guardianship / functional-mix moderator (§3.6) and route-mode mix (§3.5).

### 4.2 Offering advantage — location quotient
How over-represented category $c$ is in BG $t_i$ vs the national/city average, i.e. a
TF-IDF-like local distinctiveness score:

$$
OA_c(t_i) = \frac{1 + V_c(t_i)}{1 + V(t_i)} \times \frac{\text{total\_items}}{\sum_{i=1}^{N} V_c(t_i)}
$$

- $OA > 1$ ⇒ the category is locally concentrated relative to the study area.
- **Use:** isolate specific **attractor** categories (bars, ATMs, transit hubs) whose local
  concentration — not mere presence — is criminogenic. More informative than a raw count for
  rare-but-risky categories.

> **Adaptation note:** Kadar & Pletikosa compute these over Foursquare venue categories with
> checkins/subway/taxi as ambient-population signals. We substitute (a) POI/land-use categories
> we can source nationally and (b) GTFS route-types for the transit variant. Their checkin/taxi
> ambient signals are the demand-side analogue we deliberately avoid in favor of GTFS supply
> (guardrail 4); treat their result as validation that ambient/exposure structure matters, and
> approximate it with supply + LODES daytime population.

---

## 5. Summary of recommended first-cut features

| # | Feature | Shape | Enters as | Guardrail hooks |
|---|---|---|---|---|
| 1 | Stop density | stops / km² | main + quadratic | 1, 2 |
| 2 | Nearest-stop distance | m to nearest (and nearest rail) | main + threshold | 2 |
| 3 | Weekday service intensity | Σ trips/day | main + offset | 1, 5 |
| 4 | Route-mode diversity | Shannon equitability over `route_type` | main | 3 |
| 5 | Venue/land-use diversity | Shannon equitability | interaction w/ transit | 2, 6 |
| 6 | Attractor offering advantage | location quotient (bars/ATMs) | main | 6 |
| 7 | Transit × guardianship | `stop_density × activity_density` | interaction | 2, 6 |
| 8 | Transit × city-transit-share | interaction | interaction | 3 |

Validate every transit coefficient against the `*_rate` validators and re-check Moran's I;
expect the strongest, most stable effects on `larceny`/`mvt`, weakest on violent categories.

---

## 6. Key references (2015–2026)

- Cohen & Felson (1979) — Routine Activity Theory.
- Brantingham & Brantingham (1995) — Crime Pattern Theory (generators/attractors).
- Mayhew, Clarke, Sturman & Hough (1976) — Crime as Opportunity.
- Phillips & Sandler (2015), *Reg. Sci. Urban Econ.* — station closures reduce nearby crime (causal).
- Kadar & Pletikosa (2018), *EPJ Data Science* — ambient-population features for tract-level crime; equitability & offering-advantage definitions.
- *Bus Stops and Violence, Are Risky Places Really Risky?* (2018) — RTM; count vs per-person risk.
- Chicago transit station areas & built environment (2022), *JAPA* — BG-level U-shape, guardianship.
- Díaz, Fernández, Fossati & Trajtenberg (2026), *Crime Science* — micro-temporal causal DiD; crowding→theft only; flow ≠ opportunity.

# Crime Concentration — Issue Tree & Feature Hypotheses

Working document for the next feature round. It asks one question: **why is a block group
in the top 20% of crime, the next 30%, or the bottom 50%?** It maps candidate factors
(beyond the demographic shares we already use) to that question, with data status.
Transit-specific theory lives in [`transit_hypothesis.md`](transit_hypothesis.md); parked
ideas in [`backlog.md`](backlog.md).

## 1. Why this framing

From the concentration analysis in `01_eda_02`:

- Gini is 0.4–0.6 in almost every city.
- The top 20% of BGs hold roughly 40–60% of crime; ~50% of BGs hold 80%.
- Detroit is the exception: crime is spread out, not concentrated.

The product ranks **where crime happens per property** (insurance use case), so the model's
first job is to **isolate the top 20%**, and its second job is to **order the next 30%**.
Feature rounds so far (neighbour context, store density, street morphology) gave small
pooled lifts: median city r² +0.015 (GBM), with New York, San Francisco and Philadelphia
getting worse. Working assumption: those features describe the *residential* tiers, and the
signal that defines the top tier is missing.

## 2. Constraints on candidate features

- **No jobs/employment predictors.** A state DOI rejected a jobs-within-1-mile feature
  (`docs/CONTEXT.md`, "regulator-rejected predictor").
- **No protected attributes or close proxies** (race, sex, age, income as a group trait).
- **Licensing:** public-domain or attribution-only sources. No share-alike (OSM/Overture are
  ODbL, see `roadway_plan.md` §5).
- **National scalability:** every feature must be buildable for any BG, not just the 20
  POC cities.

## 3. Issue tree

Grounded in routine activity theory: crime needs a suitable **target**, a motivated
**offender** present, and an absent **guardian**. Crime pattern theory adds where people
move (nodes, paths, edges).

| Branch | Factor | Proxy | Status |
|---|---|---|---|
| **1. Targets** — what's there to take | Retail / commercial presence (larceny is most property crime) | Parcels with 20+ addresses (§4) | Buildable (CoreLogic) |
| | Vehicles exposed (car theft, theft from vehicles) | Imagery hardscape share (parking/paving) | Have (`hardscapes_pct_avg`) |
| | Housing type and value | `lap_pct`, `det_pct`, own share; CoreLogic values | Have / buildable |
| **2. Activity** — who's present besides residents | Activity generators: restaurants, bars, hotels, venues, hospitals, schools | Firmographics NAICS counts | Buildable (`backlog.md`) |
| | Through-traffic | Transit service, arterial density, HPMS AADT | Have / buildable |
| | Daytime workers | LODES jobs | **Blocked (DOI)** |
| **3. Guardianship** — who's watching | Occupancy | Vacancy, owner-occupancy | Have |
| | Night-time activity / lighting | VIIRS night lights (public domain), 24h businesses | Buildable |
| | Nobody home at night | Share of non-residential parcels/addresses | Buildable |
| **4. Access & escape** | Highway access, grid permeability | Ramps, interstate distance, `x_ratio` | Have (weak) |
| **5. Disorder / neglect** | Physical decline, distress | Vacancy, roof condition, liens, foreclosures | Have |
| **6. Spatial spillover** | Proximity to hotspots | Queen-neighbour means/sums | Have (helped Detroit) |
| **7. Measurement artifacts** | Geocoding defaults, bulk reporting, tiny denominators | Police stations (§5), malls, property count | Diagnostic |

## 4. Multi-address parcels (retail / condo proxy)

Land-use codes are unreliable, so retail is proxied structurally: **parcels with ≥ 20
active addresses**. Source: `vw_parcel_to_address` (already used in `sql/build/vacancy.sql`),
rolled up to BG via `NS_pcl_universe_xref`.

A 20+-address parcel is one of three things, and they should mean different things for
crime:

| Type | Expected crime role |
|---|---|
| Strip mall / shopping centre | Target + activity generator (larceny, theft from vehicles) |
| Condo building | Residential density, more guardianship |
| Apartment complex (single owner) | Residential density, turnover |

Signals to tell them apart (cheapest first):

1. **Businesses on the parcel:** any firmographics match on the parcel's clips → commercial.
2. **Clips per parcel:** condos carry one clip per unit in `clip_list`; malls and rental
   complexes usually carry one or a few.
3. **Address unit designators:** `APT`/`UNIT` vs `STE`/`SUITE`.
4. **USPS residential delivery indicator** (if `vw_address_connect` carries one).
5. **Imagery:** footprint size, parking share around the structure.

Candidate BG features: count and per-km² of each type, and the share of the BG's addresses
on commercial multi-address parcels. Threshold (20) to be tuned (10 / 20 / 50).

## 5. Police stations

Source: Esri U.S. Federal Datasets "Police Stations" (USGS NGDA structures, FCODE 74034),
[ArcGIS Hub](https://hub.arcgis.com/datasets/fedmaps::police-stations/about), FeatureServer
`Structures_Law_Enforcement_v1/0`. 18,223 points nationally, CC BY 4.0 (attribution only).
A Chicago spot check found the district stations plus the headquarters, plus some duplicates
and generic "Chicago Police Department" records, so it needs dedup.

Two uses, with different value:

- **Diagnostic (priority):** flag BGs containing a station or headquarters and check
  whether they are over-represented in the top 20%. Walk-in reports and default geocodes are
  often recorded at the station address, which inflates that BG's count.
- **Predictor (low expectation):** distance from BG centroid to nearest station.
  - Sign is ambiguous: deterrence vs stations sited in high-crime areas.
  - With ~25 stations per big city, distance largely restates `city_centers_dist`.
  - Compliance risk: police proximity can read as a proxy for policing intensity, which
    correlates with protected attributes. Confirm with compliance before modelling.

## 6. Tier hypotheses

| Tier | Hypothesised profile | Branches that should separate it |
|---|---|---|
| Top 20% | Activity nodes: commercial strips, malls, downtown, transit hubs. Few residents, larceny-heavy, some reporting artifacts | 1, 2, 7 |
| Next 30% | Residential with distress and spillover from adjacent hotspots; burglary / vehicle theft | 3, 5, 6 |
| Bottom 50% | Stable owner-occupied, low permeability, little commercial land | — |

If true, we have been refining tier-2 features while the tier-1 signal is missing.

## 7. Test plan (read-only, before building data)

1. **Tier profile per city:** top 20 / next 30 / bottom 50 by observed rate, compared on
   existing features plus:
   - crime-category mix (larceny share),
   - property count and residential vs commercial address share,
   - presence of a police station / headquarters, and of multi-address parcels.
2. **Artifact check:** share of top-20% BGs whose crime is concentrated at a single
   geocoded point.
3. **Multi-address parcel prototype:** BQ read-only count of ≥ 20-address parcels per BG,
   split by "has a business on the parcel"; within-city Spearman vs rate and LOCO increment.
4. **Modelling:** if tiers behave differently, test a ranking objective (LightGBM
   LambdaRank) or a tier classifier against the capture metric, not r².

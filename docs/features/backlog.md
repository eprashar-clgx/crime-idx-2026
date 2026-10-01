# Feature backlog — roadway morphology & POIs

Ideas parked for later feature-engineering rounds. Each needs the usual gate before promotion:
within-city Spearman (sign agreement across 20 cities) → redundancy vs current predictors →
LOCO increment over the current selected set (ADR 0010; Ridge + LightGBM).

Prompted by Mao et al. (2025), *Humanities & Social Sciences Communications* 12:981 —
"Unraveling the Nexus: how street network morphology influences crime in Detroit"
(879 BGs, NB regression with street length as exposure; single-city, in-sample).

**Licensing constraint (roadway_plan.md §5):** public-domain sources only — TIGER/Line, FHWA
HPMS, CoreLogic internal data. No OSM / Overture (ODbL share-alike).

---

## 1. Roadway morphology

### Done — low-effort probe (2026-10-01, console only, not in repo)

Built from TIGER EDGES node degree over S1200/S1400 street edges (dead end = degree 1 across
all S1xxx road edges), deduped by TLID across counties. Within-city ρ vs `cl_total_rate`:

| Feature | median ρ [min, max] | sign | redundancy | outcome |
|---|---|---|---|---|
| `x_ratio` = deg≥4 / deg≥3 junctions | +0.21 [−0.12, +0.37] | 16/20 + | intersection_density 0.35 | **candidate** |
| `deadend_share` = dead ends / (dead ends + junctions) | −0.13 [−0.39, +0.09] | 16/20 − | x_ratio −0.40, intersection −0.44 | **candidate** (sqrt; 46% zero) |
| `x_density`, `street_density` | +0.16 / +0.10 | — | intersection 0.71 / 0.90 | drop |
| `t_density`, `alley_share` (S1730), `servpriv_share` (S1640/S1740) | ≈ 0 | — | — | drop |

LOCO increment over the final-10 set + agency anchor (wprop, 20 cities): Ridge +0.003 r2
(15/20 cities up); LightGBM −0.008 pooled r2 (NYC −0.057) but recall 0.399 → 0.409 (13/20 up).
→ Add `x_ratio` + `sqrt(deadend_share)` to the next stepwise candidate pool.

- [ ] Promote the probe into `roadway/tiger.py` (keep full node degree, not just ≥ 3) and
      `roadway/build.py`; `x_ratio` NaN when a BG has no junctions (9%) → city median.
- [ ] Load neighbouring counties as a buffer so county-edge nodes aren't false dead ends.

### Medium effort

- [ ] **Cell ratio (Marshall)** — closed street loops vs cul-de-sacs per BG.
  - Build: cyclomatic number of the street subgraph inside each BG (E − V + C), or
    `shapely.polygonize` road edges and count faces; cell_ratio = cells / (cells + dead ends).
  - Paper: + assault, robbery, larceny; n.s. MVT.
  - Open: boundary-clipping rule for faces straddling BGs; likely overlaps `deadend_share`.
- [ ] **Road-hierarchy length shares** — freeway (f_system 1–2) / arterial (3–4) /
      collector (5–6) / local share of road length.
  - Build: HPMS `f_system` clipped lengths; local = TIGER S1200+S1400 total − HPMS 1–6.
  - HPMS doesn't fully map local roads (f_system 7), so local must be a TIGER residual.
  - Paper: minor-road share + all four types; local share − larceny.
  - Open: arterial share vs existing `roadway_arterial_density` redundancy.
- [ ] **AADT / traffic exposure** — already deferred in roadway_plan.md §8; 10-city check
      found it redundant with arterial density / ramp count. Revisit only if hierarchy shares
      show signal.

### High effort

- [ ] **Betweenness centrality at 800 m (pedestrian permeability)** — length-weighted mean
      segment betweenness per BG.
  - Build: graph from TIGER EDGES (TNIDF/TNIDT, edge length) → distance-cutoff betweenness
    (igraph / networkit); also 1,500 m and global for comparison.
  - Paper: 800 m positive for all four crime types; larger radii n.s.
  - Open: compute cost for NYC/Chicago (buffer the graph beyond city limits to avoid edge
    effects); TIGER has no footpaths or one-way restrictions; check the licence of the
    library chosen.

---

## 2. Points of interest (CoreLogic firmographics via `STORE_DEFS` + `sql/build/stores.sql`)

Current POIs: `unq_convenience_stores_clips`, `unq_gas_stations_clips`,
`unq_liquor_stores_clips`. New categories = new `STORE_DEFS` entry + `FeatureSource`.
Verify NAICS codes against the firmographics view (2017/2022 dual vintage).

| POI | Hypothesis | Candidate NAICS | Paper finding | Notes |
|---|---|---|---|---|
| **Restaurants** | foot traffic / targets (RAT) | 7225xx (722511, 722513, 722514, 722515) | + all four types (strongest node effect) | split full- vs limited-service? |
| **Bars / drinking places** | alcohol disinhibition | 722410 | − robbery (counter-intuitive) | may overlap liquor stores |
| **Banks / credit unions** | cash targets | 522110, 522130 | n.s. (ATMs + banks) | branch proxy only |
| **ATMs** | cash targets (RTM risky facility) | none — standalone ATMs aren't businesses | n.s. | needs a locator source; also blocks `transit_risky_*` ATM co-location (transit_eda_plan.md) |
| **Check-cashing / pawn** | cash + motivated offenders | 522390, 522298 (verify) | — | small counts; binary flag likely |
| **Parking lots / garages** | MVT targets | 812930 | − MVT | |
| **Schools, places of worship, leisure** | guardianship / gatherings | 6111xx, 8131xx, 713xxx | leisure + larceny | sparse → presence dummies |

- [ ] Pull restaurants + bars first (largest counts, strongest paper signal).
- [ ] For each: count, presence dummy, and density; check sparsity (`sparse_zero_max`) and
      redundancy with the convenience-store count before LOCO.
- [ ] Find an ATM location source (internal or licensable); bank branches are a fallback.

---

## 3. Other parked ideas

- [ ] Transactions: decile shape check + partial ρ controlling for own/vacant/moved;
      transaction-type shares (investor, cash, quitclaim) and price features.
- [ ] Continuous distance to downtown to replace `city_centers_dist`.
- [ ] Best-so-far guard in `StopRule` (stepwise r2 drift).
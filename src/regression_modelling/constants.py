"""Registries + column lists for the regression_modelling task."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSource:
    name: str                       # 'vacancy' → cache file name
    backend: str                    # 'bq', 'gcs', or 'file' (materialized out-of-band)
    location: str                   # BQ: pull-sql name | GCS: path template | file: builder note
    key_col: str = "geoid"          # join key in the pulled data
    feature_cols: tuple = ()        # predictor columns to keep


@dataclass(frozen=True)
class TransitFeed:
    """One GTFS feed for a city (a city may have several, e.g. SF = Muni + BART).

    Feeds are identified by their stable Mobility Database file stem (`feed_id`, e.g.
    ``mdb-389`` or ``tld-764``); the downloaded zip is resolved by glob
    (``{feed_id}-*.zip``) so re-downloading a fresher snapshot of the same feed does not
    break paths.
    """
    agency: str                     # short label, e.g. 'cta', 'bart'
    feed_id: str                    # Mobility Database file stem; zip = {feed_id}-*.zip


# Per-city GTFS feeds, keyed by the same keys as CITIES (crime_blockgroup_mapping).
# One representative mid-2025 snapshot per feed (record feed_version on download).
# SF unions Muni + BART; shared stations are deduped by proximity in feeds.load_city_stops.
TRANSIT_FEEDS = {
    "chicago":       (TransitFeed("cta",   "mdb-389"),),
    "houston":       (TransitFeed("metro", "mdb-2060"),),
    "atlanta":       (TransitFeed("marta", "mdb-368"),),
    "san_francisco": (TransitFeed("muni",  "mdb-2886"), TransitFeed("bart", "mdb-53")),
    "pittsburgh":    (TransitFeed("prt",   "mdb-409"),),
    # Secondary (property-only crime) cities — transit supply features still valid.
    "jacksonville":  (TransitFeed("jta",   "tld-764"),),
    "kansas_city":   (TransitFeed("kcata", "mdb-187"),),
    "sacramento":    (TransitFeed("sacrt", "mdb-2137"),),
    "detroit":       (TransitFeed("ddot",  "mdb-464"),),
    "columbus":      (TransitFeed("cota",  "mdb-404"),),
}

# Representative service date to pin trips/day and service span (a typical Wednesday,
# feed active ~June 2025). Overnight window lives in transit.feeds.
TRANSIT_REPRESENTATIVE_DATE = "2025-06-04"


FEATURE_SOURCES = {
    "vacancy": FeatureSource(
        name="vacancy",
        backend="bq",
        location="vacancy",         # → sql/pull/vacancy.sql
        key_col="geoid",
        feature_cols=("vacant_pct", "vacant_pct_lag6"),
    ),
    "liens": FeatureSource(
        name="liens",
        backend="bq",
        location="liens",          # → sql/pull/liens.sql
        key_col="geoid",
        feature_cols=("clip_liens_pct", "clip_liens_pct_lag6"),
    ),
    "foreclosures": FeatureSource(
        name="foreclosures",
        backend="bq",
        location="foreclosures",     # → sql/pull/foreclosures.sql
        key_col="geoid",
        feature_cols=("clip_foreclosure_pct", "clip_foreclosure_pct_lag6"),
    ),
    "convenience_stores": FeatureSource(
        name="convenience_stores",
        backend="bq",
        location="convenience_stores",  # → sql/pull/convenience_stores.sql
        key_col="geoid",
        feature_cols=("unq_convenience_stores_clips",),
    ),
    "gas_stations": FeatureSource(
        name="gas_stations",
        backend="bq",
        location="gas_stations",      # → sql/pull/gas_stations.sql
        key_col="geoid",
        feature_cols=("unq_gas_stations_clips",),
    ),
    "liquor_stores": FeatureSource(
        name="liquor_stores",
        backend="bq",
        location="liquor_stores",     # → sql/pull/liquor_stores.sql
        key_col="geoid",
        feature_cols=("unq_liquor_stores_clips",),
    ),
    # Transit is materialized out-of-band by transit.build.build_all_transit (backend="file").
    # Covers the 10 ingested cities only; null elsewhere on the national spine. feature_cols are
    # the candidate BG predictors (docs/features/transit_eda_plan.md §5); the non-geo ones are
    # promoted to PREDICTOR_COLS, the risky (POI-geo) ones stay gated until the BQ pull lands.
    "transit": FeatureSource(
        name="transit",
        backend="file",
        location="build via regression_modelling.data_wrangling.transit.build_all_transit",
        key_col="geoid",
        feature_cols=(
            "transit_stop_count",
            "transit_stop_density",
            "transit_nearest_stop_m",
            "transit_service_intensity",
            "transit_overnight_stop_count",
            "transit_overnight_stop_share",
            "transit_risky_stop_count",
            "transit_risky_stop_share",
            "transit_risky_allnight_count",
            "transit_route_mode_diversity",
        ),
    ),
    # Imagery (Vexcel aerial structure features) — BG averages of per-structure roof/parcel
    # condition, built by sql/build/imagery.sql (structure-level -> BG via the parcel xref).
    # feature_cols are pulled into the matrix for EDA; they are NOT yet in PREDICTOR_COLS
    # (see IMAGERY_PREDICTORS) — promote after 01_eda decides which carry signal.
    "imagery": FeatureSource(
        name="imagery",
        backend="bq",
        location="imagery",          # → sql/pull/imagery.sql
        key_col="geoid",
        feature_cols=(
            "roof_condition_avg",
            "roof_debris_pct_avg",
            "roof_discoloration_pct_avg",
            "hardscapes_pct_avg",
            "roof_missing_material_pct",
            "imagery_structure_count",
        ),
    ),
    # Roadway (TIGER + FHWA HPMS, public domain) — materialized out-of-band by roadway.build_all_roadway
    # (backend="file"), mirroring transit. National-coverage layer (every BG in the model
    # cities gets a real value; no structural nulls the way transit has stopless BGs).
    # feature_cols are the candidate BG predictors (docs/features/roadway_plan.md §4);
    # CANDIDATE only — not yet in PREDICTOR_COLS until distribution/correlation EDA decides.
    "roadway": FeatureSource(
        name="roadway",
        backend="file",
        location="build via regression_modelling.data_wrangling.roadway.build_all_roadway",
        key_col="geoid",
        feature_cols=(
            "roadway_nearest_ramp_m",
            "roadway_nearest_interstate_m",
            "roadway_ramp_count",
            "roadway_arterial_density",
            "roadway_intersection_density",
        ),
    ),
}

# Store universes for the generic block-group builder (sql/build/stores.sql).
# {store} = table/column stem (bg_{store}, unq_{store}_clips); value = firmographics
# match predicate (NAICS/SIC codes and/or business-name LIKEs).
# NAICS codes are dual-vintage: subsector 447 (Gasoline Stations, 2017) was renumbered
# to 457 in NAICS 2022, and 445120 (Convenience Stores, 2017) became 445131 (Convenience
# Retailers, 2022). The firmographics view mixes vintages, so each concept lists both.
# Split is non-overlapping: gas-with-mart -> convenience (attractor); fuel-only -> gas.
STORE_DEFS = {
    "convenience_stores": (
        # convenience stores (445131=2022, 445120=2017) + gas w/ convenience mart (457110=2022, 447110=2017)
        "naics_6_digit_primary_code IN ('445131','445120','457110','447110')"
    ),
    "gas_stations": (
        # fuel-only stations (457120=2022, 447190=2017); mart stations counted under convenience
        "naics_6_digit_primary_code IN ('457120','447190')\n"
        "     AND NOT LOWER(business_name) LIKE '%charging station%'"
    ),
    "liquor_stores": (
        # beer/wine/liquor retailers, 4453x covers 445310 (2017) & 445320 (2022)
        "naics_6_digit_primary_code LIKE '4453%'\n"
        "     AND (LOWER(business_brand_name) LIKE '%liquor%'"
        " OR LOWER(business_name) LIKE '%liquor%')"
    ),
}

# crime categories we model (each has a *_count and *_rate column downstream)
TARGET_CATEGORIES = [
    "cl_total", "violent", "property",
    "assault", "murder", "rape", "robbery",
    "burglary", "larceny", "mvt",
]

# Predictors that vary within a single city (Division is constant per-city → excluded).
# Organized into semantic families so analysis code can select a slice (e.g. correlation on
# TRANSIT_PREDICTORS) without minting a per-purpose constant. PREDICTOR_COLS below is DERIVED
# from these groups — it is the single active fit-set and cannot drift from its parts.
DEMOGRAPHIC_PREDICTORS = [
    "moved1yr_pct",         # Percentage of households moving in past year
    "own_pct",              # Percentage of owner-occupied housing units
    "lap_pct",              # Percentage of housing units in 5+ unit structures
    "city_centers_dist",    # Distance in miles from central business district of nearest city
    "pop_est_5mile",        # Population density within 5 miles
    "pop_ch_1mile",         # Population change within 1 mile
]
# det_pct (single-family-detached share) removed: strongly collinear with own_pct / lap_pct
# (the tenure/structure trio moved together), so it added variance-inflation without signal.

# Functional form for demographic predictors in modeling. `pop_est_5mile` is a raw
# population COUNT within a 5-mile ring, extremely right-skewed (~46 → 2.3M, skew ≈ 5.7).
# Left raw it dominates the design matrix and — under the log1p target with a ridge/robust
# fit — produces expm1 blow-ups on the largest-population units (in the 2026-09-10 agency
# ablation a sparse model's held-out adj R² collapsed to −0.76 the moment raw pop_est_5mile
# entered). log1p-compressing it into `pop_est_5mile_log` removes the instability and lifts
# the standalone our-set adj R² from 0.22 → 0.36; the baseline national model log-scales the
# same feature identically (`zlg10_pop_est_5mile`). See ADR 0006. `pop_ch_1mile` is a bounded
# % change (−83 → +94) and stays raw. Consumed by apply_transforms (no has_transit / hurdle).
DEMOGRAPHIC_MODEL_TRANSFORMS = {
    "pop_est_5mile": "log1p",
}

# Demographic predictors in model form (what enters PREDICTOR_COLS): DEMOGRAPHIC_PREDICTORS
# with the transformed columns renamed to the apply_transforms `{col}_log` convention.
# DERIVED so it cannot drift from the raw list or the transform spec.
DEMOGRAPHIC_MODEL_PREDICTORS = [
    f"{c}_log" if c in DEMOGRAPHIC_MODEL_TRANSFORMS else c
    for c in DEMOGRAPHIC_PREDICTORS
]

PROPERTY_PREDICTORS = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
]

# KNN(6) spatial-lag columns: the within-state 6-nearest-neighbour mean of the RAW pct
# (self excluded), computed in BigQuery (data_wrangling/sql/build/{vacancy,liens,
# foreclosures}.sql). They capture the surrounding neighbourhood's distress level so the
# fit sees a transferable spatial gradient, not city-specific coordinates. Present only
# after the BQ ingestion is re-run; the pipeline skips them until then.
PROPERTY_LAG_COLS = [
    "vacant_pct_lag6",
    "clip_liens_pct_lag6",
    "clip_foreclosure_pct_lag6",
]

# Functional form for property predictors in modeling/EDA. The three distress shares
# (vacancy, liens, foreclosures) and their spatial lags are right-skewed → log1p into
# `{col}_log`; the POI store counts stay raw. Consumed by apply_transforms (no has_transit
# indicator / hurdle — those are transit-only).
PROPERTY_MODEL_TRANSFORMS = {
    "vacant_pct":                 "log1p",
    "clip_liens_pct":             "log1p",
    "clip_foreclosure_pct":       "log1p",
    "vacant_pct_lag6":            "log1p",
    "clip_liens_pct_lag6":        "log1p",
    "clip_foreclosure_pct_lag6":  "log1p",
}

# Retained property predictors in model form (what actually enters PREDICTOR_COLS): the
# log distress shares + raw store counts + the log spatial lags. Lag names follow the
# apply_transforms `{col}_log` convention.
PROPERTY_MODEL_PREDICTORS = [
    "vacant_pct_log",
    "clip_liens_pct_log",
    "clip_foreclosure_pct_log",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
    "vacant_pct_lag6_log",
    "clip_liens_pct_lag6_log",
    "clip_foreclosure_pct_lag6_log",
]

# transit (GTFS) — non-geo supply/exposure + overnight features, POC cities only.
# See docs/features/transit_eda_plan.md. Raw columns; functional form for modeling/EDA is
# given by TRANSIT_MODEL_TRANSFORMS below. The risky-facility co-location (H1) + interaction
# (H3) columns need the BigQuery POI point pull to populate (emit 0 offline) — gated here:
#   "transit_risky_stop_count", "transit_risky_stop_share", "transit_risky_allnight_count"
TRANSIT_PREDICTORS = [
    "transit_stop_count",
    "transit_stop_density",
    "transit_nearest_stop_m",
    "transit_service_intensity",
    "transit_overnight_stop_count",
    "transit_overnight_stop_share",
    "transit_route_mode_diversity",
]

# Functional form for transit predictors in modeling/EDA (see distribution EDA,
# docs/features/transit_eda_plan.md §5). Single source of truth consumed by
# feature_engineering.transforms.apply_transforms:
#   "log1p"    → add a compressed `{col}_log` column (tames right-skewed counts/distance)
#   "identity" → use the raw bounded column as-is (shares, diversity ∈ [0,1])
# The structural zeros (stopless BGs) are split into a separate `transit_has_transit`
# indicator (derived from transit_stop_count > 0), so "no transit" ≠ "little transit".
# NOTE: Pearson corr/OLS see these forms directly; Spearman is transform-invariant.
TRANSIT_MODEL_TRANSFORMS = {
    "transit_stop_count":           "log1p",
    "transit_stop_density":         "log1p",
    "transit_nearest_stop_m":       "log1p",
    "transit_service_intensity":    "log1p",
    "transit_overnight_stop_count": "log1p",
    "transit_overnight_stop_share": "identity",
    "transit_route_mode_diversity": "identity",
}

# Retained transit predictors in model form (the redundancy-pruned set from the correlation
# EDA — see docs/features/transit_stats.md). These are produced by
# feature_engineering.transforms.apply_transforms(hurdle=True) and are what actually enter
# PREDICTOR_COLS / the OLS. Names follow the transform naming convention (kept as literals to
# avoid a constants→feature_engineering import cycle):
#   transit_has_transit             extensive margin (1[stop_count > 0])
#   transit_service_intensity_logc  intensive supply, log1p centered on served mass (hurdle)
#   transit_nearest_stop_m_log      access/proximity, log1p
#   transit_overnight_stop_share    H2 nighttime exposure, raw bounded [0,1]
TRANSIT_MODEL_PREDICTORS = [
    "transit_has_transit",
    "transit_service_intensity_logc",
    "transit_nearest_stop_m_log",
    "transit_overnight_stop_share",
]

# imagery (Vexcel aerial structure features) — BG averages of per-structure roof/parcel
# condition. CANDIDATE predictors, deliberately NOT in PREDICTOR_COLS yet: pulled into the
# feature matrix (FEATURE_SOURCES["imagery"]) for distribution EDA, promoted to the fit-set
# only after 01_eda shows which columns carry signal. `imagery_structure_count` is a
# coverage/EDA column (structures backing each BG average), not itself a candidate predictor.
IMAGERY_PREDICTORS = [
    "roof_condition_avg",           # avg Vexcel roof condition score
    "roof_debris_pct_avg",          # avg roof debris %
    "hardscapes_pct_avg",           # avg parcel hardscape %
    "roof_missing_material_pct",    # share of structures with missing roof material
]
# roof_discoloration_pct_avg removed: strongly collinear with roof_condition_avg (both
# proxy the same roof-degradation signal), so only roof_condition_avg is retained.

# roadway (TIGER + FHWA HPMS) — highway-access/edge (R1), arterial (R2), permeability (R3,
# exploratory) candidates. CANDIDATE predictors, deliberately NOT in PREDICTOR_COLS yet:
# see docs/features/roadway_plan.md §4/§7 — pull into the feature matrix for distribution +
# correlation EDA first, promote a retained (functional-form) set after that (mirrors how
# TRANSIT_PREDICTORS -> TRANSIT_MODEL_PREDICTORS worked).
ROADWAY_PREDICTORS = [
    "roadway_nearest_ramp_m",
    "roadway_nearest_interstate_m",
    "roadway_ramp_count",
    "roadway_arterial_density",
    "roadway_intersection_density",
]

# Active fit-set: demographic (model form: log1p population ring count) + property (model
# form: log distress shares + spatial lags + store counts) + transit (model form) + imagery.
# PREDICTOR_COLS is DERIVED so it cannot drift from its parts. The raw DEMOGRAPHIC_PREDICTORS
# / PROPERTY_PREDICTORS / TRANSIT_PREDICTORS are the transform *inputs* (and imputation
# targets in ZERO_FILL/MEDIAN_FILL); they are replaced here by the model-form lists. Spatial-
# lag entries stay dormant until the BQ ingestion adds their raw columns — the pipeline skips
# any predictor whose source column is absent.
PREDICTOR_COLS = [*DEMOGRAPHIC_MODEL_PREDICTORS, *PROPERTY_MODEL_PREDICTORS,
                  *TRANSIT_MODEL_PREDICTORS, *IMAGERY_PREDICTORS]

# ── Transit A/B variants (step 2) ────────────────────────────────────────────
# Two interchangeable transit representations for the same fit-set, compared under the
# prediction harness:
#   GTFS  — the default `TRANSIT_MODEL_PREDICTORS` (per-city GTFS supply/exposure; richer but
#           needs a feed per city → only the 10 ingested cities carry it).
#   ACS   — commute-by-public-transit share (+ zero-vehicle households, a transit-dependence
#           proxy); nationally available for EVERY BG (deployable to any city), governance-
#           clean (commute mode / vehicle access, not a protected class).
# Both are DERIVED from PREDICTOR_COLS so they cannot drift; the ACS variant swaps the GTFS
# transit block out and the ACS transit block in. Pass either to `run_loco`/`run_holdout`
# via `predictors=`; the active default fit-set (PREDICTOR_COLS) stays GTFS.
ACS_TRANSIT_PREDICTORS = ["transit_pct", "veh0_pct"]
ACS_TRANSIT_MODEL_TRANSFORMS = {"transit_pct": "log1p", "veh0_pct": "log1p"}
ACS_TRANSIT_MODEL_PREDICTORS = ["transit_pct_log", "veh0_pct_log"]

GTFS_TRANSIT_PREDICTOR_COLS = list(PREDICTOR_COLS)                       # == default
ACS_TRANSIT_PREDICTOR_COLS = ([p for p in PREDICTOR_COLS if p not in TRANSIT_MODEL_PREDICTORS]
                              + ACS_TRANSIT_MODEL_PREDICTORS)

# ── Existing-model (Department-approved) features ported to BG level (POC idea #1) ─────────
# The incumbent national model's governance-APPROVED predictors (03_agency_comparison
# `dept_approved`), MINUS the four our set already represents (pop-ring, vacancy, moved-1yr,
# CBD-distance → `dept_redundant`). Split by the AXIS each acts on so the POC can show them
# apart:
#   WITHIN-city varying → real BG-level signal that can lift within-city RANKING.
#   BETWEEN-city level  → Census-division dummies, CONSTANT within a city (each POC city sits
#                         in exactly one division). They only shift cross-city LEVEL and demean
#                         to zero in within-city models — the incumbent's analog of the agency
#                         anchor below.
APPROVED_WITHIN_PREDICTORS = ["in_household_pct", "det_pct"]
APPROVED_LEVEL_PREDICTORS = ["div_encentral", "div_midatlantic", "div_southatlantic"]
APPROVED_EXISTING_PREDICTORS = APPROVED_WITHIN_PREDICTORS + APPROVED_LEVEL_PREDICTORS

# Census Division code (bg_predictors 'Division') -> incumbent dummy. 2=Mid-Atlantic,
# 3=East North Central, 5=South Atlantic — the three the original 22-feature model carried;
# every other division is the reference category.
DIVISION_DUMMIES = {"div_midatlantic": 2, "div_encentral": 3, "div_southatlantic": 5}

# ── Observed lagged UCR agency crime anchor (Model D leveller, ADR 0007) ───────────────────
# One observed per-city crime level from the year BEFORE the target, broadcast to every BG in
# the city (constant within city → a between-city LEVEL feature, like the division dummies).
# Target-paired: wtotal models use the lagged wtotal level, wprop models the lagged wprop.
AGENCY_ANCHOR_COL = {"wtotal": "agency_lag_wtotal_log", "wprop": "agency_lag_wprop_log"}

# ── POC predictor SETS (resolved by models/dataset.predictor_set) ─────────────────────────
# DERIVED so they cannot drift from their parts. The agency anchor is added per-target by the
# experiment runner (AGENCY_ANCHOR_COL) rather than hard-listed, since its column is
# target-specific.
PREDICTOR_SETS = {
    "ours": list(PREDICTOR_COLS),
    "ours+approved": list(PREDICTOR_COLS) + APPROVED_EXISTING_PREDICTORS,
}

ZERO_FILL = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "vacant_pct_lag6",              # spatial lags: 0 = no distress in the neighbourhood
    "clip_liens_pct_lag6",
    "clip_foreclosure_pct_lag6",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
    # transit: null on the national spine / BGs with no stop = genuinely 0 transit
    "transit_stop_count",
    "transit_stop_density",
    "transit_service_intensity",
    "transit_overnight_stop_count",
    "transit_overnight_stop_share",
    "transit_route_mode_diversity",
    # ACS-transit variant inputs: 0 = no transit commuters / no zero-vehicle households
    "transit_pct",
    "veh0_pct",
    # roadway: national TIGER layer, no structural nulls — 0 = no ramp/road segment observed
    "roadway_ramp_count",
    "roadway_arterial_density",
    "roadway_intersection_density",
    # "transit_risky_stop_count",      # promote with the risky predictors above
    # "transit_risky_stop_share",
    # "transit_risky_allnight_count",
]                                                       # 0 = none observed
MEDIAN_FILL = [
    "city_centers_dist", "pop_est_5mile", "pop_ch_1mile",  # 0 would be wrong
    "transit_nearest_stop_m",                              # distance; 0 = stop at centroid
    "in_household_pct", "det_pct",                         # incumbent approved shares (bounded)
    "roadway_nearest_ramp_m", "roadway_nearest_interstate_m",  # distance; 0 = ramp at centroid
]

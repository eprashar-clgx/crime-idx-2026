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
        feature_cols=("vacant_pct",),
    ),
    "liens": FeatureSource(
        name="liens",
        backend="bq",
        location="liens",          # → sql/pull/liens.sql
        key_col="geoid",
        feature_cols=("clip_liens_pct",),
    ),
    "foreclosures": FeatureSource(
        name="foreclosures",
        backend="bq",
        location="foreclosures",     # → sql/pull/foreclosures.sql
        key_col="geoid",
        feature_cols=("clip_foreclosure_pct",),
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
    "det_pct",              # Percentage of housing units that are single family detached houses
    "moved1yr_pct",         # Percentage of households moving in past year
    "own_pct",              # Percentage of owner-occupied housing units
    "lap_pct",              # Percentage of housing units in 5+ unit structures
    "city_centers_dist",    # Distance in miles from central business district of nearest city
    "pop_est_5mile",        # Population density within 5 miles
    "pop_ch_1mile",         # Population change within 1 mile
]

PROPERTY_PREDICTORS = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
    "unq_convenience_stores_clips",
    "unq_gas_stations_clips",
    "unq_liquor_stores_clips",
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
    "roof_discoloration_pct_avg",   # avg roof discoloration %
    "hardscapes_pct_avg",           # avg parcel hardscape %
    "roof_missing_material_pct",    # share of structures with missing roof material
]

# Active fit-set: demographic + property (raw) + transit (model form). PREDICTOR_COLS is
# DERIVED so it cannot drift from its parts. The raw TRANSIT_PREDICTORS above are the
# transform *inputs* (and imputation targets in ZERO_FILL/MEDIAN_FILL); they are replaced
# here by the pruned TRANSIT_MODEL_PREDICTORS.
PREDICTOR_COLS = [*DEMOGRAPHIC_PREDICTORS, *PROPERTY_PREDICTORS, *TRANSIT_MODEL_PREDICTORS, *IMAGERY_PREDICTORS]

ZERO_FILL = [
    "vacant_pct",
    "clip_liens_pct",
    "clip_foreclosure_pct",
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
    # "transit_risky_stop_count",      # promote with the risky predictors above
    # "transit_risky_stop_share",
    # "transit_risky_allnight_count",
]                                                       # 0 = none observed
MEDIAN_FILL = [
    "city_centers_dist", "pop_est_5mile", "pop_ch_1mile",  # 0 would be wrong
    "transit_nearest_stop_m",                              # distance; 0 = stop at centroid
]

"""Build the per-city model table: crime target ⋈ predictor features, inside-city, imputed."""
import numpy as np
import pandas as pd

from core.config import CITIES, INTERIM_DIR, PROCESSED_DIR
from core import geo_utils as geo
from prediction.features import assemble_features

# crime categories we model (each has a *_count and *_rate column downstream)
TARGET_CATEGORIES = [
    "cl_total", "violent", "property",
    "assault", "murder", "rape", "robbery",
    "burglary", "larceny", "mvt",
]

# predictors that vary within a single city (Division is constant per-city → excluded)
PREDICTOR_COLS = [
    "det_pct", "in_household_pct", "moved1yr_pct",
    "city_centers_dist", "pop_est_5mile", "pop_ch_1mile",
    "vacant_pct", "clip_liens_pct",
]

ZERO_FILL   = ["vacant_pct", "clip_liens_pct"]                       # 0 = none observed
MEDIAN_FILL = ["city_centers_dist", "pop_est_5mile", "pop_ch_1mile"] # 0 would be wrong


def build_bg_crime(city: str, refresh: bool = False) -> pd.DataFrame:
    """Run the core geo pipeline → BG-level crime target (counts, rates, population).
    Caches to data/interim/bg_crime/{city}.parquet."""
    path = INTERIM_DIR / "bg_crime" / f"{city}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    cfg = CITIES[city]
    city_gdf = geo.load_city_boundary(cfg)
    bg_gdf   = geo.load_state_block_groups(cfg)
    bg_gdf   = geo.label_bgs_within_city(bg_gdf, city_gdf)

    crime_gdf = geo.load_crime_data(cfg)
    crime_bg  = geo.sjoin_crimes_to_bgs(crime_gdf, bg_gdf)
    crime_bg  = geo.map_crime_categories(crime_bg, cfg)
    bg_cat    = geo.aggregate_by_bg_category(crime_bg)
    bg_all    = geo.merge_all_bgs_with_crimes(bg_gdf, bg_cat)

    # population (from crime-risk model, inner join) + rate normalization (pop and pop+jobs)
    crisk      = geo.load_model_data(bg_all)
    comparison = geo.merge_model_with_actuals(bg_all, crisk)
    comparison = geo.normalize_actuals(comparison, lodes_bg=geo.load_lodes_bg())

    out = pd.DataFrame(comparison.drop(
        columns=[c for c in ["geometry", "bg_geo"] if c in comparison.columns]))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path)
    return out


def build_model_table(city: str, refresh: bool = False,
                      inside_city_only: bool = True) -> pd.DataFrame:
    """Join features to the city crime target, filter to inside-city BGs, impute, log-transform."""
    target = build_bg_crime(city, refresh=refresh)
    feats  = assemble_features(refresh=False)   # national BG predictor matrix

    count_cols = [f"{c}_count" for c in TARGET_CATEGORIES]
    rate_cols  = [f"{c}_rate"  for c in TARGET_CATEGORIES]
    keep = (["geoid", "within_city", "population"]
            + [c for c in count_cols + rate_cols if c in target.columns])

    df = feats.merge(target[keep], on="geoid", how="inner")
    if inside_city_only:
        df = df[df["within_city"] == True].copy()

    # explicit per-column imputation of predictors
    df[ZERO_FILL]   = df[ZERO_FILL].fillna(0)
    df[MEDIAN_FILL] = df[MEDIAN_FILL].fillna(df[MEDIAN_FILL].median())

    # log(count + 1) targets (primary); *_rate kept as validators
    for c in count_cols:
        if c in df.columns:
            df[c.replace("_count", "_logcount")] = np.log1p(df[c])

    out = PROCESSED_DIR / "prediction" / f"{city}_model_table.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"{city}: model table {df.shape} → {out.relative_to(PROCESSED_DIR.parent.parent)}")
    return df
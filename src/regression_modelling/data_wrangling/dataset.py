"""Build the per-city model table: crime target ⋈ predictor features, inside-city, imputed."""
import numpy as np
import pandas as pd

from crime_blockgroup_mapping.config import INTERIM_DIR, PROCESSED_DIR
from crime_blockgroup_mapping.constants import CITIES
from crime_blockgroup_mapping.boundaries import (
    load_city_boundary, load_state_block_groups, label_bgs_within_city,
)
from crime_blockgroup_mapping.crime import (
    load_crime_data, sjoin_crimes_to_bgs, map_crime_categories,
    aggregate_by_bg_category, merge_all_bgs_with_crimes,
)
from crime_blockgroup_mapping.rates import (
    load_model_data, merge_model_with_actuals, normalize_actuals, load_lodes_bg,
)
from crime_blockgroup_mapping.scores import compute_weighted_scores
from regression_modelling.constants import (
    TARGET_CATEGORIES, PREDICTOR_COLS, ZERO_FILL, MEDIAN_FILL, PROPERTY_MODEL_TRANSFORMS,
    DEMOGRAPHIC_MODEL_TRANSFORMS,
)
from regression_modelling.data_wrangling.features import assemble_features
from regression_modelling.feature_engineering.transforms import apply_transforms


def build_bg_crime(city: str, refresh: bool = False) -> pd.DataFrame:
    """Run the shared crime_blockgroup_mapping pipeline → BG-level crime target
    (counts, rates, population). Caches to data/interim/bg_crime/{city}.parquet."""
    path = INTERIM_DIR / "bg_crime" / f"{city}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    cfg = CITIES[city]
    city_gdf = load_city_boundary(cfg)
    bg_gdf   = load_state_block_groups(cfg)
    bg_gdf   = label_bgs_within_city(bg_gdf, city_gdf)

    crime_gdf = load_crime_data(cfg)
    crime_bg  = sjoin_crimes_to_bgs(crime_gdf, bg_gdf)
    crime_bg  = map_crime_categories(crime_bg, cfg)
    bg_cat    = aggregate_by_bg_category(crime_bg)
    bg_all    = merge_all_bgs_with_crimes(bg_gdf, bg_cat)

    # population (from crime-risk model, inner join) + rate normalization (pop and pop+jobs)
    crisk      = load_model_data(bg_all)
    comparison = merge_model_with_actuals(bg_all, crisk)
    comparison = normalize_actuals(comparison, lodes_bg=load_lodes_bg())

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

    count_cols   = [f"{c}_count" for c in TARGET_CATEGORIES]
    rate_cols    = [f"{c}_rate"  for c in TARGET_CATEGORIES]
    # daytime-adjusted rate (per 1K of population + LODES jobs) kept alongside the plain
    # population rate so the two denominators can be A/B-compared as modeling targets.
    daytime_cols = [f"{c}_rate_daytime" for c in TARGET_CATEGORIES]
    exposure_cols = ["daytime_pop", "c000"]
    keep = (["geoid", "within_city", "population"]
            + [c for c in count_cols + rate_cols + daytime_cols + exposure_cols
               if c in target.columns])

    df = feats.merge(target[keep], on="geoid", how="inner")
    if inside_city_only:
        df = df[df["within_city"] == True].copy()

    # explicit per-column imputation of RAW predictors (incl. raw transit/property transform
    # inputs). Restricted to columns actually present so the spatial-lag columns stay dormant
    # until the BQ ingestion adds them (config-over-hardcoding; mirrors the file-backed skip).
    zero_fill = [c for c in ZERO_FILL if c in df.columns]
    median_fill = [c for c in MEDIAN_FILL if c in df.columns]
    df[zero_fill]   = df[zero_fill].fillna(0)
    df[median_fill] = df[median_fill].fillna(df[median_fill].median())

    # derive transit model-form predictors from the imputed raw columns: hurdle form
    # (transit_has_transit + service_intensity log1p centered on the served mass) plus
    # log1p distance/supply. These are the transit entries of PREDICTOR_COLS.
    # See docs/features/transit_stats.md.
    df, _ = apply_transforms(df, hurdle=True)
    # property model forms: log1p the distress shares + their spatial lags (no has_transit
    # indicator / hurdle — those are transit-only). Missing lag inputs are skipped.
    df, _ = apply_transforms(df, spec=PROPERTY_MODEL_TRANSFORMS, has_transit_from=None)
    # demographic model form: log1p the right-skewed 5-mile population ring count into
    # pop_est_5mile_log (ADR 0006) — raw scale destabilises the log1p-target fit.
    df, _ = apply_transforms(df, spec=DEMOGRAPHIC_MODEL_TRANSFORMS, has_transit_from=None)

    # log(count + 1) targets (primary); *_rate kept as validators
    for c in count_cols:
        if c in df.columns:
            df[c.replace("_count", "_logcount")] = np.log1p(df[c])

    # single modeling target: log of the weighted total rate (ADR 0003/0005). The weighted
    # rate is a population-denominator construct (national *_pt_u benchmarks are per-resident),
    # so it is built from the plain population *_rate columns, not the daytime variant.
    df = compute_weighted_scores(df)                     # adds wtotal_rel/rate, wprop_rel/rate
    df["wtotal_lograte"] = np.log1p(df["wtotal_rate"])
    df["wprop_lograte"]  = np.log1p(df["wprop_rate"])

    out = PROCESSED_DIR / "regression_modelling" / f"{city}_model_table.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    predictors = [c for c in PREDICTOR_COLS if c in df.columns]
    missing = [c for c in PREDICTOR_COLS if c not in df.columns]
    if missing:
        print(f"  note: {len(missing)} predictor(s) not yet available (skipped): {missing}")
    df[predictors] = df[predictors].apply(pd.to_numeric, errors="coerce").astype("float64")
    df.to_parquet(out)
    print(f"{city}: model table {df.shape} → {out.relative_to(PROCESSED_DIR.parent.parent)}")
    return df

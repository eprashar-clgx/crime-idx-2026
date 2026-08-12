import pandas as pd
from crime_blockgroup_mapping.config import INTERIM_DIR
from crime_blockgroup_mapping.constants import GCS_ROOT, UCR_YEAR
from regression_modelling.config import source_parquet
from regression_modelling.constants import FEATURE_SOURCES
from regression_modelling.data_wrangling.sources import get_gcs_fs, read_sav_from_gcs, pull_source

ACS_COLS = ["det_pct", "in_household_pct", "moved1yr_pct"]

def wavg(df, group, weight, values):
    """Weighted average of `values` by `group`, weighted by `weight`."""
    agg = df.copy()
    add = pd.DataFrame(columns=[group]).set_index(group)
    for i in values:
        agg[i] = agg[i] * agg[weight]
        add = add.join(agg[agg[i].notna()].groupby(group)[[i, weight]].sum(), how="outer")
        add[i] = add[i] / add[weight]
        add.drop(weight, axis=1, inplace=True)
    return add

def _load_acs(fs, year):
    bg_acs, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/acs/5/{year-1}/bg_acs.sav", fs)
    return bg_acs[["bg_key"] + ACS_COLS].copy()

def _add_division(bg_df, fs):
    div, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/census/2020/county_ct_msa.sav", fs)
    div = div[["STATE", "Division"]].drop_duplicates()
    bg_df["state_key"] = bg_df["bg_key"].str[:2]
    bg_df = bg_df.join(div.set_index("STATE"), on="state_key")
    bg_df["Division"] = bg_df["Division"].fillna(0)
    return bg_df

def _add_distance_to_center(bg_df, fs):
    poi, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/proximity/bg_poi_walk_drive_counts.sav", fs)
    poi = poi[["bg_key", "city_centers_dist"]]
    cross, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/census/2020/bg_bg_crosswalk.sav", fs)
    cross = cross.join(poi.set_index("bg_key"), on="bg10_key")            # 2010 → 2020
    dist = wavg(cross, "bg_key", "pop2020_pct", ["city_centers_dist"])
    return bg_df.join(dist, on="bg_key")

def _add_pop_rings(bg_df, fs, year):
    pop_rings, _ = read_sav_from_gcs(f"{GCS_ROOT}/demographic/population_estimates/{year}/pop_rings.sav", fs)
    pop_rings = pop_rings[["bg_key", "pop_est_5mile", "pop_ch_1mile"]].set_index("bg_key")
    return bg_df.join(pop_rings, on="bg_key")

def build_demographic_features(year: int = UCR_YEAR, refresh: bool = False) -> pd.DataFrame:
    path = source_parquet("demographic")
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    fs = get_gcs_fs()
    bg = _load_acs(fs, year)
    bg = _add_division(bg, fs)
    bg = _add_distance_to_center(bg, fs)
    bg = _add_pop_rings(bg, fs, year)
    bg = bg.rename(columns={"bg_key": "geoid"}).drop(columns=["state_key"])   # normalize key
    path.parent.mkdir(parents=True, exist_ok=True)
    bg.to_parquet(path)
    return bg

def assemble_features(year: int = UCR_YEAR, refresh: bool = False) -> pd.DataFrame:
    """National BG feature matrix: demographic spine ⋈ all registry sources, keyed by geoid."""
    feats = build_demographic_features(year, refresh=refresh)      # spine (~242k BGs)
    for src in FEATURE_SOURCES.values():                            # vacancy, liens, ...
        df = pull_source(src, refresh=refresh).rename(columns={src.key_col: "geoid"})
        feats = feats.merge(df[["geoid", *src.feature_cols]], on="geoid", how="left")
    out = INTERIM_DIR / "features" / "bg_predictors.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out)
    return feats
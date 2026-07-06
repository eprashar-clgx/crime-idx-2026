import pandas as pd
import geopandas as gpd
from src.core.config import CityConfig, DATA_DIR, PLACES_PATH
import pyreadstat
from src.core.config import MODEL_SAV, LODES_PATH

def load_city_boundary(cfg: CityConfig) -> gpd.GeoDataFrame:
    """Load city polygon from US Census places shapefile."""
    places = gpd.read_file(PLACES_PATH)
    city = places[(places['STATEFP'] == cfg.state_fips) & (places['PLACEFP'] == cfg.place_fips)].copy()
    city = city.to_crs("EPSG:4326")
    print(f"{cfg.name} city boundary loaded: {len(city)} row(s)")
    return city

def load_state_block_groups(cfg: CityConfig) -> gpd.GeoDataFrame:
    """Load block groups for a state and standardize columns."""
    bg = gpd.read_file(cfg.bg_zip).to_crs("EPSG:4326")
    bg['county_fips'] = bg['STATEFP'] + bg['COUNTYFP']
    bg.columns = bg.columns.str.lower().str.replace(' ', '_')
    print(f"{cfg.name} state block groups loaded: {len(bg):,}")
    return bg

def label_bgs_within_city(bg_gdf: gpd.GeoDataFrame, city_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Label block groups as within_city=True/False based on centroid."""
    bg = bg_gdf.copy()
    city_union = city_gdf.union_all()
    bg['centroid'] = bg.geometry.centroid
    bg['within_city'] = bg['centroid'].within(city_union).values.astype(bool)
    bg = bg.drop(columns=['centroid'])
    n_in = bg['within_city'].sum()
    print(f"Block groups within city: {n_in:,} / {len(bg):,}")
    return bg

def load_crime_data(cfg: CityConfig, csv_path: str = None) -> gpd.GeoDataFrame:
    """Load crime CSV, drop missing coords, return GeoDataFrame."""
    path = DATA_DIR / (csv_path or cfg.crime_csv)
    df = pd.read_csv(path, on_bad_lines='skip', engine='python')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Normalize variations like 'maplatitude' → 'map_latitude'
    df = df.rename(columns={
        'maplatitude': 'map_latitude',
        'maplongitude': 'map_longitude',
        'nibrsclass': 'nibrs_class',
        'nibrsdescription': 'nibrs_description',
        'rmsoccurrencedate': 'occurrence_date',
        'rmsoccurrencehour': 'occurrence_hour',
        'offensecount': 'offense_count',
        'streetno': 'street_number',
        'streetname': 'street_name',
        'streettype': 'street_type',
        'zipcode': 'zip_code'
        })
    valid = df.dropna(subset=[cfg.lat_col, cfg.lon_col])
    gdf = gpd.GeoDataFrame(
        valid,
        geometry=gpd.points_from_xy(valid[cfg.lon_col], valid[cfg.lat_col]),
        crs="EPSG:4326",
    )
    print(f"{cfg.name} crime data: {len(gdf):,} rows with valid coords (of {len(df):,})")
    return gdf

def sjoin_crimes_to_city(crime_gdf: gpd.GeoDataFrame, city_gdf: gpd.GeoDataFrame):
    """Spatial join crimes to city boundary. Returns (inside_gdf, outside_gdf)."""
    joined = gpd.sjoin(crime_gdf, city_gdf[['geometry']], how='left', predicate='within')
    inside = joined[joined['index_right'].notna()].drop(columns=['index_right'])
    outside = joined[joined['index_right'].isna()].drop(columns=['index_right'])
    print(f"Inside city: {len(inside):,} | Outside: {len(outside):,}")
    return inside, outside

def sjoin_crimes_to_bgs(crime_gdf: gpd.GeoDataFrame, bg_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join crimes to block groups. Returns joined GDF with bg_key + within_city."""
    bg = bg_gdf.copy()
    bg['bg_geo'] = bg.geometry
    cols = ['county_fips', 'geoid', 'bg_geo', 'within_city', 'geometry']
    cols = [c for c in cols if c in bg.columns]
    joined = gpd.sjoin(crime_gdf, bg[cols], how='left', predicate='within')
    joined = joined.rename(columns={'geoid': 'bg_key', 'geometry': 'crime_geo'})
    if 'index_right' in joined.columns:
        joined = joined.drop(columns=['index_right'])
    matched = joined['bg_key'].notna().sum()
    print(f"Matched to BG: {matched:,} | Unmatched: {len(joined) - matched:,}")
    return joined

def aggregate_by_bg(crime_bg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate crime counts per block group."""
    agg = crime_bg.groupby(['bg_key', 'bg_geo']).agg(
        crime_count=('bg_key', 'size'),
        within_city=('within_city', 'first'),
    ).reset_index()
    agg['within_city'] = agg['within_city'].astype(bool)
    agg['county_fips'] = agg['bg_key'].str[:5]
    agg = gpd.GeoDataFrame(agg, geometry='bg_geo', crs="EPSG:4326")
    n_in = agg['within_city'].sum()
    print(f"BGs with crimes: {len(agg):,} ({n_in:,} within city, {len(agg)-n_in:,} outside)")
    return agg

def map_crime_categories(crime_bg: gpd.GeoDataFrame, cfg: CityConfig) -> gpd.GeoDataFrame:
    """Map crime type codes to standardized categories using city-specific mapping."""
    df = crime_bg.copy()
    df['crime_category'] = df[cfg.crime_type_col].map(cfg.crime_type_mapping)
    mapped = df['crime_category'].notna().sum()
    total = len(df)
    print(f"Mapped: {mapped:,} / {total:,} ({mapped/total*100:.1f}%)")
    print("Category counts:")
    print(df['crime_category'].value_counts().to_string())
    unmapped = df.loc[df['crime_category'].isna(), cfg.crime_type_col].value_counts()
    print(f"\nUnmapped: {unmapped.sum():,} records across {len(unmapped)} codes")
    return df

def aggregate_by_bg_category(crime_bg: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate crime counts per block group, pivoted by crime_category.
    
    Returns DataFrame with columns: bg_key, bg_geo, within_city, county_fips,
    total_count, plus one count column per crime category (assault_count, etc.),
    and composite columns (violent_count, property_count).
    """
    df = crime_bg.dropna(subset=['bg_key']).copy()
    
    # Total count per BG
    bg_total = df.groupby(['bg_key', 'bg_geo']).agg(
        total_count=('bg_key', 'size'),
        within_city=('within_city', 'first'),
    ).reset_index()
    
    # Per-category counts (only for mapped crimes)
    cat_df = df.dropna(subset=['crime_category'])
    if len(cat_df) > 0:
        cat_counts = (cat_df.groupby(['bg_key', 'crime_category'])
                      .size().reset_index(name='count'))
        cat_wide = cat_counts.pivot(index='bg_key', columns='crime_category', values='count').fillna(0)
        cat_wide.columns = [f'{c}_count' for c in cat_wide.columns]
        cat_wide = cat_wide.reset_index()
        
        # Merge
        result = bg_total.merge(cat_wide, on='bg_key', how='left')
    else:
        result = bg_total
    
    # Fill NaN category counts with 0
    count_cols = [c for c in result.columns if c.endswith('_count') and c != 'total_count']
    result[count_cols] = result[count_cols].fillna(0)
    
    # Composites
    for col in ['assault_count', 'murder_count', 'rape_count', 'robbery_count',
                'burglary_count', 'larceny_count', 'mvt_count', 'vandal_count']:
        if col not in result.columns:
            result[col] = 0
    
    result['violent_count'] = (result['assault_count'] + result['murder_count']
                               + result['rape_count'] + result['robbery_count'])
    result['property_count'] = (result['burglary_count'] + result['larceny_count']
                                + result['mvt_count']) #+ result['vandal_count']) # Vandalism isn't included
    result['cl_total_count'] = (result['violent_count'] + result['property_count'])
    
    result['within_city'] = result['within_city'].astype(bool)
    result['county_fips'] = result['bg_key'].str[:5]
    
    n_in = result['within_city'].sum()
    print(f"BG-level category aggregation: {len(result):,} BGs "
          f"({n_in:,} within city, {len(result)-n_in:,} outside)")
    print("="*80)
    print("Crime Totals:")
    print(result[['assault_count', 'murder_count', 'rape_count', 'robbery_count',
                'burglary_count', 'larceny_count', 'mvt_count', 'vandal_count',
                'violent_count','property_count','cl_total_count']].sum())
    result = gpd.GeoDataFrame(result, geometry='bg_geo', crs="EPSG:4326")
    return result

def merge_all_bgs_with_crimes(bg_gdf, bg_cat_df):
    """Build analysis set: all BGs inside city (incl. zeros) + only outside BGs with crime data."""
    bg_all = bg_gdf[['geoid', 'within_city', 'geometry']].copy()
    drop_cols = [c for c in ['bg_geo', 'within_city', 'county_fips'] if c in bg_cat_df.columns]
    bg_all = bg_all.merge(bg_cat_df.drop(columns=drop_cols), left_on='geoid', right_on='bg_key', how='left')
    count_cols = [c for c in bg_all.columns if c.endswith('_count')]
    bg_all[count_cols] = bg_all[count_cols].fillna(0)

    # Keep: all inside-city BGs + only outside BGs with crime data
    inside = bg_all[bg_all['within_city'] == True]
    outside = bg_all[(bg_all['within_city'] == False) & (bg_all['total_count'] > 0)]
    result = pd.concat([inside, outside], ignore_index=True)

    n_in = len(inside)
    n_in_data = (inside['total_count'] > 0).sum()
    n_out = len(outside)

    print(f"Analysis set: {len(result):,} BGs")
    print(f"  Inside city:  {n_in:,} ({n_in_data:,} with data, {n_in - n_in_data:,} without)")
    print(f"  Outside city: {n_out:,} (all with crime data)")
    return result

def load_model_data(bg_all):
    """Load crime risk model .sav, filter to BGs in our analysis set."""
    crisk_df, meta = pyreadstat.read_sav(str(MODEL_SAV))
    # Filter to our universe
    crisk_filtered = crisk_df[crisk_df['bg_key'].isin(bg_all['geoid'])].copy()
    n = len(crisk_filtered)
    n_total = len(crisk_df)
    print(f"Model loaded: {n_total:,} total BGs, {n:,} matched to analysis set")
    return crisk_filtered

def merge_model_with_actuals(bg_all, crisk_df):
    """Merge model predictions with actual crime data from bg_all.
    Returns a DataFrame with both _pt_ct (model) and _count (actuals) columns + within_city."""
    model_cols = ['bg_key', 'population'] + [c for c in crisk_df.columns if c.endswith('_pt_ct')]
    merged = bg_all.merge(crisk_df[model_cols], left_on='geoid', right_on='bg_key', how='inner',
                          suffixes=('', '_model'))
    # Drop duplicate bg_key if created
    if 'bg_key_model' in merged.columns:
        merged = merged.drop(columns=['bg_key_model'])
    elif 'bg_key' in merged.columns and 'geoid' in merged.columns:
        pass  # keep both for reference
    
    print(f"Merged comparison set: {len(merged):,} BGs")
    n_in = (merged['within_city'] == True).sum()
    print(f"  Inside city: {n_in:,} | Outside: {len(merged) - n_in:,}")
    return merged

def normalize_actuals(comparison_df, lodes_bg=None):
    """Add _rate columns: actual counts normalized to per 1,000 population.
    Skips BGs with zero population."""
    df = comparison_df.copy()
    count_cols = [c for c in df.columns if c.endswith('_count')]
    # Rate per 1K population
    for col in count_cols:
        rate_col = col.replace('_count', '_rate')
        df[rate_col] = (df[col] / df['population']) * 1000

    # Daytime adjusted: rate per 1K (population + jobs)
    if lodes_bg is not None:
        df = df.merge(lodes_bg, on='geoid', how='left')
        df['c000'] = df['c000'].fillna(0)
        df['daytime_pop'] = df['population'] + df['c000']
        for col in count_cols:
            rate_col = col.replace('_count', '_rate_daytime')
            df[rate_col] = (df[col] / df['daytime_pop']) * 1000
    
    # Replace inf/NaN from zero-population BGs
    rate_cols = [c for c in df.columns if '_rate' in c]
    df[rate_cols] = df[rate_cols].replace([float('inf'), -float('inf')], float('nan'))
    n_valid = df[[c for c in df.columns if c.endswith('_rate')]].iloc[:,0].notna().sum()
    print(f"Normalized actuals to rate/1K: {n_valid:,} BGs with population > 0")
    if lodes_bg is not None:
        n_daytime = df['daytime_pop'].gt(0).sum()
        print(f"Daytime-adjusted rates: {n_daytime:,} BGs with daytime_pop > 0")
    return df

def load_lodes_bg(path=None):
    """
    Load LODES data, aggregate jobs (C000) to block group level
    """
    path = path or LODES_PATH
    lodes = pd.read_csv(path)
    # block keys are missing leading zeroes so add that
    lodes['block_key'] = lodes['block_key'].astype(str).str.zfill(15)
    lodes['geoid'] = lodes['block_key'].str[:12]  # first 12 digits represent the block group
    bg_jobs = lodes.groupby('geoid')['C000'].sum().reset_index()
    bg_jobs.rename(columns={'C000':'c000'}, inplace=True)
    print(f"LODES loaded: {len(lodes):,} blocks -> {len(bg_jobs):,} block groups")
    return bg_jobs

def compute_weighted_scores(comparison_df, national_rates):
    """
    Compute equal-representation weighted crime scores.
    Each of the 7 primary crime types contributes equally (1/7) to total.
    Each of the 3 property crimes contributes equally (1/3) to property.

    Parameters:
    -----------
    comparison_df: DataFrame with *_rate_columns (per 1K population)
    national_rates: dict with keys like 'murder_pt_u', 'assault_pt_u', etc.

    Returns
    ----------
    Dataframe with wtotal_rel, wprop_rel, wtotal_rate, wprop_rate columns added.
    """
    df = comparison_df.copy()
    primary_crimes = ['murder', 'rape', 'robbery', 'assault', 'burglary', 'larceny', 'mvt']
    property_crimes = ['burglary', 'larceny', 'mvt']

    # Compute relative risk for each crime type
    for crime in primary_crimes:
        rate_col = f'{crime}_rate'
        nat_key = f'{crime}_pt_u'
        if rate_col in df.columns and nat_key in national_rates:
            df[f'{crime}_rel'] = df[rate_col] / national_rates[nat_key]
    
    # Equal weight total (1/7 each)
    rel_cols_total = [f'{c}_rel' for c in primary_crimes if f'{c}_rel' in df.columns]
    df['wtotal_rel'] = df[rel_cols_total].mean(axis=1)

    # Equal-weight property (1/3 each)
    rel_cols_prop = [f'{c}_rel' for c in property_crimes if f'{c}_rel' in df.columns]
    df['wprop_rel'] = df[rel_cols_prop].mean(axis=1)
    

    # Scale back to interpretable rate (per 1K)
    nat_total = national_rates.get('violent_pt_u',0) + national_rates.get('property_pt_u',0)
    nat_prop = national_rates.get('property_pt_u',0)
    df['wtotal_rate'] = df['wtotal_rel'] * nat_total
    df['wprop_rate'] = df['wprop_rel'] * nat_prop

    # Clean up intermediate columns
    df = df.drop(columns=[f'{c}_rel' for c in primary_crimes if f'{c}_rel' in df.columns])
    print(f"Weighted scores computed: wtotal_rel median = {df['wtotal_rel'].median():.2f}, "
          f"wprop_rel median={df['wprop_rel'].median():.2f}")
    return df

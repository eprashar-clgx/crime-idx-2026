"""Load crime incidents, spatial-join to block groups, map categories, aggregate to BG counts."""
import pandas as pd
import geopandas as gpd

from crime_blockgroup_mapping.config import DATA_DIR
from crime_blockgroup_mapping.constants import CityConfig


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

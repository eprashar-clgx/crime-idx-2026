"""Load city boundary + state block groups, and label block groups within a city."""
import geopandas as gpd

from crime_blockgroup_mapping.config import PLACES_PATH
from crime_blockgroup_mapping.constants import CityConfig


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

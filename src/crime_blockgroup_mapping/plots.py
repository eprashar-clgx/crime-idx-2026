"""General, reusable map/plot helpers shared by both tasks (BG maps + choropleths)."""
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import HeatMap
import seaborn as sns
from pathlib import Path


def plot_city_with_bgs(bg_gdf, city_gdf, title="Block Groups by City Boundary"):
    """Two-color BG map with city boundary overlay, zoomed to city."""
    fig, ax = plt.subplots(figsize=(14, 12))
    inside = bg_gdf[bg_gdf['within_city'] == True].to_crs(epsg=3857)
    outside = bg_gdf[bg_gdf['within_city'] == False].to_crs(epsg=3857)
    outside.plot(ax=ax, color='white', edgecolor='orange', linewidth=0.5, alpha=0.5)
    inside.plot(ax=ax, color='lightblue', edgecolor='blue', linewidth=0.5, alpha=0.5)
    city_3857 = city_gdf.to_crs(epsg=3857)
    city_3857.boundary.plot(ax=ax, color='black', linewidth=2)
    minx, miny, maxx, maxy = city_3857.total_bounds
    dx, dy = (maxx - minx) * 0.2, (maxy - miny) * 0.2
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_crimes_vs_city(inside_gdf, outside_gdf, city_gdf, title="Crimes vs City Boundary"):
    """Scatter of crime points colored by inside/outside city."""
    fig, ax = plt.subplots(figsize=(14, 12))
    city_3857 = city_gdf.to_crs(epsg=3857)
    city_3857.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=2, alpha=0.3)
    outside_gdf.to_crs(epsg=3857).plot(ax=ax, markersize=20, alpha=0.3, color='blue', label='Outside')
    inside_gdf.to_crs(epsg=3857).plot(ax=ax, markersize=0.3, alpha=0.3, color='yellow', label='Inside')
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_title(title)
    ax.legend(markerscale=10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def plot_bg_crimes(bg_agg_gdf, city_gdf, title="Block Groups with Crimes", crime_bg=None):
    """BG choropleth colored by within/outside city, with city boundary and optional crime dots."""
    fig, ax = plt.subplots(figsize=(14, 12))
    inside = bg_agg_gdf[bg_agg_gdf['within_city'] == True].to_crs(epsg=3857)
    outside = bg_agg_gdf[bg_agg_gdf['within_city'] == False].to_crs(epsg=3857)
    outside.plot(ax=ax, color='orange', edgecolor='orange', linewidth=0.5, alpha=0.5)
    inside.plot(ax=ax, color='lightblue', edgecolor='blue', linewidth=0.5, alpha=0.5)
    if crime_bg is not None:
        crime_pts = gpd.GeoDataFrame(crime_bg, geometry='crime_geo', crs="EPSG:4326")
        pts_in = crime_pts[crime_pts['within_city'] == True].to_crs(epsg=3857)
        pts_out = crime_pts[crime_pts['within_city'] == False].to_crs(epsg=3857)
        pts_out.plot(ax=ax, markersize=20, alpha=0.3, color='blue', label='Outside')
        pts_in.plot(ax=ax, markersize=0.3, alpha=0.3, color='yellow', label='Inside')
    city_gdf.to_crs(epsg=3857).boundary.plot(ax=ax, color='black', linewidth=2)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_title(title)
    if crime_bg is not None:
        ax.legend(markerscale=10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def folium_bg_crimes(bg_agg_gdf, city_gdf, crime_bg, cfg, output_dir):
    """Folium interactive map: BG polygons + city boundary + crime heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centroid = city_gdf.union_all().centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=10, tiles='CartoDB positron')

    # BG polygons colored by within/outside
    inside = bg_agg_gdf[bg_agg_gdf['within_city'] == True]
    outside = bg_agg_gdf[bg_agg_gdf['within_city'] == False]

    folium.GeoJson(
        inside,
        style_function=lambda f: {'fillColor': 'lightblue', 'color': 'blue', 'weight': 0.5, 'fillOpacity': 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=['within_city','bg_key', 'total_count','assault_count','property_count'], aliases=['Location:','Block Group:', 'Crime Count:', 'Assault Count:', 'Property Count:'], sticky=True
        ),
        name='BGs (within city)',
    ).add_to(m)

    folium.GeoJson(
        outside,
        style_function=lambda f: {'fillColor': 'orange', 'color': 'orange', 'weight': 1, 'fillOpacity': 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=['within_city','bg_key', 'total_count','assault_count','property_count'], aliases=['Location:','Block Group:', 'Crime Count:', 'Assault Count:', 'Property Count:'], sticky=True
        ),
        name='BGs (outside city)',
    ).add_to(m)

    # City boundary
    folium.GeoJson(
        city_gdf.boundary,
        style_function=lambda f: {'color': 'black', 'weight': 1},
        name='City Boundary',
    ).add_to(m)

    # Crime heatmap
    crime_pts = gpd.GeoDataFrame(crime_bg, geometry='crime_geo', crs="EPSG:4326")
    heat_data = [[pt.y, pt.x] for pt in crime_pts.geometry if pt is not None]
    HeatMap(heat_data, radius=2, blur=2, max_zoom=13, name='Crime Heatmap').add_to(m)

    folium.LayerControl().add_to(m)

    out_path = output_dir / f"{cfg.name.lower()}_bg_crimes.html"
    m.save(str(out_path))
    print(f"Saved {out_path}")
    return m

def plot_bg_choropleth(bg_gdf, city_gdf, crime_type='cl_total', block_groups='all',
                       cmap='YlOrRd', title=None, vmax=None):
    """Choropleth map of block groups colored by crime count.
    
    Args:
        vmax: If int 1-100, treated as percentile. If float > 100, used as raw value.
              If None, uses data max.
    """
    col = f'{crime_type}_count'
    
    if block_groups == 'inside':
        df = bg_gdf[bg_gdf['within_city'] == True]
    elif block_groups == 'outside':
        df = bg_gdf[bg_gdf['within_city'] == False]
    else:
        df = bg_gdf

    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {[c for c in df.columns if c.endswith('_count')]}")
        return

    gdf = gpd.GeoDataFrame(df, geometry=df.geometry.name, crs=df.crs).to_crs(epsg=3857)

    # Determine vmax
    if vmax is not None and vmax <= 100:
        vmax_val = df[col].quantile(vmax / 100)
    elif vmax is not None:
        vmax_val = vmax
    else:
        vmax_val = df[col].max()

    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"{crime_type.title()} Crime Count — {scope_label} ({len(df):,} BGs, scale capped at {vmax_val:.0f})"

    fig, ax = plt.subplots(figsize=(14, 12))
    gdf.plot(ax=ax, column=col, cmap=cmap, edgecolor='grey', linewidth=0.2,
             legend=True, vmin=0, vmax=vmax_val,
             legend_kwds={'shrink': 0.7, 'label': f'{crime_type.title()} Count'})
    city_gdf.to_crs(epsg=3857).boundary.plot(ax=ax, color='black', linewidth=2)
    minx, miny, maxx, maxy = gdf.total_bounds
    dx, dy = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    ax.set_xlim(minx - dx, maxx + dx)
    ax.set_ylim(miny - dy, maxy + dy)
    cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

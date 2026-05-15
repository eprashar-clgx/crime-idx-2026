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

def plot_crime_distribution(bg_cat_df, crime_type='total', block_groups='all', title=None):
    """Plot crime distribution histogram + percentiles for a given crime type and BG scope.
    
    Args:
        bg_cat_df: DataFrame from aggregate_by_bg_category()
        crime_type: 'total', 'assault', 'burglary', 'violent', 'property', etc.
        block_groups: 'all', 'inside', or 'outside'
        title: optional custom title
    """
    # Filter by scope
    if block_groups == 'inside':
        df = bg_cat_df[bg_cat_df['within_city'] == True]
    elif block_groups == 'outside':
        df = bg_cat_df[bg_cat_df['within_city'] == False]
    else:
        df = bg_cat_df

    col = f'{crime_type}_count' if crime_type != 'total' else 'total_count'
    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {[c for c in df.columns if c.endswith('_count')]}")
        return

    values = df[col]
    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"{crime_type.title()} Crime Distribution — {scope_label} ({len(df):,} BGs)"

    print(f"\n{title}")
    print("=" * 60)
    print(values.describe().round(3).to_string())
    n_zero = (values == 0).sum()
    print(f"Block groups with zero {crime_type}: {n_zero:,} / {len(df):,} ({n_zero/len(df)*100:.1f}%)")
    print("-" * 60)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = values.quantile(p / 100)
        count_ge = (values >= val).sum()
        print(f"  P{p:>2}: {val:>8.0f}   ({count_ge:,} BGs >= this)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    for p, color in [(95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[0].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {val:.0f}')
    axes[0].legend()
    axes[0].set_title('Distribution')
    axes[0].set_xlabel(f'{crime_type.title()} Count')
    axes[0].set_ylabel('Number of Block Groups')

    axes[1].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    for p, color in [(5, 'blue'), (95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[1].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {val:.0f}')
    axes[1].legend()
    axes[1].set_title('Log Scale')
    axes[1].set_xlabel(f'{crime_type.title()} Count')
    axes[1].set_ylabel('Block Groups (log)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_model_distribution(comparison_df, crime_type='total', block_groups='all', title=None):
    """Plot model score (_pt_ct) distribution + percentiles, same format as plot_crime_distribution."""
    if block_groups == 'inside':
        df = comparison_df[comparison_df['within_city'] == True]
    elif block_groups == 'outside':
        df = comparison_df[comparison_df['within_city'] == False]
    else:
        df = comparison_df

    col = f'{crime_type}_pt_ct'
    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {[c for c in df.columns if c.endswith('_pt_ct')]}")
        return

    values = df[col]
    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"Model {crime_type.title()} Score Distribution — {scope_label} ({len(df):,} BGs)"

    print(f"\n{title}")
    print("=" * 60)
    print(values.describe().round(3).to_string())
    n_zero = (values == 0).sum()
    print(f"Block groups with zero {crime_type} score: {n_zero:,} / {len(df):,} ({n_zero/len(df)*100:.1f}%)")
    print("-" * 60)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = values.quantile(p / 100)
        count_ge = (values >= val).sum()
        print(f"  P{p:>2}: {val:>10.3f}   ({count_ge:,} BGs >= this)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    for p, color in [(95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[0].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {val:.1f}')
    axes[0].legend()
    axes[0].set_title('Distribution')
    axes[0].set_xlabel(f'{crime_type.title()} Model Score (per 1K pop)')
    axes[0].set_ylabel('Number of Block Groups')

    axes[1].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    for p, color in [(5, 'blue'), (95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[1].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {val:.1f}')
    axes[1].legend()
    axes[1].set_title('Log Scale')
    axes[1].set_xlabel(f'{crime_type.title()} Model Score (per 1K pop)')
    axes[1].set_ylabel('Block Groups (log)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(comparison_df, block_groups='all', title=None):
    """Spearman correlation heatmap: model _pt_ct vs actual _count for all crime categories."""
    from config import CRIME_CATEGORIES
    
    if block_groups == 'inside':
        df = comparison_df[comparison_df['within_city'] == True]
    elif block_groups == 'outside':
        df = comparison_df[comparison_df['within_city'] == False]
    else:
        df = comparison_df

    # Build pairs
    model_cols = []
    actual_cols = []
    labels = []
    for cat in CRIME_CATEGORIES:
        m_col = f'{cat}_pt_ct'
        a_col = f'{cat}_count'
        if m_col in df.columns and a_col in df.columns:
            model_cols.append(m_col)
            actual_cols.append(a_col)
            labels.append(cat)

    # Compute Spearman correlation matrix between model and actuals
    corr_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, m_col in enumerate(model_cols):
        for j, a_col in enumerate(actual_cols):
            corr_matrix.iloc[i, j] = df[[m_col, a_col]].corr(method='spearman').iloc[0, 1]

    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"Spearman Correlation: Model vs Actuals — {scope_label} ({len(df):,} BGs)"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-0.2, vmax=1, ax=ax,
                xticklabels=[f'{l}_actual' for l in labels],
                yticklabels=[f'{l}_model' for l in labels])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    
    # Print diagonal (same-category correlations)
    print(f"\nSame-category Spearman correlations ({scope_label}):")
    for i, label in enumerate(labels):
        print(f"  {label:>10}: {corr_matrix.iloc[i, i]:.3f}")

def plot_correlation_heatmap_normalized(comparison_df, block_groups='all', title=None):
    """Spearman correlation heatmap: model _pt_ct vs actual _rate (per 1K pop)."""
    from config import CRIME_CATEGORIES
    
    if block_groups == 'inside':
        df = comparison_df[comparison_df['within_city'] == True]
    elif block_groups == 'outside':
        df = comparison_df[comparison_df['within_city'] == False]
    else:
        df = comparison_df

    # Drop zero-population BGs
    df = df[df['population'] > 0]

    model_cols = []
    actual_cols = []
    labels = []
    for cat in CRIME_CATEGORIES:
        m_col = f'{cat}_pt_ct'
        a_col = f'{cat}_rate'
        if m_col in df.columns and a_col in df.columns:
            model_cols.append(m_col)
            actual_cols.append(a_col)
            labels.append(cat)

    corr_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, m_col in enumerate(model_cols):
        for j, a_col in enumerate(actual_cols):
            corr_matrix.iloc[i, j] = df[[m_col, a_col]].corr(method='spearman').iloc[0, 1]

    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"Spearman Correlation: Model vs Actuals (rate/1K) — {scope_label} ({len(df):,} BGs)"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-0.2, vmax=1, ax=ax,
                xticklabels=[f'{l}_rate' for l in labels],
                yticklabels=[f'{l}_model' for l in labels])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    
    print(f"\nSame-category Spearman correlations — normalized ({scope_label}):")
    for i, label in enumerate(labels):
        print(f"  {label:>10}: {corr_matrix.iloc[i, i]:.3f}")


def plot_lorenz_curve(comparison_df, crime_type='total', block_groups='all', title=None):
    """Lorenz curve + Gini coefficient: how well does the model concentrate actual crime risk?
    
    Sorts BGs by model score, then plots cumulative share of actual crime.
    Perfect model → curve hugs top-left. Random → diagonal.
    """
    if block_groups == 'inside':
        df = comparison_df[comparison_df['within_city'] == True]
    elif block_groups == 'outside':
        df = comparison_df[comparison_df['within_city'] == False]
    else:
        df = comparison_df

    model_col = f'{crime_type}_pt_ct'
    actual_col = f'{crime_type}_count'
    
    if model_col not in df.columns or actual_col not in df.columns:
        print(f"Missing columns: need '{model_col}' and '{actual_col}'")
        return

    # Drop rows where either is NaN
    valid = df[[model_col, actual_col]].dropna()
    if len(valid) == 0:
        print("No valid data for Lorenz curve.")
        return

    # Sort by model score descending (highest risk first)
    sorted_df = valid.sort_values(model_col, ascending=False)
    
    cumulative_actual = np.cumsum(sorted_df[actual_col].values)
    total_actual = cumulative_actual[-1]
    
    if total_actual == 0:
        print(f"No actual {crime_type} crimes — can't compute Lorenz curve.")
        return

    cumulative_share = cumulative_actual / total_actual
    population_share = np.arange(1, len(sorted_df) + 1) / len(sorted_df)

    # Gini = 2 * area between Lorenz curve and diagonal
    gini = 2 * (np.trapezoid(cumulative_share, population_share) - 0.5)

    if title is None:
        scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
        title = f"Lorenz Curve: {crime_type.title()} — {scope_label} (Gini = {gini:.3f})"

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(population_share, cumulative_share, 'b-', linewidth=2, label=f'Model (Gini={gini:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (Gini=0)')
    ax.fill_between(population_share, population_share, cumulative_share, alpha=0.15, color='blue')
    ax.set_xlabel('Cumulative Share of Block Groups (sorted by model score, highest first)')
    ax.set_ylabel(f'Cumulative Share of Actual {crime_type.title()} Crime')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    # Lift table at key cutoffs
    print(f"\nGini coefficient: {gini:.3f}")
    print(f"\n{'Top %':>8} {'Crime Captured':>16} {'Lift vs Random':>16}")
    print("-" * 44)
    for pct in [0.05, 0.10, 0.20, 0.30, 0.50]:
        idx = min(int(pct * len(population_share)) - 1, len(cumulative_share) - 1)
        if idx < 0:
            idx = 0
        captured = cumulative_share[idx]
        lift = captured / pct
        print(f"  {pct*100:>5.0f}%   {captured*100:>13.1f}%   {lift:>14.1f}x")

    # Where do we capture 50% and 80% of crime?
    for target in [0.50, 0.80]:
        idx_target = np.searchsorted(cumulative_share, target)
        if idx_target < len(population_share):
            pct_needed = population_share[idx_target] * 100
            print(f"  To capture {target*100:.0f}% of {crime_type}: need top {pct_needed:.1f}% of BGs")

    return gini
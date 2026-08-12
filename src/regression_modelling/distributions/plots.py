"""Distribution / EDA plots for the regression_modelling task (histograms, corr, Lorenz, POI maps)."""
import matplotlib.pyplot as plt
import contextily as cx
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from folium.plugins import HeatMap
import seaborn as sns
from pathlib import Path
from sklearn.metrics import auc as sk_auc


def plot_distribution(df, crime_type='cl_total', suffix='count', block_groups='all', title=None):
    """Plot distribution histogram + percentiles for any crime metric.
    
    Args:
        df: DataFrame (bg_cat, bg_all, or comparison)
        crime_type: 'cl_total', 'assault', 'violent', 'property', etc.
        suffix: 'count', 'pt_ct', 'rate', 'rate_daytime'
        block_groups: 'all', 'inside', or 'outside'
        title: optional custom title
    """
    if block_groups == 'inside':
        df = df[df['within_city'] == True]
    elif block_groups == 'outside':
        df = df[df['within_city'] == False]

    col = f'{crime_type}_{suffix}'
    if col not in df.columns:
        print(f"Column '{col}' not found. Available: {[c for c in df.columns if c.endswith(f'_{suffix}')]}")
        return

    values = df[col]
    scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
    suffix_label = {'count': 'Crime Count', 'pt_ct': 'Model Score (per 1K pop)', 
                    'rate': 'Rate (per 1K pop)', 'rate_daytime': 'Rate (per 1K daytime pop)'}
    x_label = f"{crime_type.title()} {suffix_label.get(suffix, suffix)}"
    
    if title is None:
        title = f"{x_label} Distribution — {scope_label} ({len(df):,} BGs)"

    # Use more decimal places for rates/scores, integers for counts
    is_integer = suffix == 'count'
    fmt_val = lambda v: f"{v:>8.0f}" if is_integer else f"{v:>10.3f}"
    fmt_pline = lambda v: f"{v:.0f}" if is_integer else f"{v:.1f}"

    print(f"\n{title}")
    print("=" * 60)
    print(values.describe().round(3).to_string())
    n_zero = (values == 0).sum()
    print(f"Block groups with zero: {n_zero:,} / {len(df):,} ({n_zero/len(df)*100:.1f}%)")
    print("-" * 60)
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = values.quantile(p / 100)
        count_ge = (values >= val).sum()
        print(f"  P{p:>2}: {fmt_val(val)}   ({count_ge:,} BGs >= this)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    for p, color in [(95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[0].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {fmt_pline(val)}')
    axes[0].legend()
    axes[0].set_title('Distribution')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Number of Block Groups')

    axes[1].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_yscale('log')
    for p, color in [(5, 'blue'), (95, 'orange'), (99, 'red')]:
        val = values.quantile(p / 100)
        axes[1].axvline(val, color=color, linestyle='--', linewidth=2, label=f'P{p} = {fmt_pline(val)}')
    axes[1].legend()
    axes[1].set_title('Log Scale')
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel('Block Groups (log)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(comparison_df, block_groups='all', actual_suffix='count', title=None):
    """Spearman correlation heatmap: model _pt_ct vs actual crime measures.
    
    Args:
        comparison_df: DataFrame with model and actual columns.
        block_groups: 'all', 'inside', or 'outside'.
        actual_suffix: 'count', 'rate', or 'rate_daytime' — determines which actual columns to use.
        title: Optional custom title.
    """
    from crime_blockgroup_mapping.constants import CRIME_CATEGORIES
    
    if block_groups == 'inside':
        df = comparison_df[comparison_df['within_city'] == True]
    elif block_groups == 'outside':
        df = comparison_df[comparison_df['within_city'] == False]
    else:
        df = comparison_df

    # Drop zero-population BGs for rate-based comparisons
    if actual_suffix in ('rate', 'rate_daytime'):
        df = df[df['population'] > 0]

    # Build pairs
    model_cols = []
    actual_cols = []
    labels = []
    for cat in CRIME_CATEGORIES: # e.g. assault, property etc.
        m_col = f'{cat}_pt_ct'
        a_col = f'{cat}_{actual_suffix}'
        if m_col in df.columns and a_col in df.columns:
            model_cols.append(m_col)
            actual_cols.append(a_col)
            labels.append(cat)

    # Compute Spearman correlation matrix
    corr_matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for i, m_col in enumerate(model_cols):
        for j, a_col in enumerate(actual_cols):
            corr_matrix.iloc[i, j] = df[[m_col, a_col]].corr(method='spearman').iloc[0, 1]

    scope_label = {'all': 'All BGs', 'inside': 'Inside City', 'outside': 'Outside City'}[block_groups]
    suffix_label = {'count': 'raw counts', 'rate': 'rate/1K pop', 'rate_daytime': 'rate/1K daytime pop'}
    if title is None:
        title = (f"Spearman Correlation: Model vs Actuals ({suffix_label.get(actual_suffix, actual_suffix)}) "
                 f"— {scope_label} ({len(df):,} BGs)")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                center=0, vmin=-0.2, vmax=1, ax=ax,
                xticklabels=[f'{l}_{actual_suffix}' for l in labels],
                yticklabels=[f'{l}_model' for l in labels])
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    
    print(f"\nSame-category Spearman correlations — {actual_suffix} ({scope_label}):")
    for i, label in enumerate(labels):
        print(f"  {label:>10}: {corr_matrix.iloc[i, i]:.3f}")


def plot_lorenz_curve(comparison_df, crime_type='cl_total', block_groups='all', title=None):
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

    if crime_type == 'cl_total':
        model_col = f'total_pt_ct'
    else:
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

    return (gini.round(2))


def store_points_map(df, geom_col="parcel_polygon_at_eventtime",
                     label_col="business_name", mode="cluster", sample=None):
    """Folium map of store parcels from a store_points DataFrame (BQ GEOGRAPHY as WKT).

    mode: 'cluster' (MarkerCluster of centroids) or 'heat' (HeatMap of centroids).
    sample: optional int to randomly downsample very large national pulls before rendering.
    """
    from shapely import wkt
    from folium.plugins import MarkerCluster

    g = df.dropna(subset=[geom_col]).copy()
    if sample is not None and len(g) > sample:
        g = g.sample(sample, random_state=0)
    geom = g[geom_col].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(g, geometry=geom, crs="EPSG:4326")
    cent = gdf.geometry.representative_point()
    gdf["lat"], gdf["lon"] = cent.y.values, cent.x.values

    m = folium.Map(location=[gdf["lat"].mean(), gdf["lon"].mean()],
                   zoom_start=5, tiles="CartoDB positron")
    if mode == "heat":
        HeatMap(gdf[["lat", "lon"]].to_numpy().tolist(), radius=8).add_to(m)
    else:
        cluster = MarkerCluster().add_to(m)
        for _, r in gdf.iterrows():
            popup = str(r[label_col]) if label_col in gdf.columns else None
            folium.CircleMarker([r["lat"], r["lon"]], radius=3, popup=popup,
                                color="crimson", fill=True, fill_opacity=0.7).add_to(cluster)
    print(f"Mapped {len(gdf):,} store parcels ({mode})")
    return m

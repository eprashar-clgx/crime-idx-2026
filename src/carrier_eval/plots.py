"""Carrier-evaluation plots (Lorenz / concentration curves for carrier outcomes)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc as sk_auc


def plot_carrier_lorenz(df, risk_score_col, outcome_col, exposure_col='earned_exposures',
                        x_axis='exposure_weighted', title=None, label=None, ax=None):
    """Single Lorenz curve for a carrier dataset sorted by a risk score.
    
    Args:
        df: DataFrame with risk score, outcome, and exposure columns.
        risk_score_col: Column to sort by (descending = highest risk first).
        outcome_col: Column with the outcome to accumulate (claims or losses).
        exposure_col: Column with earned exposures (used when x_axis='exposure_weighted').
        x_axis: How to compute the x-axis.
            - 'exposure_weighted': Cumulative % of earned exposures.
              "If I price by crime score, how much of my loss dollar is concentrated
              in the riskiest slice of my premium base?" → relevant for pricing/underwriting.
            - 'equal_weight': Cumulative % of block groups (each BG = 1/N).
              "Does crime score identify geographic areas where claims concentrate?"
              → relevant for validating the crime data itself.
        title: Plot title (used only if ax is None, i.e., standalone plot).
        label: Legend label for this curve.
        ax: Matplotlib axes to plot on. If None, creates a new figure.
    
    Returns:
        gini (float): Gini coefficient for this curve.
    """
    valid = df[[risk_score_col, outcome_col, exposure_col]].dropna()
    valid = valid[valid[exposure_col] > 0]
    
    if len(valid) == 0 or valid[outcome_col].sum() == 0:
        print(f"No valid data for Lorenz curve (score={risk_score_col}, outcome={outcome_col})")
        return None

    # Sort by risk score descending
    sorted_df = valid.sort_values(risk_score_col, ascending=False).reset_index(drop=True)
    
    total_outcome = sorted_df[outcome_col].sum()
    cum_outcome = sorted_df[outcome_col].cumsum() / total_outcome

    if x_axis == 'exposure_weighted':
        total_exposure = sorted_df[exposure_col].sum()
        cum_x = sorted_df[exposure_col].cumsum() / total_exposure
    else:  # equal_weight
        cum_x = np.arange(1, len(sorted_df) + 1) / len(sorted_df)

    # Prepend origin
    cum_x = np.concatenate([[0], cum_x.values if hasattr(cum_x, 'values') else cum_x])
    cum_outcome = np.concatenate([[0], cum_outcome.values])

    # Gini
    gini = 2 * np.trapezoid(cum_outcome, cum_x) - 1

    # Plot
    if label is None:
        label = risk_score_col
    curve_label = f'{label} (Gini={gini:.2f})'

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5,4))

    ax.plot(cum_x, cum_outcome, linewidth=2, label=curve_label)

    if standalone:
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
        ax.set_xlabel('Cumulative % of Exposure' if x_axis == 'exposure_weighted' 
                      else 'Cumulative % of Block Groups')
        ax.set_ylabel(f'Cumulative % of {outcome_col}')
        if title:
            ax.set_title(title)
        ax.legend(loc='center right', bbox_to_anchor=(1.75, 0.5))
        ax.minorticks_on()
        ax.grid(True)
        ax.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    return (gini.round(2))


def plot_carrier_lorenz_multi(df, risk_score_cols, outcome_col, exposure_col='earned_exposures',
                              x_axis='exposure_weighted', labels=None, title=None):
    """Plot multiple Lorenz curves on the same figure, one per risk score.
    
    Args:
        df: DataFrame with all columns needed.
        risk_score_cols: List of columns to use as risk scores (one curve each).
        outcome_col: Column with outcome to accumulate (claims or losses).
        exposure_col: Column with earned exposures.
        x_axis: 'exposure_weighted' (default) or 'equal_weight'.
            - 'exposure_weighted': Cumulative % of earned exposures.
              "If I price by crime score, how much of my loss dollar is concentrated
              in the riskiest slice of my premium base?" → relevant for pricing/underwriting.
            - 'equal_weight': Cumulative % of block groups (each BG = 1/N).
              "Does crime score identify geographic areas where claims concentrate?"
              → relevant for validating the crime data itself.
        labels: Dict mapping risk_score_col → display label. If None, uses column names.
        title: Plot title.
    
    Returns:
        Dict of {risk_score_col: gini_coefficient}.
    """
    if labels is None:
        labels = {col: col for col in risk_score_cols}

    fig, ax = plt.subplots(figsize=(5,4))
    ginis = {}

    for col in risk_score_cols:
        gini = plot_carrier_lorenz(
            df, risk_score_col=col, outcome_col=outcome_col,
            exposure_col=exposure_col, x_axis=x_axis,
            label=labels.get(col, col), ax=ax
        )
        if gini is not None:
            ginis[col] = gini

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', label='Random')
    
    x_label = ('Cumulative % of Exposure by Crime Score' if x_axis == 'exposure_weighted'
               else 'Cumulative % of Block Groups by Crime Score')
    ax.set_xlabel(x_label)
    
    outcome_name = outcome_col.replace('_', ' ').title()
    ax.set_ylabel(f'Cumulative % of {outcome_name}')
    
    if title is None:
        total = df[outcome_col].sum()
        title = f'Lorenz Curve ({outcome_name}: {total:,.0f})'
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize='small', framealpha=0.9)
    ax.minorticks_on()
    ax.grid(True)
    ax.grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.show()
    return ginis


"""Score-reconstruction math: weighted crime scores + national reference rates (group D)."""
import pandas as pd

from carrier_eval.config import EVALS_PATH


def extract_national_rates(path=None):
    """Extract national crime rates (*_pt_u columns) from evals parquet.
    These are constant across all rows - take first non-null value for each."""
    path = path or EVALS_PATH
    df = pd.read_parquet(path, engine='fastparquet')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('#', 'num')
    pt_u_cols = [c for c in df.columns if c.endswith('_pt_u')]
    # Since these values are repeated through the dataframe, we can just pick the first one
    # Saved as a dictionary for easy lookup using keys instead of relying on idx
    national = df[pt_u_cols].dropna().iloc[0].round(3).to_dict()
    print(f"National rates extracted: {national}")
    return national


def compute_weighted_scores(comparison_df, national_rates):
    """Compute equal-representation weighted crime scores.
    Each of the 7 primary crime types contributes equally (1/7) to total.
    Each of the 3 property crimes contributes equally (1/3) to property.

    Parameters
    ----------
    comparison_df: DataFrame with *_rate columns (per 1K population)
    national_rates: dict with keys like 'murder_pt_u', 'assault_pt_u', etc.

    Returns
    -------
    DataFrame with wtotal_rel, wprop_rel, wtotal_rate, wprop_rate columns added.
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
    nat_total = national_rates.get('violent_pt_u', 0) + national_rates.get('property_pt_u', 0)
    nat_prop = national_rates.get('property_pt_u', 0)
    df['wtotal_rate'] = df['wtotal_rel'] * nat_total
    df['wprop_rate'] = df['wprop_rel'] * nat_prop

    # Clean up intermediate columns
    df = df.drop(columns=[f'{c}_rel' for c in primary_crimes if f'{c}_rel' in df.columns])
    print(f"Weighted scores computed: wtotal_rel median = {df['wtotal_rel'].median():.2f}, "
          f"wprop_rel median={df['wprop_rel'].median():.2f}")
    return df

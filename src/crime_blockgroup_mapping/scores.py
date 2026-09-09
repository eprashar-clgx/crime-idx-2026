"""Weighted crime-score / relative-risk math (shared foundation, ADR 0005).

Promoted out of `carrier_eval.scores` so BOTH tasks can use it without a task→task
import (ADR 0001 seam): `carrier_eval` re-exports these, `regression_modelling` builds
the `log(wtotal_rate)` target from them. The math is documented in
`docs/weightage_methodology.md`.

The composite is an *equal-representation average of relative risks* (each local
`{crime}_rate` divided by its national `*_pt_u` benchmark), not a raw sum of rates —
normalizing to national rates first stops high-volume crimes (larceny) from mechanically
dominating. National benchmarks are per-1,000 RESIDENTS, so the weighted rate is a
population-denominator construct.
"""
import pandas as pd

from crime_blockgroup_mapping.constants import NATIONAL_PT_U_RATES

# The 7 primary crimes (equal 1/7 weight in the Overall composite) and the 3 property
# crimes (equal 1/3 weight in the Property composite). Vandalism is intentionally excluded.
PRIMARY_CRIMES = ['murder', 'rape', 'robbery', 'assault', 'burglary', 'larceny', 'mvt']
PROPERTY_CRIMES = ['burglary', 'larceny', 'mvt']


def extract_national_rates(path):
    """Extract national crime rates (`*_pt_u` columns) from an evals parquet.

    These values are constant across all rows, so we take the first non-null. Kept for
    the carrier workflow that has the evals file; regression code should prefer the
    hardcoded `NATIONAL_PT_U_RATES` constant so it needs no carrier artifact.
    """
    df = pd.read_parquet(path, engine='fastparquet')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('#', 'num')
    pt_u_cols = [c for c in df.columns if c.endswith('_pt_u')]
    national = df[pt_u_cols].dropna().iloc[0].round(3).to_dict()
    print(f"National rates extracted: {national}")
    return national


def compute_weighted_scores(comparison_df, national_rates=None):
    """Compute equal-representation weighted crime scores.

    Each of the 7 primary crime types contributes equally (1/7) to the Overall composite;
    each of the 3 property crimes contributes equally (1/3) to the Property composite.

    Parameters
    ----------
    comparison_df : DataFrame with `{crime}_rate` columns (per 1K RESIDENTS).
    national_rates : dict with keys like 'murder_pt_u', 'assault_pt_u', … Defaults to the
        canonical `NATIONAL_PT_U_RATES` foundation constant.

    Returns
    -------
    DataFrame with `wtotal_rel`, `wprop_rel`, `wtotal_rate`, `wprop_rate` columns added.
    """
    national_rates = national_rates or NATIONAL_PT_U_RATES
    df = comparison_df.copy()

    # Relative risk per crime = local rate / national benchmark
    for crime in PRIMARY_CRIMES:
        rate_col = f'{crime}_rate'
        nat_key = f'{crime}_pt_u'
        if rate_col in df.columns and nat_key in national_rates:
            df[f'{crime}_rel'] = df[rate_col] / national_rates[nat_key]

    # Equal-weight composites (1/7 overall, 1/3 property)
    rel_cols_total = [f'{c}_rel' for c in PRIMARY_CRIMES if f'{c}_rel' in df.columns]
    df['wtotal_rel'] = df[rel_cols_total].mean(axis=1)
    rel_cols_prop = [f'{c}_rel' for c in PROPERTY_CRIMES if f'{c}_rel' in df.columns]
    df['wprop_rel'] = df[rel_cols_prop].mean(axis=1)

    # Rescale back to interpretable per-1K units (a single constant multiplier per BG,
    # so it does not change the BG ranking — only restores units).
    nat_total = national_rates.get('violent_pt_u', 0) + national_rates.get('property_pt_u', 0)
    nat_prop = national_rates.get('property_pt_u', 0)
    df['wtotal_rate'] = df['wtotal_rel'] * nat_total
    df['wprop_rate'] = df['wprop_rel'] * nat_prop

    df = df.drop(columns=[f'{c}_rel' for c in PRIMARY_CRIMES if f'{c}_rel' in df.columns])
    print(f"Weighted scores computed: wtotal_rel median = {df['wtotal_rel'].median():.2f}, "
          f"wprop_rel median={df['wprop_rel'].median():.2f}")
    return df

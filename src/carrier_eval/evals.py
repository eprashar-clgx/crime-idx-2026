"""Ingest carrier insurance evals and aggregate them to block-group level."""
import pandas as pd

from carrier_eval.config import EVALS_PATH
from carrier_eval.constants import CLAIMS_COLS, LOSSES_COLS, EXPOSURE_COL


def load_evals(path=None):
    """Load carrier evals parquet, standardize column names."""
    path = path or EVALS_PATH
    df = pd.read_parquet(path, engine='fastparquet')
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('#', 'num')
    print(f"Evals loaded: {len(df):,} rows, {df['carrier'].nunique()} carriers")
    print(f"Carriers: {df['carrier'].value_counts().to_dict()}")
    return df


def extract_bg_geoid(df):
    """Derive 12-digit block group GEOID from crime_censusblock (15-digit block ID)."""
    df = df.copy()
    df['bg_geoid'] = df['crime_censusblock'].astype(str).str[:12]
    print(f"Extracted bg_geoid from crime_censusblock. Sample: {df['bg_geoid'].iloc[:3].tolist()}")
    return df


def validate_block_cols(df):
    """Check that censusblock_name and crime_censusblock refer to the same block group."""
    # censusblock_name is numeric (may lose leading zeros), crime_censusblock is string
    df_check = df[['censusblock_name', 'crime_censusblock']].dropna()
    df_check['cb_from_name'] = df_check['censusblock_name'].astype(int).astype(str).str.zfill(15)
    match_rate = (df_check['cb_from_name'] == df_check['crime_censusblock']).mean()
    print(f"Block column match rate: {match_rate*100:.1f}%")
    return match_rate


def aggregate_evals_to_bg(df, carrier):
    """Aggregate eval data to block group level for a specific carrier.

    Filters to carrier + earned_exposures > 0, then groups by bg_geoid.
    Returns DataFrame with summed claims, losses, and exposure per BG.
    """
    carrier_df = df[(df['carrier'] == carrier) & (df[EXPOSURE_COL] > 0)].copy()
    if len(carrier_df) == 0:
        print(f"No data for carrier '{carrier}' with exposure > 0")
        return pd.DataFrame()

    agg_cols = [EXPOSURE_COL] + CLAIMS_COLS + LOSSES_COLS
    # Only include columns that exist
    agg_cols = [c for c in agg_cols if c in carrier_df.columns]

    bg_agg = carrier_df.groupby('bg_geoid')[agg_cols].sum().reset_index()

    # Add total claims column
    claims_present = [c for c in CLAIMS_COLS if c in bg_agg.columns]
    losses_present = [c for c in LOSSES_COLS if c in bg_agg.columns]
    bg_agg['total_claims'] = bg_agg[claims_present].sum(axis=1)
    bg_agg['total_losses'] = bg_agg[losses_present].sum(axis=1)

    print(f"Carrier '{carrier}': {len(carrier_df):,} rows → {len(bg_agg):,} block groups")
    print(f"  Total exposure: {bg_agg[EXPOSURE_COL].sum():,.0f}")
    print(f"  Total claims: {bg_agg['total_claims'].sum():,.0f}")
    print(f"  Total losses: {bg_agg['total_losses'].sum():,.0f}" if 'total_losses' in bg_agg.columns else "")
    return bg_agg


def merge_evals_with_crime(eval_bg, comparison_df):
    """Inner join carrier BG data with city crime comparison data on geoid.

    Brings crime counts and model scores from the city pipeline into the eval BG frame.
    """
    # comparison_df has 'geoid' column from our pipeline
    merged = eval_bg.merge(comparison_df, left_on='bg_geoid', right_on='geoid', how='inner')

    n_eval = len(eval_bg)
    n_matched = len(merged)
    print(f"Merged: {n_matched:,} of {n_eval:,} eval BGs matched to city crime data ({n_matched/n_eval*100:.1f}%)")
    return merged

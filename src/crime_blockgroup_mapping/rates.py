"""Population source + LODES jobs + per-1,000 rate normalization for the BG crime table."""
import pandas as pd
import pyreadstat

from crime_blockgroup_mapping.config import MODEL_SAV, LODES_PATH


def load_model_data(bg_all):
    """Load crime risk model .sav, filter to BGs in our analysis set.

    NOTE: the existing NeighborhoodScout model artifact is used here only to obtain
    `population` (and existing model score columns for carrier_eval). Flagged in
    docs/adr/0001 as a coupling to revisit.
    """
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


def load_lodes_bg(path=None):
    """Load LODES data, aggregate jobs (C000) to block group level."""
    path = path or LODES_PATH
    lodes = pd.read_csv(path)
    # block keys are missing leading zeroes so add that
    lodes['block_key'] = lodes['block_key'].astype(str).str.zfill(15)
    lodes['geoid'] = lodes['block_key'].str[:12]  # first 12 digits represent the block group
    bg_jobs = lodes.groupby('geoid')['C000'].sum().reset_index()
    bg_jobs.rename(columns={'C000': 'c000'}, inplace=True)
    print(f"LODES loaded: {len(lodes):,} blocks -> {len(bg_jobs):,} block groups")
    return bg_jobs


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
    n_valid = df[[c for c in df.columns if c.endswith('_rate')]].iloc[:, 0].notna().sum()
    print(f"Normalized actuals to rate/1K: {n_valid:,} BGs with population > 0")
    if lodes_bg is not None:
        n_daytime = df['daytime_pop'].gt(0).sum()
        print(f"Daytime-adjusted rates: {n_daytime:,} BGs with daytime_pop > 0")
    return df

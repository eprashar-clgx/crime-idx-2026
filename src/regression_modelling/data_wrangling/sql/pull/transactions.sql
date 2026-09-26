SELECT
  census_block_group_geoid AS geoid,
  clip_transaction_pct,
  clip_transaction_pct_lag6,
  total_unq_clips,
  unq_clip_w_transaction
FROM `{bq_project}.{staging_dataset}.bg_clip_transactions`
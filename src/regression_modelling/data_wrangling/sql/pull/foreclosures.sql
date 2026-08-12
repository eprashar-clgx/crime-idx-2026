SELECT
  census_block_group_geoid AS geoid,
  clip_foreclosure_pct,
  total_unq_clips,
  unq_clip_w_foreclosure
FROM `{bq_project}.{staging_dataset}.bg_clip_foreclosures`
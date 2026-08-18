SELECT
  census_block_group_geoid AS geoid,
  liquor_stores_clip_pct,
  total_unq_clips,
  unq_liquor_stores_clips
FROM `{bq_project}.{staging_dataset}.bg_liquor_stores`

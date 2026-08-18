SELECT
  census_block_group_geoid AS geoid,
  convenience_stores_clip_pct,
  total_unq_clips,
  unq_convenience_stores_clips
FROM `{bq_project}.{staging_dataset}.bg_convenience_stores`

SELECT
  census_block_group_geoid AS geoid,
  liquor_store_clip_pct,
  tot_unq_clips,
  unq_liquor_store_clips
FROM `{bq_project}.{staging_dataset}.bg_liquor_stores`
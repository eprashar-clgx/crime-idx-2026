SELECT
  census_block_group_geoid AS geoid,
  clip_liens_pct,
  total_clips,
  clip_w_liens
FROM `{bq_project}.{staging_dataset}.bg_clip_liens`
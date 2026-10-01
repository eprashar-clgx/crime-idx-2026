SELECT
  census_block_group_geoid AS geoid,
  pawn_shops_clip_pct,
  total_unq_clips,
  unq_pawn_shops_clips
FROM `{bq_project}.{staging_dataset}.bg_pawn_shops`
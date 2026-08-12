SELECT
  census_block_group_geoid AS geoid,
  seven_eleven_clip_pct,
  tot_unq_clips,
  unq_seven_eleven_clips
FROM `{bq_project}.{staging_dataset}.bg_seven_eleven`
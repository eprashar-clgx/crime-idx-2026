SELECT
  census_block_group_geoid AS geoid,
  gas_stations_clip_pct,
  total_unq_clips,
  unq_gas_stations_clips
FROM `{bq_project}.{staging_dataset}.bg_gas_stations`

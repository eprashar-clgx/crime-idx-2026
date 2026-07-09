SELECT
  census_block_group_geoid AS geoid,
  gas_station_clip_pct,
  tot_unq_clips,
  unq_gas_station_clips
FROM `{bq_project}.{staging_dataset}.bg_gas_stations`
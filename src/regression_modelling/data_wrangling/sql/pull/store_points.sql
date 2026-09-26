SELECT
  clip_id,
  geoid,
  lat,
  lon
FROM `{bq_project}.{staging_dataset}.{store}_points`

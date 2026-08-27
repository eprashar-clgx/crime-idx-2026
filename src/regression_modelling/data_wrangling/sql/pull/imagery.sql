SELECT
  geoid,
  roof_condition_avg,
  roof_debris_pct_avg,
  roof_discoloration_pct_avg,
  hardscapes_pct_avg,
  roof_missing_material_pct,
  imagery_structure_count
FROM `{bq_project}.{staging_dataset}.bg_imagery`

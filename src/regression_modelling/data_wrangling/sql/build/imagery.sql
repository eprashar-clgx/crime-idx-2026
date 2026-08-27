-- Block-group Vexcel aerial-feature builder (structure-level -> BG averages).
-- Vexcel elements are per-structure (clip); this rolls them up to block group via the
-- parcel-universe xref (clip_list is pipe-delimited), averaging roof/parcel condition
-- measures per BG. Materializes `bg_imagery` in the staging dataset. Structure-level
-- source re-aggregated to BG geoid via the SAME xref the store builder uses -- NOT a tract
-- broadcast (the source query grouped by tract; this groups by census_block_group_geoid).
--
-- Task params (filled on top of the project/dataset defaults):
--   {imagery_project}  project holding the Vexcel aerial-features view (prd)
--   {fips_filter}      county FIPS scope, quoted CSV, e.g. '17031' or '17031','48201'
CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_imagery` AS
WITH vexcel AS (
  -- per-structure (clip) Vexcel elements pertaining to crime, scoped to the county set
  SELECT clip,
         structure_roof_characteristics_condition,
         structure_roof_defects_missing_material_detected,
         structure_roof_defects_debris_percentage,
         structure_roof_defects_discoloration_percentage,
         parcel_hardscapes_percent
  FROM `{imagery_project}.edr_ent_property_aerial_features.vw_vexcel_all_join_classes_structures`
  WHERE clip IS NOT NULL
    AND clgx_fips IN ({fips_filter})
),
clip_bg AS (
  -- map each clip to its block group via the parcel-universe xref
  SELECT DISTINCT census_block_group_geoid AS geoid, clip
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref`,
       UNNEST(SPLIT(clip_list, "|")) AS clip
  WHERE SUBSTR(census_tract_geoid, 1, 5) IN ({fips_filter})
),
bg_avg AS (
  -- block-group averages of the Vexcel elements
  SELECT a.geoid,
         ROUND(AVG(b.structure_roof_characteristics_condition), 2)        AS roof_condition_avg,
         ROUND(AVG(b.structure_roof_defects_debris_percentage), 2)        AS roof_debris_pct_avg,
         ROUND(AVG(b.structure_roof_defects_discoloration_percentage), 2) AS roof_discoloration_pct_avg,
         ROUND(AVG(b.parcel_hardscapes_percent), 2)                       AS hardscapes_pct_avg,
         ROUND(SAFE_DIVIDE(COUNTIF(b.structure_roof_defects_missing_material_detected),
                           COUNT(b.structure_roof_defects_missing_material_detected)), 2)
                                                                          AS roof_missing_material_pct,
         COUNT(b.clip) AS imagery_structure_count   -- Vexcel-observed structures backing the averages
  FROM clip_bg a
  LEFT JOIN vexcel b ON a.clip = b.clip
  GROUP BY 1
)
SELECT * FROM bg_avg

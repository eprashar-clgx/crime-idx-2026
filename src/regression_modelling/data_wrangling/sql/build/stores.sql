-- Generic block-group store-count builder (one template for every store type).
-- Rolls a firmographics business universe up to block-group clip counts via the
-- parcel-universe xref. Materializes `bg_{store}` in the staging dataset.
--
-- Task params (filled per store on top of the project/dataset defaults):
--   {store}            table/column stem, e.g. convenience_stores | liquor_stores | gas_stations
--   {match_predicate}  firmographics WHERE clause defining the store universe
--                      (NAICS/SIC codes and/or business-name LIKEs)
CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_{store}` AS
WITH matched AS (
  -- clips whose firmographics record matches the store definition
  SELECT DISTINCT clip_id
  FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
  WHERE {match_predicate}
),
bg_geo AS (
  -- deduped block group boundaries
  SELECT GEOID AS geoid, geometry
  FROM `{bq_project}.{boundary_dataset}.census_blockgroup`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY GEOID ORDER BY geometry IS NOT NULL DESC) = 1
),
bg_counts AS (
  -- count matched business clips per block group via the parcel-universe xref
  SELECT a.census_block_group_geoid,
         COUNT(DISTINCT a.clip_list) AS total_unq_clips,
         COUNT(DISTINCT b.clip_id)   AS unq_{store}_clips,
         ROUND(100 * SAFE_DIVIDE(COUNT(DISTINCT b.clip_id),
                                 COUNT(DISTINCT a.clip_list)), 2) AS {store}_clip_pct
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref` a
  LEFT JOIN matched b ON a.clip_list = CAST(b.clip_id AS STRING)
  GROUP BY 1
)
SELECT a.*,
       LEFT(a.census_block_group_geoid, 2) AS statefp,
       c.geometry
FROM bg_counts a
LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid

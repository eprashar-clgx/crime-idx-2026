-- Generic block-group store-count builder (one template for every store type).
-- Rolls a firmographics business universe up to block-group clip counts via the
-- parcel-universe xref. Materializes `bg_{{store}}` in the staging dataset.
--
-- Task params (filled per store on top of the project/dataset defaults):
--   {{store}}            table/column stem, e.g. convenience_stores | liquor_stores | gas_stations
--   {{match_predicate}}  firmographics WHERE clause defining the store universe
--                      (NAICS/SIC codes and/or business-name LIKEs)
CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_{store}` AS
WITH matched AS (
  -- clips whose firmographics record matches the store definition
  SELECT DISTINCT CAST(clip_id AS STRING) AS clip_id
  FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
  WHERE {match_predicate}
),
clip_bg AS (
  -- property (clip) universe: explode the pipe-delimited clip_list so multi-clip parcels
  -- (strip malls, condos) match and the denominator counts properties, not parcel shapes.
  -- The ~0.2% of clips that fall in >1 BG are assigned to the BG holding most of their
  -- parcel shapes.
  SELECT clip, census_block_group_geoid
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref`,
       UNNEST(SPLIT(clip_list, "|")) AS clip
  WHERE clip != '' AND census_block_group_geoid IS NOT NULL
  GROUP BY clip, census_block_group_geoid
  QUALIFY ROW_NUMBER() OVER (PARTITION BY clip ORDER BY COUNT(*) DESC, census_block_group_geoid) = 1
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
         COUNT(DISTINCT a.clip)      AS total_unq_clips,
         COUNT(DISTINCT b.clip_id)   AS unq_{store}_clips,
         ROUND(100 * SAFE_DIVIDE(COUNT(DISTINCT b.clip_id),
                                 COUNT(DISTINCT a.clip)), 2) AS {store}_clip_pct
  FROM clip_bg a
  LEFT JOIN matched b ON a.clip = b.clip_id
  GROUP BY 1
)
SELECT a.*,
       LEFT(a.census_block_group_geoid, 2) AS statefp,
       c.geometry
FROM bg_counts a
LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid

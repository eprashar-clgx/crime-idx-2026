CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_clip_foreclosures` AS
WITH clip_foreclosures AS (
  SELECT DISTINCT puid
  FROM `{idap_project}.edr_ent_property_fulfillment.vw_transaction_v1`
  WHERE deedcattyp = 'U'                                   -- U = foreclosure
    AND puid IS NOT NULL                                   -- keep only clipped records
    AND SAFE.PARSE_DATE('%Y%m%d', CAST(recordingdt AS STRING)) >= DATE '2022-01-01'-- 3 years of foreclosures; excluding Covid years
    AND SAFE.PARSE_DATE('%Y%m%d', CAST(recordingdt AS STRING)) <  DATE '2025-01-01'
),
bg_geo AS (
  -- deduped block group boundaries
  SELECT GEOID AS geoid, geometry
  FROM `{bq_project}.{boundary_dataset}.census_blockgroup`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY GEOID ORDER BY geometry IS NOT NULL DESC) = 1
),
bg_foreclosures AS (
  -- aggregate clips to block groups via the parcel-universe xref
  SELECT a.census_block_group_geoid,
         COUNT(DISTINCT a.clip_list) AS total_unq_clips,
         COUNT(DISTINCT b.puid)      AS unq_clip_w_foreclosure,
         ROUND(100 * SAFE_DIVIDE(COUNT(DISTINCT b.puid),
                                 COUNT(DISTINCT a.clip_list)), 2) AS clip_foreclosure_pct
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref` a
  LEFT JOIN clip_foreclosures b ON a.clip_list = CAST(b.puid AS STRING)
  GROUP BY 1
)
SELECT a.*,
       LEFT(a.census_block_group_geoid, 2) AS statefp,
       c.geometry
FROM bg_foreclosures a
LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid
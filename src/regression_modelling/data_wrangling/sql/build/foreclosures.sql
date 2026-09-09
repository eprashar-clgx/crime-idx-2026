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
),
base AS (
  SELECT a.*,
         LEFT(a.census_block_group_geoid, 2) AS statefp,
         c.geometry,
         ST_CENTROID(c.geometry) AS centroid
  FROM bg_foreclosures a
  LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid
),
-- KNN(6) spatial lag: each BG's 6 nearest neighbours WITHIN THE SAME STATE (self excluded).
-- The 25km ST_DWITHIN prefilter prunes the candidate set (enables BQ's spatial join
-- optimization) while comfortably covering the 6 nearest neighbours in populated areas.
neighbors AS (
  SELECT b.census_block_group_geoid AS geoid,
         n.clip_foreclosure_pct AS nbr_pct,
         ROW_NUMBER() OVER (PARTITION BY b.census_block_group_geoid
                            ORDER BY ST_DISTANCE(b.centroid, n.centroid)) AS rnk
  FROM base b
  JOIN base n
    ON b.statefp = n.statefp
   AND b.census_block_group_geoid != n.census_block_group_geoid
   AND ST_DWITHIN(b.centroid, n.centroid, 25000)
),
lag AS (
  SELECT geoid, AVG(nbr_pct) AS clip_foreclosure_pct_lag6
  FROM neighbors
  WHERE rnk <= 6
  GROUP BY 1
)
SELECT b.* EXCEPT (centroid),
       l.clip_foreclosure_pct_lag6
FROM base b
LEFT JOIN lag l ON b.census_block_group_geoid = l.geoid
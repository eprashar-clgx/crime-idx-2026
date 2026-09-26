CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_clip_foreclosures` AS
-- Foreclosure share OF TRANSACTED properties: numerator = properties with a foreclosure deed,
-- denominator = properties with ANY recorded transaction (the same 2020-2024 universe as
-- transactions.sql), so this measures distress among sales separately from turnover itself.
WITH clip_transactions AS (
  -- one row per transacted property, flagged if any of its transactions was a foreclosure
  SELECT CAST(puid AS STRING) AS puid,
         LOGICAL_OR(COALESCE(deedcattyp = 'U', FALSE)) AS is_foreclosure   -- U = foreclosure
  FROM `{idap_project}.edr_ent_property_fulfillment.vw_transaction_v1`
  WHERE puid IS NOT NULL                                   -- keep only clipped records
    AND SAFE.PARSE_DATE('%Y%m%d', CAST(recordingdt AS STRING)) >= DATE '2020-01-01'-- 5 years (2020-2024)
    AND SAFE.PARSE_DATE('%Y%m%d', CAST(recordingdt AS STRING)) <  DATE '2025-01-01'
  GROUP BY 1
),
clip_bg AS (
  -- property (clip) universe: explode the pipe-delimited clip_list so the denominator counts
  -- properties, not parcel shapes (a condo parcel carries many clips). The ~0.2% of clips
  -- that fall in >1 BG are assigned to the BG holding most of their parcel shapes.
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
bg_foreclosures AS (
  -- share of the BG's transacted properties that were foreclosed. BGs with no transactions
  -- get NULL (SAFE_DIVIDE), which ZERO_FILL turns into 0 downstream.
  SELECT a.census_block_group_geoid,
         COUNT(DISTINCT a.clip)                           AS total_unq_clips,
         COUNT(DISTINCT b.puid)                           AS unq_clip_w_transaction,
         COUNT(DISTINCT IF(b.is_foreclosure, b.puid, NULL)) AS unq_clip_w_foreclosure,
         ROUND(100 * SAFE_DIVIDE(COUNT(DISTINCT IF(b.is_foreclosure, b.puid, NULL)),
                                 COUNT(DISTINCT b.puid)), 2) AS clip_foreclosure_pct
  FROM clip_bg a
  LEFT JOIN clip_transactions b ON a.clip = b.puid
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
-- KNN(6) POOLED spatial lag: each BG's 6 nearest neighbours WITHIN THE SAME STATE (self
-- excluded), combined as SUM(neighbour numerators) / SUM(neighbour denominators) rather than
-- the mean of neighbour rates, so a neighbour with a tiny denominator (e.g. 1 of 2 sales)
-- cannot dominate the lag. See docs/features/property_distress.md.
-- The 25km ST_DWITHIN prefilter prunes the candidate set (enables BQ's spatial join
-- optimization) while comfortably covering the 6 nearest neighbours in populated areas.
neighbors AS (
  SELECT b.census_block_group_geoid AS geoid,
         n.unq_clip_w_foreclosure AS nbr_num,
         n.unq_clip_w_transaction AS nbr_den,
         ROW_NUMBER() OVER (PARTITION BY b.census_block_group_geoid
                            ORDER BY ST_DISTANCE(b.centroid, n.centroid)) AS rnk
  FROM base b
  JOIN base n
    ON b.statefp = n.statefp
   AND b.census_block_group_geoid != n.census_block_group_geoid
   AND ST_DWITHIN(b.centroid, n.centroid, 25000)
),
lag AS (
  SELECT geoid, ROUND(100 * SAFE_DIVIDE(SUM(nbr_num), SUM(nbr_den)), 2) AS clip_foreclosure_pct_lag6
  FROM neighbors
  WHERE rnk <= 6
  GROUP BY 1
)
SELECT b.* EXCEPT (centroid),
       l.clip_foreclosure_pct_lag6
FROM base b
LEFT JOIN lag l ON b.census_block_group_geoid = l.geoid
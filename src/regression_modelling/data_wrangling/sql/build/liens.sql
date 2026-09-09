CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_clip_liens` AS
WITH clip_liens AS (
  SELECT clip, lien_amount, type_of_tax
  FROM `{idap_project}.edr_ent_property_transactions.vw_involuntary_liens_lien`
  WHERE category_type = 'J'                       -- J = lien, R = release
    AND clip IS NOT NULL                          -- keep only clipped records
    AND EXTRACT(YEAR FROM lien_date) >= 2022     -- 3 years (2022-2024), matching foreclosures; excludes Covid years
    AND EXTRACT(YEAR FROM lien_date) <  2025
    AND type_of_tax IN ('DELINQUENT TAX', 'PERSONAL PROPERTY TAX',
                        'POSTPONED PROPERTY TAX', 'UNSECURED PROPERTY (TAXES)')
    AND lien_amount >= 100
),
bg_geo AS (
  -- deduped block group boundaries
  SELECT GEOID AS geoid, geometry
  FROM `{bq_project}.{boundary_dataset}.census_blockgroup`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY GEOID ORDER BY geometry IS NOT NULL DESC) = 1
),
bg_liens AS (
  -- aggregate clips to block groups via the parcel-universe xref
  SELECT a.census_block_group_geoid,
         COUNT(DISTINCT a.clip_list) AS total_clips,
         COUNT(DISTINCT b.clip)      AS clip_w_liens,
         ROUND(100 * SAFE_DIVIDE(COUNT(DISTINCT b.clip),
                                 COUNT(DISTINCT a.clip_list)), 2) AS clip_liens_pct
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref` a
  LEFT JOIN clip_liens b ON a.clip_list = b.clip
  GROUP BY 1
),
base AS (
  SELECT a.*,
         LEFT(a.census_block_group_geoid, 2) AS statefp,
         c.geometry,
         ST_CENTROID(c.geometry) AS centroid
  FROM bg_liens a
  LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid
),
-- KNN(6) spatial lag: each BG's 6 nearest neighbours WITHIN THE SAME STATE (self excluded).
-- The 25km ST_DWITHIN prefilter prunes the candidate set (enables BQ's spatial join
-- optimization) while comfortably covering the 6 nearest neighbours in populated areas.
neighbors AS (
  SELECT b.census_block_group_geoid AS geoid,
         n.clip_liens_pct AS nbr_pct,
         ROW_NUMBER() OVER (PARTITION BY b.census_block_group_geoid
                            ORDER BY ST_DISTANCE(b.centroid, n.centroid)) AS rnk
  FROM base b
  JOIN base n
    ON b.statefp = n.statefp
   AND b.census_block_group_geoid != n.census_block_group_geoid
   AND ST_DWITHIN(b.centroid, n.centroid, 25000)
),
lag AS (
  SELECT geoid, AVG(nbr_pct) AS clip_liens_pct_lag6
  FROM neighbors
  WHERE rnk <= 6
  GROUP BY 1
)
SELECT b.* EXCEPT (centroid),
       l.clip_liens_pct_lag6
FROM base b
LEFT JOIN lag l ON b.census_block_group_geoid = l.geoid
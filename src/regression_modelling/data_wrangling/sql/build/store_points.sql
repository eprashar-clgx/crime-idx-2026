-- Store POI point layer: one row per matched business property (clip), for the transit
-- risky-facility co-location features (H1/H3, transit/colocation.py). Uses the SAME
-- firmographics universe as stores.sql (STORE_DEFS predicate), geocoded with the EAP
-- lat/lon from clip_address_xref. Materializes `{{store}}_points` in the staging dataset.
--
-- Task params (filled per store on top of the project/dataset defaults):
--   {{store}}            table stem, e.g. convenience_stores | liquor_stores | gas_stations
--   {{match_predicate}}  firmographics WHERE clause defining the store universe
CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.{store}_points` AS
WITH matched AS (
  -- clips whose firmographics record matches the store definition
  SELECT DISTINCT CAST(clip_id AS STRING) AS clip
  FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
  WHERE {match_predicate}
),
coords AS (
  -- EAP coordinates; a clip can carry several addresses, averaged to one point per clip
  SELECT CAST(x.clip AS STRING) AS clip,
         AVG(SAFE_CAST(x.eapLatitude  AS FLOAT64)) AS lat,
         AVG(SAFE_CAST(x.eapLongitude AS FLOAT64)) AS lon
  FROM `{idap_project}.edr_pmd_property_pipeline.clip_address_xref` x
  JOIN matched m ON CAST(x.clip AS STRING) = m.clip
  WHERE x.eapLatitude IS NOT NULL AND x.eapLongitude IS NOT NULL
  GROUP BY 1
),
clip_bg AS (
  -- block group of each matched clip (exploded clip_list; clips in >1 BG go to the BG
  -- holding most of their parcel shapes, same rule as the count builds)
  SELECT clip, census_block_group_geoid
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref`,
       UNNEST(SPLIT(clip_list, "|")) AS clip
  WHERE clip IN (SELECT clip FROM matched) AND census_block_group_geoid IS NOT NULL
  GROUP BY clip, census_block_group_geoid
  QUALIFY ROW_NUMBER() OVER (PARTITION BY clip ORDER BY COUNT(*) DESC, census_block_group_geoid) = 1
)
SELECT c.clip                       AS clip_id,
       b.census_block_group_geoid   AS geoid,
       c.lat,
       c.lon,
       ST_GEOGPOINT(c.lon, c.lat)   AS geom
FROM coords c
LEFT JOIN clip_bg b ON c.clip = b.clip
WHERE c.lat BETWEEN -90 AND 90
  AND c.lon BETWEEN -180 AND 180
  AND NOT (c.lat = 0 AND c.lon = 0)          -- drop null-island geocodes

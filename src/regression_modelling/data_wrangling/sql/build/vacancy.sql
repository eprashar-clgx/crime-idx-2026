CREATE OR REPLACE TABLE `{bq_project}.{staging_dataset}.bg_vacancy` AS
WITH addr_vacancy AS (
  -- vacancy info for ~180M of 214M addresses
  SELECT address_id, DELIVERY_POINT_OCCUPANCY
  FROM `{idap_project}.edr_pmd_property_pipeline.vw_address_connect`
  WHERE mls_pin != 1        -- exclude MLS-sourced addresses
    AND mls_redistrb != 1   -- exclude non-marketable addresses
),
parcel_addr AS (
  -- vacancy rolled up to the ~120M parcels linked to addresses
  SELECT a.parcel_shape_id,
         COUNTIF(b.DELIVERY_POINT_OCCUPANCY = 1) AS vacant_addr,
         COUNT(b.DELIVERY_POINT_OCCUPANCY)        AS total_addr
  FROM `{idap_project}.edr_pmd_property_pipeline.vw_parcel_to_address` a
  LEFT JOIN addr_vacancy b ON a.address_id = b.address_id
  WHERE addr_status = 'A' AND parcel_status = 'A'
  GROUP BY 1
),
bg_geo AS (
  -- deduped block group boundaries
  SELECT GEOID AS geoid, geometry
  FROM `{bq_project}.{boundary_dataset}.census_blockgroup`
  QUALIFY ROW_NUMBER() OVER (PARTITION BY GEOID ORDER BY geometry IS NOT NULL DESC) = 1
),
bg_addr AS (
  -- aggregate parcels to block groups via the parcel-universe xref
  SELECT a.census_block_group_geoid,
         SUM(b.vacant_addr) AS vacant_addr,
         SUM(b.total_addr)  AS total_addr,
         100 * SAFE_DIVIDE(SUM(b.vacant_addr), SUM(b.total_addr)) AS vacant_pct
  FROM `{bq_project}.{boundary_dataset}.NS_pcl_universe_xref` a
  LEFT JOIN parcel_addr b ON a.parcel_shape_id = b.parcel_shape_id
  GROUP BY 1
)
SELECT a.*,
       LEFT(a.census_block_group_geoid, 2) AS statefp,
       c.geometry
FROM bg_addr a
LEFT JOIN bg_geo c ON a.census_block_group_geoid = c.geoid
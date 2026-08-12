-- Store counts by state (state = first 2 chars of county_code) and business_name.
SELECT
  LEFT(b.county_code, 2) AS state_code,
  a.business_name,
  COUNT(*) AS cnts
FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise` a
JOIN `{idap_project}.edr_pmd_property_pipeline.clip_to_parcel` b
  ON CAST(a.clip_id AS STRING) = b.clip
WHERE {naics_predicate}
  AND {name_predicate}
GROUP BY state_code, a.business_name
ORDER BY cnts DESC

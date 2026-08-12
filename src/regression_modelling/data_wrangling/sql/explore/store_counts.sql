-- National store counts per business_name, filtered by NAICS + name/brand.
SELECT
  business_name,
  COUNT(*) AS cnts,
  SUM(COUNT(*)) OVER() AS total_stores,
  ROUND((COUNT(*) * 100.0) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total
FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
WHERE {naics_predicate}
  AND {name_predicate}
GROUP BY business_name
ORDER BY cnts DESC

-- Distribution of primary NAICS codes for a store's name/brand match.
-- Discovery query: no NAICS filter (that is what we are trying to identify).
SELECT
  naics_6_digit_primary_code,
  naics_6_digit_primary_code_description,
  COUNT(*) AS cnts,
  ROUND((COUNT(*) * 100.0) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total
FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
WHERE {name_predicate}
GROUP BY naics_6_digit_primary_code, naics_6_digit_primary_code_description
ORDER BY cnts DESC

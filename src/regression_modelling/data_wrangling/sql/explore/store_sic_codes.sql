-- Distribution of primary SIC codes for a store's name/brand match.
SELECT
  sic_4_digit_primary_code,
  sic_4_digit_primary_code_description,
  COUNT(*) AS cnts,
  ROUND((COUNT(*) * 100.0) / SUM(COUNT(*)) OVER(), 2) AS pct_of_total
FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise`
WHERE {name_predicate}
GROUP BY sic_4_digit_primary_code, sic_4_digit_primary_code_description
ORDER BY cnts DESC

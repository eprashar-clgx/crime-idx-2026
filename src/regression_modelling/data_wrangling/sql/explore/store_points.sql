-- Store parcel geometries (for folium mapping), filtered by NAICS + name/brand.
SELECT
  a.clip_id,
  a.business_name,
  b.parcel_polygon_at_eventtime
FROM `{idap_project}.edr_ent_property_firmographics.vw_edr_firmographics_enterprise` a
JOIN `{idap_project}.edr_pmd_property_pipeline.clip_to_parcel` b
  ON CAST(a.clip_id AS STRING) = b.clip
WHERE {naics_predicate}
  AND {name_predicate}

-- sql/pull/vacancy.sql
SELECT census_block_group_geoid AS geoid, vacant_pct, vacant_pct_lag6, vacant_addr, total_addr
FROM `work_eprashar.bg_vacancy`
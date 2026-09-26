# Data catalog — crime-idx-2026

`data/` is **gitignored** (only this catalog is tracked). It holds every dataset the
tasks read or write. Keep this table updated when a source is added, moved, or retired —
it is the map of *which dataset feeds which module and why*.

Tiers: **raw** (immutable external sources) -> **interim** (shared derived, cached
Parquet) -> **processed** (experiment-specific model tables).

## Raw (`data/raw`) — immutable external sources

| Dataset | Path | Source | Consumed by | Purpose |
|---|---|---|---|---|
| US Census places | `raw/boundaries/cb_2024_us_place_500k/` | Census TIGER | `crime_blockgroup_mapping.boundaries` | City polygons (place FIPS) to define city extent |
| State block groups | `raw/boundaries/cb_2025_{state}_bg_500k.zip` | Census TIGER | `crime_blockgroup_mapping.boundaries` | BG polygons per state (`CityConfig.bg_zip`) |
| Chicago crime | `raw/city_crime/chicago/Crimes_-_2025_*.csv` | Chicago data portal | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (FBI-code mapping) |
| Houston crime | `raw/city_crime/houston/NIBRSPublicView2025.csv` | Houston PD (NIBRS) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS mapping) |
| Atlanta crime | `raw/city_crime/atlanta/OpenDataWebsite_Crime_view_*.csv` | [Atlanta PD ArcGIS Open Data](https://experience.arcgis.com/experience/d5dd2be2977d40acb340ef42f80671b8/) (FeatureServer, rolling 2021-present) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `NibrsUcrCode` mapping); `Latitude`/`Longitude` in WGS84 |
| Sacramento crime | `raw/city_crime/sacramento/Sacramento_Report_Data_2025_*.csv` | [Sacramento PD ArcGIS Public Safety](https://experience.arcgis.com/experience/cb5a65e7a67743778cd277c7e3761b5a) (per-year FeatureServer) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `Offense_Code` mapping); coords in `x`/`y` **CA State Plane Zone II, EPSG:2226** (reprojected to 4326 on load); ~44% rows have null coords (confidential cases) |
| San Francisco crime | `raw/city_crime/san francisco/Police_Department_Incident_Reports__2018_to_Present_*.csv` | [DataSF / SFPD (Socrata `wg3w-h783`)](https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-2018-to-Present/wg3w-h783/about_data) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (text `incident_category` mapping); `Latitude`/`Longitude` in WGS84 (intersection-offset). **PROPERTY-ONLY** (`property_only=True`): only ~21 incidents/yr carry the "Rape" category (coords present but implausibly low, ~350/yr expected) — sexual assault is effectively not published as mappable rape; non-rape violent is geolocated. |
| Pittsburgh crime | `raw/city_crime/pittsburgh/incidents_2024_2026.xlsx` | [WPRDC / City of Pittsburgh (Monthly Criminal Activity, NIBRS)](https://data.wprdc.org/dataset/monthly-criminal-activity-dashboard/resource/bd41992a-987a-4cca-8798-fbe1cd946b07) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `NIBRS_Offense_Code` mapping); **XLSX** source; `XCOORD`/`YCOORD` = WGS84 lon/lat as text. **PROPERTY-ONLY** (`property_only=True`): sex-offense coords nulled at source (301 rape rows 11A-D/36B in 2025, 0 geolocated, vs ~99% overall); non-rape violent is geolocated. |
| Detroit crime | `raw/city_crime/detroit/RMS_Crime_Incidents.csv` | [Detroit Open Data / DPD (RMS Crime Incidents, ArcGIS)](https://data.detroitmi.gov/datasets/detroitmi::rms-crime-incidents/about) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (Michigan MICR text `offense_category` mapping); `latitude`/`longitude` WGS84, ~99.6% populated **including violent crime** (unlike Sacramento); ships pre-joined `census_block_2020_geoid`. Full multi-year download; filtered to 2025 via `year_filter`. Chosen as the full-coverage replacement for Sacramento. |
| Kansas City crime | `raw/city_crime/kansas/KCPD_Crime_Data_2025_*.csv` | [Open Data KC / KCPD (Socrata `dmnp-9ajg`)](https://data.kcmo.org/dataset/KCPD-Crime-Data-2025/dmnp-9ajg) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `IBRS` mapping); coords are WKT `POINT (lon lat)` in a single `Location` column (WGS84, parsed via `wkt_col`). **One row per person-involvement** (VIC/SUS/ARR) -> collapsed to one row per offense via `dedup_keys=(report_no, ibrs)`, preferring coord-bearing rows. ~97% geolocated **including rape** (452 offenses); occurrence date = `from_date`. Car-dependent city (low-transit feature test). |
| Columbus crime | `raw/city_crime/columbus/Police_Incident_Reports_*.csv` | [Open Data Columbus / CPD (ArcGIS Hub item `b70656e5b22d4a7db8af9b4289bf8c27`)](https://opendata.columbus.gov/datasets/columbus::police-incident-reports/about) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (text `GeneralSubject` mapping); ArcGIS Hub **CSV export** (`.../api/download/v1/items/b70656e5.../csv?layers=0&spatialRefId=4326`) adds `x`/`y` = WGS84 lon/lat, populated only when `Is_Mapped='Yes'` (~84% of rows). **PROPERTY-ONLY** (`property_only=True`): rape coords 100% suppressed at source (non-rape violent — robbery/assault/murder — is geolocated). Full 3-year rolling feed; filtered to 2025 via `year_filter` on `occurred_on`. Car-dependent (COTA bus-only, ~1.9% transit). |
| Jacksonville crime | `raw/city_crime/jacksonville/JSO_Public_Transparency_*.csv` | [JSO Transparency (ArcGIS item `29a91fb9881e405e914c41ce07fc05f9`)](https://www.arcgis.com/home/item.html?id=29a91fb9881e405e914c41ce07fc05f9) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `nibrs_code` mapping; JSO-local `23X`=THEFT added). ArcGIS Hub **CSV export** (`.../api/download/v1/items/29a91fb9.../csv?layers=0&spatialRefId=4326`) adds `x`/`y` = WGS84 lon/lat, ~100% populated. **PROPERTY-ONLY** (`property_only=True`): rape (11A-11D) absent by Florida Marsy's Law (non-rape violent geolocated). Full multi-year rolling feed; filtered to 2025 via `year_filter` on `incident_date`. Car-dependent city (low-transit feature test). |
| Philadelphia crime | `raw/city_crime/philadelphia/incidents_2025_*.csv` | [OpenDataPhilly / Philadelphia PD (Carto SQL API, `incidents_part1_part2`)](https://opendataphilly.org/datasets/crime-incidents/) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (UCR general text `text_general_code` mapping). `point_x`/`point_y` WGS84, ~98% populated **including rape** (no extra suppression). Pulled 2025-only via Carto SQL `WHERE dispatch_date` filter; date col `dispatch_date`. |
| Milwaukee crime | `raw/city_crime/milwaukee/wibr_*.csv` | [City of Milwaukee Open Data / MPD (CKAN, "NIBRS Crime Data")](https://data.milwaukee.gov/dataset/wibr) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS codes in `Offense_All`, semicolon-joined for multi-offense incidents). `Address_Longitude`/`Address_Latitude` WGS84, ~94% populated **including rape** (drop a known placeholder point at 43.1953...). Full multi-year rolling feed (2024-2026); filtered to 2025 via `year_filter` on `Incident_Date`. |
| Baltimore crime | `raw/city_crime/baltimore/nibrs_group_a_*.csv` | [Open Baltimore / BPD (ArcGIS Hub item `204beefe92a645d79fdf0969957bbdf8`, "NIBRS Group A Crime Data")](https://data.baltimorecity.gov/datasets/baltimore::nibrs-group-a-crime-data/about) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `CrimeCode` mapping). `Latitude`/`Longitude` WGS84, ~99.7% populated **including rape**. Full multi-year rolling feed (NIBRS since 2025-01-01); filtered to 2025 via `year_filter` on `CrimeDateTime`. |
| Washington DC crime | `raw/city_crime/dc/crime_incidents_2025_*.csv` | [Open Data DC / MPD (ArcGIS Hub item `74d924ddc3374e3b977e6f002478cb9b`, "Crime Incidents in 2025", layer 7)](https://opendata.dc.gov/datasets/DCGIS::crime-incidents-in-2025) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (local 9-value `OFFENSE` mapping). `LATITUDE`/`LONGITUDE` WGS84, 100% populated; **every** crime (incl. "SEX ABUSE") is rounded to the street-block centroid uniformly — not extra-suppressed for rape. Ships a pre-computed `BLOCK_GROUP` field (calculated *before* block-level anonymization, per source metadata) that can be used in place of a spatial join. Already 2025-only (per-year Hub item); date col `REPORT_DAT`. |
| Oakland crime | `raw/city_crime/oakland/crimewatch_2025_*.csv` | [data.oaklandca.gov / OPD (Socrata `ppgh-7dqv`, "CrimeWatch Data")](https://data.oaklandca.gov/Public-Safety/CrimeWatch-Data/ppgh-7dqv) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target — **needs a manual category crosswalk**: local (non-NIBRS/UCR) `crimetype`/`description` text, e.g. rape=`FORCIBLE RAPE`. Socrata Point `location` (lon/lat) WGS84, ~90% populated **including rape** (nulls look like geocode failures on "UNKNOWN"/vague addresses, not deliberate suppression). Pulled 2025-only via SoQL `$where` on `datetime`. |
| New York City crime | `raw/city_crime/new_york/nypd_complaints_2025_*.csv` | [NYC Open Data / NYPD (Socrata `qgea-i56i`, "NYPD Complaint Data Historic" — use Historic, not the YTD feed)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NY Penal Law `ky_cd`/`ofns_desc` mapping, not NIBRS/UCR). **PROPERTY-ONLY** (`property_only=True`): `latitude`/`longitude` are non-null for rape but **fake** — every rape record is placed at its precinct station house (~78 distinct points across all rape rows; official dataset footnote 10), while non-rape violent/property crime is geocoded near-uniquely. Pulled 2025-only via SoQL `$where` on `cmplnt_fr_dt`. |
| Dallas crime | `raw/city_crime/dallas/police_incidents_2025_*.csv` | [Dallas Open Data / DPD (Socrata `qv6i-rri7`, "Police Incidents")](https://www.dallasopendata.com/Public-Safety/Police-Incidents/qv6i-rri7) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (dual `nibrs_crime_category` + `ucr_offense` mapping). `geocoded_column` (lat/long) WGS84, ~99.8% populated for the published types. **PROPERTY-ONLY** (`property_only=True`): rape/sexual-assault is **excluded from the dataset by policy** (source description lists "sexually oriented offenses" as a filtered-out category — not just unlocated, entirely absent). One row per offense/involvement -> dedup on `incidentnum`. Pulled 2025-only via SoQL `$where` on `date1`. |
| Denver crime | `raw/city_crime/denver/crime_*.csv` | [Denver Open Data / DPD (ArcGIS Hub item `16d9c82bb36c4475bf87189cfaed653c`, "Crime", layer 324)](https://opendata-geospatialdenver.hub.arcgis.com/datasets/geospatialDenver::crime) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (local NIBRS-based `OFFENSE_CATEGORY_ID`/`OFFENSE_TYPE_ID` mapping). `GEO_LAT`/`GEO_LON` WGS84, 100% populated for the 6 published types. **PROPERTY-ONLY** (`property_only=True`): the main layer has **no rape/sexual-assault category at all** (item description: "Addresses of sexual assaults are not included"); a separate "Sex Related Crimes" table exists with counts but no coordinates. Full 5yr+YTD rolling feed; filtered to 2025 via `year_filter` on `FIRST_OCCURRENCE_DATE`. |
| Las Vegas crime | `raw/city_crime/las_vegas/lvmpd_nibrs_crimes_*.csv` | [LVMPD (ArcGIS item `c1ec788896ae46e0b896b72cb98e7ae8`, "LVMPD Reported NIBRS Crimes"; unlisted but public)](https://www.arcgis.com/home/item.html?id=c1ec788896ae46e0b896b72cb98e7ae8) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `NIBRSOffenseCode`/`Offense` mapping). `Latitude`/`Longitude` WGS84, ~99.97% populated. **PROPERTY-ONLY** (`property_only=True`): rape is **not a standalone category** — NIBRS code `11` bundles rape/sodomy/fondling/object-assault into one undocumented "Sex Offense" bucket, so it can't be isolated as a clean rape target; other 6 types are clean. `ReportedOn` is a report date, not occurrence date. Full rolling feed; filtered to 2025 via `year_filter`. |
| Seattle crime | `raw/city_crime/seattle/spd_crime_2025_*.csv` | [data.seattle.gov / SPD (Socrata `tazs-3rd5`, "SPD Crime Data: 2008-Present")](https://data.seattle.gov/Public-Safety/SPD-Crime-Data-2008-Present/tazs-3rd5) | `crime_blockgroup_mapping.crime` | Incident points -> BG crime target (NIBRS `nibrs_offense_code`/`offense_sub_category` mapping). `latitude`/`longitude` literal `"REDACTED"` string when suppressed. **PROPERTY-ONLY, degraded** (`property_only=True`): rape ~98% redacted, but unlike other property-only cities here, **murder is also ~97% redacted and aggravated assault ~30% / robbery ~12%** — non-rape violent coverage is materially weaker than the rest of the property-only tier; property crime itself is ~99% clean. Pulled 2025-only via SoQL `$where` on `offense_date`. |
| GTFS transit feeds | `raw/transit/{city}/*.zip` | Agency GTFS Static (CTA, MARTA, SFMTA+BART, PAAC, METRO, JTA, KCATA, SacRT) via [Mobility Database](https://mobilitydatabase.org/) | `regression_modelling.data_wrangling.transit` | GTFS Static schedule zips (`stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt`, `calendar*.txt`) -> per-stop transit features -> BG predictors. Feed resolved by stable file stem `{feed_id}-*.zip` (`mdb-*` / `tld-*`). Representative service date auto-selected as the peak-service weekday nearest `2025-06-04` (feeds cover different windows; SacRT is a stale Jan–Apr 2025 snapshot). SF = Muni + BART (2 zips, union+dedup). See `docs/transit_eda_plan.md` + ADR 0002. |
| NIBRS offense codes | `raw/dictionaries/NIBRS_Offense_Codes.pdf` | FBI NIBRS | reference | Crime-code -> category mapping reference |
| LODES WAC jobs | `raw/lodes/location_inc_spatial_lodes_wac_2022_block_jobs.csv` | LEHD LODES | `crime_blockgroup_mapping.rates` | Block jobs (`c000`) for daytime-adjusted rates |
| NeighborhoodScout BG model | `raw/neighborhood_scout/location_inc_ns4_2025q4_block_group_data.sav` | location_inc | `crime_blockgroup_mapping.rates` | `population` + existing model `*_pt_ct` scores |
| UCR agency / crosswalk | `raw/neighborhood_scout/location_inc_crime_2024_ucr_*.sav` | location_inc | reference (planned) | UCR agency crosswalk for national rates |
| Carrier evals | `data/evals/evals.parquet` (see note) | carrier | `carrier_eval.evals`, `carrier_eval.scores` | Claims/losses/exposure per block + national `*_pt_u` rates |

_Note: `carrier_eval.config.EVALS_PATH` points at `data/evals/evals.parquet`; raw carrier
drops also live under `raw/insurance_evals/`._

_Note: **Property-only cities** (`CityConfig.property_only=True`) — coordinates for some/all
violent crimes are suppressed at the source, so only property targets (burglary/larceny/mvt)
are reliable:_
- _**Sacramento** — California victim-privacy law nulls coords for **all** violent/sex crimes
  (rape 0%, murder ~11%, aggravated assault ~57% geolocated; property ~99%)._
- _**Columbus** — only **rape** coords are suppressed (0% geolocated); non-rape violent
  (robbery/assault/murder ~96–100%) and property are geolocated._
- _**Jacksonville** — **rape** (NIBRS 11A-11D) absent entirely (FL Marsy's Law); non-rape
  violent and property are ~100% geolocated._
- _**San Francisco** — DataSF surfaces only ~21 "Rape"-category incidents/yr (coords present
  but implausibly low); sexual assault effectively unpublished as mappable rape. Non-rape
  violent geolocated._
- _**Pittsburgh** — WPRDC feed nulls coords for all sex offenses (301 rape rows in 2025, 0
  geolocated, vs ~99% overall). Non-rape violent geolocated._
- _**New York City** — NYPD places every rape record at its precinct station house (~78
  distinct points across all rape rows; official footnote), not the true incident location.
  Non-rape violent/property geocoded near-uniquely._
- _**Dallas** — rape/sexual-assault excluded from the published dataset entirely by policy
  (not just unlocated). Non-rape violent and property are ~99.8% geolocated._
- _**Denver** — main "Crime" layer has no rape/sexual-assault category at all; a separate
  low-detail table exists with counts but no coordinates. Other 6 types 100% geolocated._
- _**Las Vegas** — rape is not a standalone NIBRS category; code `11` bundles
  rape/sodomy/fondling/object-assault into one undocumented "Sex Offense" bucket. Other 6
  types ~99.97% geolocated._
- _**Seattle** — rape ~98% redacted, and **also** murder ~97% redacted, aggravated assault
  ~30% and robbery ~12% redacted — non-rape violent coverage is weaker here than in the
  other property-only cities. Property crime itself ~99% geolocated._

_Full-coverage (incl. rape) cities: **Detroit, Kansas City, Philadelphia, Milwaukee,
Baltimore, Washington DC**. **Oakland** is also full-coverage but needs a manual category
crosswalk (local `crimetype` text, not NIBRS/UCR)._

_**Phoenix** and **Spokane** were researched but not downloaded: neither publishes
coordinates (lat/long or x/y) for **any** crime type, only masked block-address text — they'd
need an address-geocoding step before they fit this pipeline, which is a bigger lift than the
`property_only` flag used above._

## Interim (`data/interim`) — shared derived caches

| Dataset | Path | Produced by | Consumed by | Purpose |
|---|---|---|---|---|
| Per-source pulls | `interim/sources/{name}.parquet` | `regression_modelling.data_wrangling.sources.pull_source` | `features.assemble_features` | Cached BQ/GCS predictor sources (vacancy, liens, foreclosures, transactions, seven_eleven, gas_stations, liquor_stores, demographic) |
| Transit BG features | `interim/sources/transit.parquet` | `regression_modelling.data_wrangling.transit.build_all_transit` | `features.assemble_features` (`transit` FeatureSource, `backend="file"`) | GTFS-derived BG transit predictors for all registered transit cities (5 POC + Jacksonville/Kansas City/Sacramento; stop density, service intensity, overnight, risky co-location, H3). Built out-of-band; see ADR 0002. |
| Transit per-stop cache | `interim/transit/stops/{city}.parquet` | `regression_modelling.data_wrangling.transit.feeds.load_city_stops` | `transit.build.build_transit` | Per-stop feature intermediate (span, overnight, trips/day, route types) before BG aggregation |
| BG predictor matrix | `interim/features/bg_predictors.parquet` | `features.assemble_features` | `data_wrangling.dataset.build_model_table` | National BG feature spine ⋈ all registry sources |
| BG crime target | `interim/bg_crime/{city}.parquet` | `data_wrangling.dataset.build_bg_crime` | `regression_modelling`, `carrier_eval` | BG-level counts + rates + population per city |

## Processed (`data/processed`) — experiment tables

| Dataset | Path | Produced by | Consumed by | Purpose |
|---|---|---|---|---|
| City model table | `processed/regression_modelling/{city}_model_table.parquet` | `data_wrangling.dataset.build_model_table` | `distributions`, `models` | Features ⋈ target, inside-city, imputed, log-transformed |

_Legacy: earlier runs wrote to `processed/prediction/` and `processed/analysis/`; new runs
use `processed/regression_modelling/`._

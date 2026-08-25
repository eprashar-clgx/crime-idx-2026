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

_Use **Detroit** or **Kansas City** for full-coverage violent-crime (incl. rape) block-group
analysis._

## Interim (`data/interim`) — shared derived caches

| Dataset | Path | Produced by | Consumed by | Purpose |
|---|---|---|---|---|
| Per-source pulls | `interim/sources/{name}.parquet` | `regression_modelling.data_wrangling.sources.pull_source` | `features.assemble_features` | Cached BQ/GCS predictor sources (vacancy, liens, foreclosures, seven_eleven, gas_stations, liquor_stores, demographic) |
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

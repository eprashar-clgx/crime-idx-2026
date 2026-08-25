"""Shared vocabulary: city registry, crime taxonomy, and BQ/GCS project ids."""
from pathlib import Path
from dataclasses import dataclass, field

from crime_blockgroup_mapping.config import BOUNDARIES_DIR

# NIBRS → model category mapping (Houston)
NIBRS_TO_CATEGORY = {
    '13A': 'assault',
    '220': 'burglary',
    '23A': 'larceny', '23B': 'larceny', '23C': 'larceny', '23D': 'larceny',
    '23E': 'larceny', '23F': 'larceny', '23G': 'larceny', '23H': 'larceny',
    '09A': 'murder', '09B': 'murder', '09C': 'murder',
    '240': 'mvt',
    '11A': 'rape', '11B': 'rape', '11C': 'rape', '11D': 'rape', '36B': 'rape',
    '120': 'robbery',
    '290': 'vandal',
    '200': 'fire',
}

# Chicago FBI Code → model category mapping
CHICAGO_FBI_TO_CATEGORY = {
    '01A': 'murder',
    '01B': 'murder',
    '02': 'rape',
    '03': 'robbery',
    '04A': 'assault',
    '04B': 'assault',
    '05': 'burglary',
    '06': 'larceny',
    '07': 'mvt',
    '09': 'fire',
    '14': 'vandal',
}

# San Francisco Incident Category -> model category mapping (text categories, not codes).
# Note: SF "Assault" bundles simple + aggravated (unlike the NIBRS cities, which map only
# 13A aggravated). "Malicious Mischief" is SF's term for vandalism.
SF_CATEGORY_TO_CATEGORY = {
    'Homicide': 'murder',
    'Rape': 'rape',
    'Robbery': 'robbery',
    'Assault': 'assault',
    'Burglary': 'burglary',
    'Larceny Theft': 'larceny',
    'Motor Vehicle Theft': 'mvt',
    'Motor Vehicle Theft?': 'mvt',
    'Arson': 'fire',
    'Malicious Mischief': 'vandal',
    'Vandalism': 'vandal',
}

# Detroit RMS Crime Incidents: text `offense_category` -> model category.
# Michigan MICR taxonomy. Choices mirror the NIBRS cities (Part I index crimes):
#   - Only AGGRAVATED ASSAULT counts as `assault` (simple "ASSAULT" excluded, like NIBRS 13A-only).
#   - SEXUAL ASSAULT (criminal sexual conduct) -> `rape`; broader "SEX OFFENSES" left unmapped.
#   - HOMICIDE -> `murder`; JUSTIFIABLE HOMICIDE intentionally excluded.
#   - STOLEN VEHICLE -> `mvt`; DAMAGE TO PROPERTY -> `vandal`; ARSON -> `fire`.
DETROIT_CATEGORY_TO_CATEGORY = {
    'HOMICIDE': 'murder',
    'SEXUAL ASSAULT': 'rape',
    'ROBBERY': 'robbery',
    'AGGRAVATED ASSAULT': 'assault',
    'BURGLARY': 'burglary',
    'LARCENY': 'larceny',
    'STOLEN VEHICLE': 'mvt',
    'DAMAGE TO PROPERTY': 'vandal',
    'ARSON': 'fire',
}

# Columbus, OH: text `GeneralSubject` -> model category (incident layer carries geometry).
# PROPERTY-ONLY use: rape coordinates are 100% suppressed at the source, so rape rows are
# dropped (no coords) before mapping. Following the NIBRS-city conventions:
#   - Only "Felony Assault" -> `assault` (simple "Assault" excluded, like NIBRS 13A-only).
#   - "Homicide" -> `murder`; justifiable/negligent/vehicular homicide left unmapped.
#   - Burglary + Breaking and Entering (+ attempts) -> `burglary` (NIBRS 220 bundles both).
#   - Motor Vehicle Theft (+ attempt) -> `mvt`; theft variants -> `larceny` (NIBRS 23x).
COLUMBUS_SUBJECT_TO_CATEGORY = {
    'Homicide': 'murder',
    'Rape/Sexual Assault Vic 16 Yr and Older': 'rape',
    'Rape/Sexual Assault Vic 15 Yr and Younger': 'rape',
    'Robbery': 'robbery',
    'Felony Assault': 'assault',
    'Burglary': 'burglary',
    'Breaking and Entering': 'burglary',
    'Burglary Attempt': 'burglary',
    'Theft': 'larceny',
    'Felony Theft': 'larceny',
    'Theft of License Plate': 'larceny',
    'Theft of Negotiable Instrument': 'larceny',
    'Theft of Utilities': 'larceny',
    'Motor Vehicle Theft': 'mvt',
    'Motor Vehicle Theft Attempt': 'mvt',
    'Criminal Damaging': 'vandal',
    'Damage To Property': 'vandal',
    'Vandalism': 'vandal',
    'Arson': 'fire',
}

# Jacksonville, FL (JSO): NIBRS codes, but JSO collapses larceny/theft into a local `23X`
# ("THEFT") code (23A-23H aren't used) alongside standard `23F` theft-from-vehicle.
# PROPERTY-ONLY use: rape (11A-11D) is absent by Florida's Marsy's Law.
JACKSONVILLE_NIBRS_TO_CATEGORY = {**NIBRS_TO_CATEGORY, '23X': 'larceny'}

CRIME_CATEGORIES = [
    'assault', 'burglary', 'larceny', 'murder', 'mvt',
    'rape', 'robbery', 'vandal', 'fire',
    'violent', 'property', 'total', 'cl_total', 'wtotal', 'wprop'
]

@dataclass
class CityConfig:
    name: str
    state_fips: str
    place_fips: str
    crime_csv: str
    lat_col: str
    lon_col: str
    crime_type_col: str
    crime_type_mapping: dict = field(repr=False)
    date_col: str = ""                                   # incident-date column (normalized name)
    crs: str = "EPSG:4326"                               # source CRS of coord cols; reprojected to 4326 on load
    year_filter: tuple = ("2025-01-01", "2026-01-01")    # keep rows with date in [start, end); None disables
    wkt_col: str = ""                                    # single WKT geometry col (e.g. 'POINT (lon lat)'); overrides lat/lon
    dedup_keys: tuple = ()                               # collapse multi-row-per-incident sources to one row per offense
    property_only: bool = False                          # True when the source suppresses violent/sex-crime coords (use property targets only)

    @property
    def bg_zip(self) -> Path:
        return BOUNDARIES_DIR / f"cb_2025_{self.state_fips}_bg_500k.zip"

CITIES = {
    "houston": CityConfig(
        name="Houston",
        state_fips="48",
        place_fips="35000",
        crime_csv="raw/city_crime/houston/NIBRSPublicView2025.csv",
        lat_col="map_latitude",
        lon_col="map_longitude",
        crime_type_col="nibrs_class",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="occurrence_date",
    ),
    "chicago": CityConfig(
        name="Chicago",
        state_fips="17",
        place_fips="14000",
        crime_csv="raw/city_crime/chicago/Crimes_-_2025_20260514.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="fbi_code",
        crime_type_mapping=CHICAGO_FBI_TO_CATEGORY,
        date_col="date",
    ),
    "atlanta": CityConfig(
        name="Atlanta",
        state_fips="13",
        place_fips="04000",
        crime_csv="raw/city_crime/atlanta/OpenDataWebsite_Crime_view_2342186639938035672.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="nibrsucrcode",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="occurredfromdate",
    ),
    # NOTE: Sacramento is PROPERTY-CRIME-ONLY. California victim-privacy law nulls the
    # coordinates for violent/sex crimes at the source (rape 0%, murder ~11%, assault ~57%
    # geolocated), so only burglary/larceny/mvt/vandal (~99% geolocated) are usable for
    # block-group regression here. Detroit was added as a full-coverage replacement.
    "sacramento": CityConfig(
        name="Sacramento",
        state_fips="06",
        place_fips="64000",
        crime_csv="raw/city_crime/sacramento/Sacramento_Report_Data_2025_7863101653284773738.csv",
        # x/y are California State Plane Zone II (US feet); reprojected to 4326 on load.
        lat_col="y",
        lon_col="x",
        crs="EPSG:2226",
        crime_type_col="offense_code",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="occurrence_date_pt",
        property_only=True,
    ),
    # NOTE: San Francisco is treated PROPERTY-ONLY. DataSF surfaces only ~21 incidents/yr
    # under the "Rape" category (coords present but the count is implausibly low for SF,
    # ~350/yr expected) — sexual assault is effectively not published as mappable rape.
    # Non-rape violent (robbery/assault/murder) is well geolocated.
    "san_francisco": CityConfig(
        name="San Francisco",
        state_fips="06",
        place_fips="67000",
        crime_csv="raw/city_crime/san francisco/Police_Department_Incident_Reports__2018_to_Present_20260818.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="incident_category",
        crime_type_mapping=SF_CATEGORY_TO_CATEGORY,
        date_col="incident_date",
        property_only=True,
    ),
    # NOTE: Pittsburgh is treated PROPERTY-ONLY. The WPRDC feed nulls coordinates for all
    # sex offenses at the source: 301 rape rows (11A-D, 36B) in 2025, 0 geolocated, while
    # overall coverage is ~99%. Non-rape violent is geolocated.
    "pittsburgh": CityConfig(
        name="Pittsburgh",
        state_fips="42",
        place_fips="61000",
        crime_csv="raw/city_crime/pittsburgh/incidents_2024_2026.xlsx",
        # XCOORD/YCOORD are WGS84 lon/lat stored as text.
        lat_col="ycoord",
        lon_col="xcoord",
        crime_type_col="nibrs_offense_code",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="reporteddate",
        property_only=True,
    ),
    # Kansas City, MO. KCPD Socrata export (dmnp-9ajg). One row per person-involvement
    # (VIC/SUS/ARR), so rows are collapsed to one per (report_no, ibrs) offense. Coordinates
    # are WKT `POINT (lon lat)` in a single `Location` column (WGS84). IBRS = NIBRS codes.
    "kansas_city": CityConfig(
        name="Kansas City",
        state_fips="29",
        place_fips="38000",
        crime_csv="raw/city_crime/kansas/KCPD_Crime_Data_2025_20260819.csv",
        lat_col="",
        lon_col="",
        wkt_col="location",
        crime_type_col="ibrs",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="from_date",
        dedup_keys=("report_no", "ibrs"),
    ),
    # Columbus, OH. ArcGIS Hub CSV export (item b70656e5...); x/y = WGS84 lon/lat (populated
    # only when Is_Mapped='Yes'). PROPERTY-ONLY: rape coords 100% suppressed at source.
    # Full 3-year rolling feed; filtered to 2025 via year_filter on occurrence date.
    "columbus": CityConfig(
        name="Columbus",
        state_fips="39",
        place_fips="18000",
        crime_csv="raw/city_crime/columbus/Police_Incident_Reports_20260819.csv",
        lat_col="y",
        lon_col="x",
        crime_type_col="general_subject",
        crime_type_mapping=COLUMBUS_SUBJECT_TO_CATEGORY,
        date_col="occurred_on",
        property_only=True,
    ),
    # Jacksonville, FL (JSO). ArcGIS Hub CSV export (item 29a91fb9...); x/y = WGS84 lon/lat,
    # ~100% populated. PROPERTY-ONLY: rape (11A-11D) absent by Florida Marsy's Law.
    # Full multi-year rolling feed; filtered to 2025 via year_filter on incident date.
    "jacksonville": CityConfig(
        name="Jacksonville",
        state_fips="12",
        place_fips="35000",
        crime_csv="raw/city_crime/jacksonville/JSO_Public_Transparency_20260819.csv",
        lat_col="y",
        lon_col="x",
        crime_type_col="nibrs_code",
        crime_type_mapping=JACKSONVILLE_NIBRS_TO_CATEGORY,
        date_col="incident_date",
        property_only=True,
    ),
    "detroit": CityConfig(
        name="Detroit",
        state_fips="26",
        place_fips="22000",
        crime_csv="raw/city_crime/detroit/RMS_Crime_Incidents.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="offense_category",
        crime_type_mapping=DETROIT_CATEGORY_TO_CATEGORY,
        date_col="incident_occurred_at",
    ),
}

# --- BigQuery projects / datasets ---
BQ_PROJECT         = "clgx-gis-app-dev-06e3"        # billing + GIS/boundary + staging
IDAP_PROJECT       = "clgx-idap-bigquery-prd-a990"  # enterprise property data
BOUNDARY_DATASET   = "boundary"
BQ_STAGING_DATASET = "work_eprashar"                # where feature build tables land

# --- GCS + data tiers ---
GCS_PROJECT = "clgx-gis-app-dev-06e3"
GCS_ROOT    = "gs://geospatial-projects/location_inc"
UCR_YEAR    = 2024

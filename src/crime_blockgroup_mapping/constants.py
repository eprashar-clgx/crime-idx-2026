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

# Seattle (SPD): standard NIBRS codes, but SPD writes the literal string "REDACTED" into
# lat/lon rather than nulling it, and the redaction is *selective by offense*: rape 11A-11D
# is ~98% redacted, murder ~97%, aggravated assault ~30%, robbery ~12%. The ~2% of rape
# rows that survive are a biased, non-random sample, so 11A-11D/36B are removed from the
# mapping entirely (rather than mapped and left to the coord filter, as in other cities).
SEATTLE_NIBRS_TO_CATEGORY = {
    k: v for k, v in NIBRS_TO_CATEGORY.items() if v != 'rape'
}

# Las Vegas (LVMPD): standard NIBRS codes EXCEPT that LVMPD publishes a single bare `11`
# "Sex Offense" bucket instead of 11A-11D, bundling rape with sodomy/fondling/object
# assault. `11` is deliberately absent from NIBRS_TO_CATEGORY, so those rows stay unmapped
# rather than inflating `rape`; the city is PROPERTY-ONLY as a result.
LAS_VEGAS_NIBRS_TO_CATEGORY = {
    k: v for k, v in NIBRS_TO_CATEGORY.items() if v != 'rape'
}

# Philadelphia (PPD `incidents_part1_part2`): text `text_general_code` -> model category.
# Follows the NIBRS-city conventions: only aggravated assault counts as `assault`
# ("Other Assaults" = simple assault, excluded); "Homicide - Criminal" -> `murder`;
# "Theft from Vehicle" -> `larceny` (UCR 23F, not burglary); "Other Sex Offenses" excluded.
PHILADELPHIA_TEXT_TO_CATEGORY = {
    'Homicide - Criminal': 'murder',
    'Rape': 'rape',
    'Robbery Firearm': 'robbery',
    'Robbery No Firearm': 'robbery',
    'Aggravated Assault Firearm': 'assault',
    'Aggravated Assault No Firearm': 'assault',
    'Burglary Residential': 'burglary',
    'Burglary Non-Residential': 'burglary',
    'Thefts': 'larceny',
    'Theft from Vehicle': 'larceny',
    'Motor Vehicle Theft': 'mvt',
    'Vandalism/Criminal Mischief': 'vandal',
    'Arson': 'fire',
}

# Washington DC (MPD): 9-value local `OFFENSE` taxonomy covering Part I crimes only.
# "ASSAULT W/DANGEROUS WEAPON" is DC's aggravated-assault category; "SEX ABUSE" is its
# rape category. DC publishes no vandalism/criminal-mischief category, so `vandal` is
# structurally absent for this city (the `property_count` composite excludes it anyway).
DC_OFFENSE_TO_CATEGORY = {
    'HOMICIDE': 'murder',
    'SEX ABUSE': 'rape',
    'ROBBERY': 'robbery',
    'ASSAULT W/DANGEROUS WEAPON': 'assault',
    'BURGLARY': 'burglary',
    'THEFT/OTHER': 'larceny',
    'THEFT F/AUTO': 'larceny',
    'MOTOR VEHICLE THEFT': 'mvt',
    'ARSON': 'fire',
}

# Oakland (OPD CrimeWatch): local, non-NIBRS `crimetype` buckets that do NOT align with
# UCR and must be read together with the free-text `description` (see OAKLAND_REFINE).
# Key deviations handled here:
#   - "BURG - AUTO" is OPD's term for theft FROM a vehicle -> `larceny` (UCR 23F), NOT
#     `burglary`; only RESIDENTIAL/COMMERCIAL/OTHER burglaries are structure entries.
#   - Vehicle *recovery* rows (RECOVERED ...) are administrative follow-ups to a theft
#     already counted, so they are left unmapped to avoid double counting.
#   - MISDEMEANOR ASSAULT / DOMESTIC VIOLENCE / THREATS are simple assault -> unmapped,
#     consistent with the NIBRS cities mapping only 13A.
OAKLAND_CRIMETYPE_TO_CATEGORY = {
    'HOMICIDE': 'murder',
    'FORCIBLE RAPE': 'rape',
    'ROBBERY': 'robbery',
    'FELONY ASSAULT': 'assault',
    'BURG - RESIDENTIAL': 'burglary',
    'BURG - COMMERCIAL': 'burglary',
    'BURG - OTHER': 'burglary',
    'BURG - AUTO': 'larceny',
    'PETTY THEFT': 'larceny',
    'GRAND THEFT': 'larceny',
    'STOLEN VEHICLE': 'mvt',
    'STOLEN AND RECOVERED VEHICLE': 'mvt',
    'VANDALISM': 'vandal',
    'ARSON': 'fire',
}

# Oakland refinement: three `crimetype` buckets are contaminated badly enough that the
# bucket alone is unusable, so a row only keeps its mapped category when `description`
# also matches. Rows that fail become unmapped. Measured on 2025 (post-dedup):
#   HOMICIDE      754 raw -> ~35   (695 are "SC UNEXPLAINED DEATH", i.e. death investigations)
#   FORCIBLE RAPE 188 raw -> ~183  (drops kidnapping / false-imprisonment misfiles)
#   FELONY ASSAULT 1730 raw -> ~675 (drops 588 negligent-firearm-discharge rows, which have
#                                    no victim and are not UCR aggravated assault, plus
#                                    brandishing/threats/simple battery)
# NOTE: the FELONY ASSAULT pattern is a judgment call and likely *understates* aggravated
# assault, because OPD files many attacks under DOMESTIC VIOLENCE (3,051) and MISDEMEANOR
# ASSAULT (3,647) instead. Revisit if Oakland is used for violent-crime modelling.
OAKLAND_REFINE = {
    'HOMICIDE': r'^MURDER',
    'FORCIBLE RAPE': (r'RAPE|SODOMY|ORAL COP|SEXUAL PENETRATION|SEXUAL BATTERY|'
                      r'PENETRATION W/FOREIGN OBJECT|UNCONSCIOUS OR ASLEEP'),
    'FELONY ASSAULT': (r'ASSAULT WITH FIREARM|ASSAULT WITH CAUSTIC|FORCE/ADW|'
                       r'ADW WITH FORCE|BATTERY W/SERIOUS|SHOOT AT INHABITED'),
}

# New York City (NYPD complaints): `KY_CD` offense key -> model category. One row per
# complaint (top offense only), NY Penal Law categories rather than NIBRS.
# PROPERTY-ONLY, and rape is deliberately NOT mapped: NYPD does not null rape coordinates,
# it *fabricates* them, placing every sex-crime complaint at the precinct station house
# (~78 distinct points; NYPD open-data footnote 10). Unlike a suppressed-coordinate city,
# those rows would survive the coordinate filter and manufacture phantom rape clusters at
# station houses, so KY_CD 104/116/233 are excluded outright.
NYC_KY_CD_TO_CATEGORY = {
    '101': 'murder',
    '105': 'robbery',
    '106': 'assault',
    '107': 'burglary',
    '109': 'larceny',
    '341': 'larceny',
    '110': 'mvt',
    '121': 'vandal',
    '351': 'vandal',
}

# Denver (DPD): local `offense_category_id` slugs -> model category. There is NO rape
# category in this layer at all (sex offences are withheld), hence PROPERTY-ONLY.
# `theft-from-motor-vehicle` is UCR larceny (23F), not burglary or MVT. Denver publishes
# no vandalism category (criminal mischief is folded into `public-disorder`, which also
# carries non-crime disorder rows), so `vandal` is left unmapped.
DENVER_CATEGORY_TO_CATEGORY = {
    'murder': 'murder',
    'robbery': 'robbery',
    'aggravated-assault': 'assault',
    'burglary': 'burglary',
    'larceny': 'larceny',
    'theft-from-motor-vehicle': 'larceny',
    'auto-theft': 'mvt',
    'arson': 'fire',
}

CRIME_CATEGORIES = [
    'assault', 'burglary', 'larceny', 'murder', 'mvt',
    'rape', 'robbery', 'vandal', 'fire',
    'violent', 'property', 'total', 'cl_total', 'wtotal', 'wprop'
]

# National reference rates (per 1,000 RESIDENTS), the `*_pt_u` benchmarks used to turn a
# BG's local per-crime rate into a unitless relative risk (local / national). Canonical
# values from the carrier evals dataset, documented in docs/weightage_methodology.md §2.
# Hardcoded here (not read from the evals parquet) so the weighted-score math is available
# to BOTH tasks without regression_modelling depending on the carrier_eval evals artifact.
# extract_national_rates (scores.py) can still re-derive these from an evals file when present.
NATIONAL_PT_U_RATES = {
    'murder_pt_u':   0.050,
    'rape_pt_u':     0.375,
    'robbery_pt_u':  0.606,
    'assault_pt_u':  2.561,
    'violent_pt_u':  3.591,
    'burglary_pt_u': 2.292,
    'larceny_pt_u':  12.721,
    'mvt_pt_u':      2.588,
    'property_pt_u': 17.601,
}

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
    explode_delim: str = ""                              # split crime_type_col on this delimiter -> one row per offense
    refine_col: str = ""                                 # secondary col used to refine crime_type_col (see refine_map)
    refine_map: dict = field(default_factory=dict, repr=False)  # raw crime type -> regex refine_col must match to keep the mapping
    drop_points: tuple = ()                              # (lat, lon) placeholder coords to discard (source CRS)
    property_only: bool = False                          # True when the source suppresses violent/sex-crime coords (use property targets only)
    sheet_name: object = 0                               # XLSX sheet to read (name or index); default first sheet

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
        crime_csv="raw/city_crime/pittsburgh/pbp_incidents_2024_2026.xlsx",
        # XCOORD/YCOORD are WGS84 lon/lat stored as text.
        lat_col="ycoord",
        lon_col="xcoord",
        crime_type_col="nibrs_offense_code",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="reporteddate",
        property_only=True,
        sheet_name="AllMergedTables",   # Sheet1/Sheet2 are pivot summaries; incidents live here
    ),
    # Kansas City, MO. KCPD Socrata export (dmnp-9ajg). One row per person-involvement
    # (VIC/SUS/ARR), so rows are collapsed to one per (report_no, ibrs) offense. Coordinates
    # are WKT `POINT (lon lat)` in a single `Location` column (WGS84). IBRS = NIBRS codes.
    "kansas_city": CityConfig(
        name="Kansas City",
        state_fips="29",
        place_fips="38000",
        crime_csv="raw/city_crime/kansas/KCPD_Crime_Data_2025_20260916.csv",
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
        crime_csv="raw/city_crime/columbus/Police_Incident_Reports_5614013531510496978.csv",
        lat_col="y",
        lon_col="x",
        crime_type_col="general_subject",
        crime_type_mapping=COLUMBUS_SUBJECT_TO_CATEGORY,
        date_col="occurred_on",
        crs="EPSG:3857",
        property_only=True,
    ),
    # Jacksonville, FL (JSO). ArcGIS Hub CSV export (item 29a91fb9...); x/y = WGS84 lon/lat,
    # ~100% populated. PROPERTY-ONLY: rape (11A-11D) absent by Florida Marsy's Law.
    # Full multi-year rolling feed; filtered to 2025 via year_filter on incident date.
    "jacksonville": CityConfig(
        name="Jacksonville",
        state_fips="12",
        place_fips="35000",
        crime_csv="raw/city_crime/jacksonville/JSO_Public_Transparency_20260916.csv",
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
     # --- 2025 scale-up cohort -------------------------------------------------------
    # Philadelphia, PA. Carto SQL API (phl.carto.com, table `incidents_part1_part2`),
    # already filtered to 2025 server-side. point_x/point_y are WGS84 lon/lat. One row per
    # offence (152,563 rows / 152,552 dc_keys), so no dedup. FULL COVERAGE: rape is
    # geolocated at ~98%, on par with every other offence type.
    "philadelphia": CityConfig(
        name="Philadelphia",
        state_fips="42",
        place_fips="60000",
        crime_csv="raw/city_crime/philadelphia/incidents_2025_20260924.csv",
        lat_col="point_y",
        lon_col="point_x",
        crime_type_col="text_general_code",
        crime_type_mapping=PHILADELPHIA_TEXT_TO_CATEGORY,
        date_col="dispatch_date",
    ),
    # Milwaukee, WI. CKAN WIBR feed (resource 87843297...), full history -> year_filter.
    # One row per *case*, with `Offense_All` holding semicolon-joined NIBRS codes
    # ("13A;13C") on ~20% of rows -> explode_delim makes it one row per offence, matching
    # the other NIBRS cities. MPD geocodes "UNKNOWN" addresses to a single placeholder
    # point (43.195304, -87.854834), which is dropped. FULL COVERAGE: rape ~94% geolocated.
    "milwaukee": CityConfig(
        name="Milwaukee",
        state_fips="55",
        place_fips="53000",
        crime_csv="raw/city_crime/milwaukee/wibr_20260924.csv",
        lat_col="address_latitude",
        lon_col="address_longitude",
        crime_type_col="offense_all",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="incident_date",
        explode_delim=";",
        drop_points=((43.19530359772807, -87.85483384211032),),
    ),
    # Baltimore, MD. ArcGIS Hub item 204beefe... layer 0 (NIBRS Group A), full history ->
    # year_filter. One row per victim, so collapse to one row per (case, offence code).
    # FULL COVERAGE: rape (11A-11C) ~99.7% geolocated. NIBRS reporting starts 2025-01-01.
    "baltimore": CityConfig(
        name="Baltimore",
        state_fips="24",
        place_fips="04000",
        crime_csv="raw/city_crime/baltimore/nibrs_group_a_20260924.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="crimecode",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="crimedatetime",
        dedup_keys=("ccnumber", "crimecode"),
    ),
    # Washington, DC. ArcGIS Hub item 74d924dd... LAYER 7 (not 0), already 2025-only.
    # FULL COVERAGE: MPD rounds every incident to its street-block centroid *uniformly* -
    # SEX ABUSE is anonymised no more than THEFT is, so rape is usable here. One row per
    # CCN (24,153 rows / 24,150 CCNs), so no dedup.
    "dc": CityConfig(
        name="Washington DC",
        state_fips="11",
        place_fips="50000",
        crime_csv="raw/city_crime/dc/crime_incidents_2025_20260924.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="offense",
        crime_type_mapping=DC_OFFENSE_TO_CATEGORY,
        date_col="report_dat",
    ),
    # Oakland, CA. Socrata `ppgh-7dqv` (CrimeWatch), filtered to 2025 server-side.
    # Coordinates arrive as WKT `POINT (lon lat)` in a single `location` column. `crimetype`
    # is functionally determined by `casenumber`, so dedup on (casenumber, crimetype).
    # Local non-NIBRS taxonomy needs OAKLAND_REFINE on `description` - see that map's note.
    # FULL COVERAGE: rape ~90% geolocated (nulls are geocode failures, not suppression).
    "oakland": CityConfig(
        name="Oakland",
        state_fips="06",
        place_fips="53000",
        crime_csv="raw/city_crime/oakland/crimewatch_2025_20260924.csv",
        lat_col="",
        lon_col="",
        wkt_col="location",
        crime_type_col="crimetype",
        crime_type_mapping=OAKLAND_CRIMETYPE_TO_CATEGORY,
        refine_col="description",
        refine_map=OAKLAND_REFINE,
        date_col="datetime",
        dedup_keys=("casenumber", "crimetype"),
    ),
    # New York, NY. Socrata `qgea-i56i` (historic complaints; the YTD feed 5uac-w243 has
    # rolled over to 2026), filtered to 2025 server-side. One row per complaint.
    # PROPERTY-ONLY: NYPD *fabricates* sex-crime coordinates at precinct station houses
    # rather than nulling them, so rape is excluded from NYC_KY_CD_TO_CATEGORY outright.
    "new_york": CityConfig(
        name="New York",
        state_fips="36",
        place_fips="51000",
        crime_csv="raw/city_crime/new_york/nypd_complaints_2025_20260924.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="ky_cd",
        crime_type_mapping=NYC_KY_CD_TO_CATEGORY,
        date_col="cmplnt_fr_dt",
        property_only=True,
    ),
    # Dallas, TX. Socrata `qv6i-rri7`, filtered to 2025 server-side. x_coordinate /
    # y_cordinate (source misspelling) are Texas State Plane North Central (US feet).
    # One row per victim/offence line, so collapse to one row per (incident, NIBRS code).
    # PROPERTY-ONLY: DPD excludes "sexually oriented offenses" from this feed by policy -
    # rape is absent entirely, and juvenile-victim / family-violence cases are also filtered.
    "dallas": CityConfig(
        name="Dallas",
        state_fips="48",
        place_fips="19000",
        crime_csv="raw/city_crime/dallas/police_incidents_2025_20260924.csv",
        lat_col="y_cordinate",
        lon_col="x_coordinate",
        crs="EPSG:2276",
        crime_type_col="nibrs_code",
        crime_type_mapping=NIBRS_TO_CATEGORY,
        date_col="date1",
        dedup_keys=("incidentnum", "nibrs_code"),
        property_only=True,
    ),
    # Denver, CO. ArcGIS item 16d9c82b... LAYER 324, full history -> year_filter.
    # One row per offense_id (60,400 in 2025), so no dedup.
    # PROPERTY-ONLY: the layer carries no rape category at all, and domestic-violence
    # aggravated assault (~33% of all agg assault) is published without coordinates.
    "denver": CityConfig(
        name="Denver",
        state_fips="08",
        place_fips="20000",
        crime_csv="raw/city_crime/denver/crime_20260924.csv",
        lat_col="geo_lat",
        lon_col="geo_lon",
        crime_type_col="offense_category_id",
        crime_type_mapping=DENVER_CATEGORY_TO_CATEGORY,
        date_col="first_occurrence_date",
        property_only=True,
    ),
    # Las Vegas, NV. ArcGIS item c1ec7888... layer 0, full rolling feed -> year_filter.
    # Covers the LVMPD jurisdiction (city + unincorporated Clark County; North Las Vegas
    # and Henderson are separate agencies) - the BG `within_city` flag handles the overspill.
    # `Reported On Date` is a REPORT date, not an occurrence date. Coords ~99.97% present.
    # PROPERTY-ONLY: LVMPD publishes a single bare `11` "Sex Offense" code bundling rape
    # with fondling/sodomy, which is left unmapped rather than inflating `rape`.
    "las_vegas": CityConfig(
        name="Las Vegas",
        state_fips="32",
        place_fips="40000",
        crime_csv="raw/city_crime/las_vegas/lvmpd_nibrs_crimes_20260924.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="nibrs_offense_code",
        crime_type_mapping=LAS_VEGAS_NIBRS_TO_CATEGORY,
        date_col="reported_on_date",
        property_only=True,
    ),
    # Seattle, WA. Socrata `tazs-3rd5` (SPD Crime Data), filtered to 2025 server-side.
    # One row per offense_id, so no dedup. SPD writes the literal string "REDACTED" into
    # lat/lon, which coerces to NaN and is dropped by the coordinate filter.
    # PROPERTY-ONLY, and unusually weak on non-rape violent crime too: murder ~97% redacted,
    # aggravated assault ~30%, robbery ~12%. Property offences are ~99% clean.
    "seattle": CityConfig(
        name="Seattle",
        state_fips="53",
        place_fips="63000",
        crime_csv="raw/city_crime/seattle/spd_crime_2025_20260924.csv",
        lat_col="latitude",
        lon_col="longitude",
        crime_type_col="nibrs_offense_code",
        crime_type_mapping=SEATTLE_NIBRS_TO_CATEGORY,
        date_col="offense_date",
        property_only=True,
    ),

}

# --- BigQuery projects / datasets ---
BQ_PROJECT         = "clgx-gis-app-dev-06e3"        # billing + GIS/boundary + staging
IDAP_PROJECT       = "clgx-idap-bigquery-prd-a990"  # enterprise property data
IMAGERY_PROJECT    = "clgx-idap-iefd-app-prd-ec54"  # Vexcel aerial-features (edr_ent_property_aerial_features)
BOUNDARY_DATASET   = "boundary"
BQ_STAGING_DATASET = "work_eprashar"                # where feature build tables land

# --- GCS + data tiers ---
GCS_PROJECT = "clgx-gis-app-dev-06e3"
GCS_ROOT    = "gs://geospatial-projects/location_inc"
UCR_YEAR    = 2024

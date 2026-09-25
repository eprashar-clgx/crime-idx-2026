"""Unit tests for the CityConfig-driven loader features in crime.py.

These run on synthetic in-memory CSVs, so they need no raw data and are safe to run
anywhere. The data-integrity check against the real (gitignored) city downloads lives
in scripts/validate_cities.py, which is deliberately NOT a test.
"""
import pandas as pd
import pytest

from crime_blockgroup_mapping.constants import CITIES, CityConfig
from crime_blockgroup_mapping.crime import load_crime_data, map_crime_categories

OAK_MAP = {"HOMICIDE": "murder", "FORCIBLE RAPE": "rape", "FELONY ASSAULT": "assault"}
OAK_REFINE = {
    "HOMICIDE": r"^MURDER",
    "FELONY ASSAULT": r"ASSAULT WITH FIREARM|FORCE/ADW",
}


def _cfg(**kw):
    base = dict(
        name="Test", state_fips="00", place_fips="00000", crime_csv="unused.csv",
        lat_col="lat", lon_col="lon", crime_type_col="code",
        crime_type_mapping={"13A": "assault", "220": "burglary", "23F": "larceny"},
        date_col="", year_filter=None,
    )
    base.update(kw)
    return CityConfig(**base)


def _write(tmp_path, rows, cols):
    path = tmp_path / "crime.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)


# --- explode_delim (Milwaukee) ---

def test_explode_delim_splits_multi_offence_cells(tmp_path):
    csv = _write(tmp_path, [["13A;220", 43.0, -87.9], ["23F", 43.1, -87.8]],
                 ["code", "lat", "lon"])
    out = load_crime_data(_cfg(explode_delim=";"), csv_path=csv)
    assert len(out) == 3
    assert sorted(out.code) == ["13A", "220", "23F"]


def test_explode_delim_strips_surrounding_whitespace(tmp_path):
    csv = _write(tmp_path, [["13A; 220", 43.0, -87.9]], ["code", "lat", "lon"])
    out = load_crime_data(_cfg(explode_delim=";"), csv_path=csv)
    assert sorted(out.code) == ["13A", "220"]


def test_multi_offence_cell_stays_intact_without_explode_delim(tmp_path):
    csv = _write(tmp_path, [["13A;220", 43.0, -87.9]], ["code", "lat", "lon"])
    out = load_crime_data(_cfg(), csv_path=csv)
    assert len(out) == 1


# --- drop_points (Milwaukee sentinel) ---

SENTINEL = (43.19530359772807, -87.85483384211032)


def test_drop_points_removes_only_the_sentinel(tmp_path):
    csv = _write(tmp_path,
                 [["220", SENTINEL[0], SENTINEL[1]], ["220", 43.05, -87.95]],
                 ["code", "lat", "lon"])
    out = load_crime_data(_cfg(drop_points=(SENTINEL,)), csv_path=csv)
    assert len(out) == 1
    assert out.iloc[0].lat == pytest.approx(43.05)


def test_drop_points_matches_a_truncated_publication_of_the_same_point(tmp_path):
    # Milwaukee publishes this sentinel at two different precisions.
    csv = _write(tmp_path, [["220", 43.195303597728, -87.85483384211]],
                 ["code", "lat", "lon"])
    out = load_crime_data(_cfg(drop_points=(SENTINEL,)), csv_path=csv)
    assert len(out) == 0


def test_rows_without_coordinates_are_dropped(tmp_path):
    csv = _write(tmp_path, [["220", 43.0, -87.9], ["220", None, None]],
                 ["code", "lat", "lon"])
    out = load_crime_data(_cfg(), csv_path=csv)
    assert len(out) == 1


def test_dedup_prefers_the_row_carrying_coordinates(tmp_path):
    csv = _write(tmp_path,
                 [["A1", "220", None, None], ["A1", "220", 43.0, -87.9]],
                 ["case", "code", "lat", "lon"])
    out = load_crime_data(_cfg(dedup_keys=("case", "code")), csv_path=csv)
    assert len(out) == 1
    assert out.iloc[0].lat == pytest.approx(43.0)


# --- refine_col / refine_map (Oakland) ---

def test_refine_map_keeps_matching_rows_and_unmaps_the_rest():
    df = pd.DataFrame({
        "crimetype": ["HOMICIDE", "HOMICIDE", "FELONY ASSAULT", "FELONY ASSAULT"],
        "description": ["MURDER", "SC UNEXPLAINED DEATH",
                        "ASSAULT WITH FIREARM ON PERSON",
                        "WILLFUL DISCHARGE FIREARM IN NEGLIGENT MANNER"],
    })
    cfg = _cfg(crime_type_col="crimetype", crime_type_mapping=OAK_MAP,
               refine_col="description", refine_map=OAK_REFINE)
    got = map_crime_categories(df, cfg).crime_category
    assert got.iloc[0] == "murder"
    assert pd.isna(got.iloc[1])
    assert got.iloc[2] == "assault"
    assert pd.isna(got.iloc[3])


def test_refine_map_leaves_buckets_it_does_not_mention_alone():
    df = pd.DataFrame({"crimetype": ["FORCIBLE RAPE"], "description": ["ANYTHING"]})
    cfg = _cfg(crime_type_col="crimetype", crime_type_mapping=OAK_MAP,
               refine_col="description", refine_map=OAK_REFINE)
    assert map_crime_categories(df, cfg).crime_category.iloc[0] == "rape"


def test_refine_map_handles_a_null_description():
    df = pd.DataFrame({"crimetype": ["HOMICIDE"], "description": [None]})
    cfg = _cfg(crime_type_col="crimetype", crime_type_mapping=OAK_MAP,
               refine_col="description", refine_map=OAK_REFINE)
    assert pd.isna(map_crime_categories(df, cfg).crime_category.iloc[0])


def test_missing_refine_col_raises():
    df = pd.DataFrame({"crimetype": ["HOMICIDE"]})
    cfg = _cfg(crime_type_col="crimetype", crime_type_mapping=OAK_MAP,
               refine_col="description", refine_map=OAK_REFINE)
    with pytest.raises(KeyError):
        map_crime_categories(df, cfg)


# --- numeric code columns (NYC ky_cd) ---

def test_numeric_code_columns_map_without_a_float_suffix():
    df = pd.DataFrame({"ky_cd": [341, 109, 999]})
    cfg = _cfg(crime_type_col="ky_cd",
               crime_type_mapping={"341": "larceny", "109": "larceny"})
    got = map_crime_categories(df, cfg).crime_category
    assert got.iloc[0] == "larceny"
    assert got.iloc[1] == "larceny"
    assert pd.isna(got.iloc[2])


# --- registry invariants ---

def test_property_only_cities_are_flagged():
    expected = {"sacramento", "san_francisco", "pittsburgh", "columbus", "jacksonville",
                "new_york", "dallas", "denver", "las_vegas", "seattle"}
    assert {k for k, c in CITIES.items() if c.property_only} == expected


def test_rape_stays_unmapped_where_coordinates_are_fabricated():
    """NYC/Las Vegas/Seattle publish misleading non-null rape coordinates.

    Unlike a suppressed-coordinate city, those rows would survive the coordinate
    filter, so rape must be absent from the mapping itself rather than dropped
    downstream. Requires the defensive LAS_VEGAS_NIBRS_TO_CATEGORY.
    """
    for city in ("new_york", "las_vegas", "seattle"):
        assert "rape" not in CITIES[city].crime_type_mapping.values(), city


def test_las_vegas_does_not_map_its_bundled_sex_offense_code():
    # LVMPD publishes a single bare `11` bundling rape with fondling/sodomy.
    assert "11" not in CITIES["las_vegas"].crime_type_mapping


def test_every_city_declares_the_columns_the_loader_needs():
    for key, cfg in CITIES.items():
        assert cfg.date_col, f"{key} has no date_col"
        assert cfg.crime_type_mapping, f"{key} has an empty mapping"
        assert cfg.wkt_col or (cfg.lat_col and cfg.lon_col), f"{key} has no geometry source"
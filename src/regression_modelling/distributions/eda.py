"""Reusable store EDA: parametrized firmographics queries + folium mapping.

A single `StoreQuery` spec (name/brand LIKE patterns + optional NAICS codes) drives
five exploratory pulls against IDAP firmographics:

  naics_code_mix / sic_code_mix  -> discover which NAICS/SIC codes a brand maps to
  store_counts                   -> national counts per business_name
  state_counts                   -> counts per state
  store_points                   -> parcel geometries for a folium map

Add a new store type = one `STORE_QUERIES` entry. No new SQL or Python needed.

NOTE on the filter: NAICS and name predicates are combined as
`{naics} AND ({name_a} OR {name_b} ...)`. This fixes the AND/OR precedence trap
where an unparenthesized OR branch would otherwise bypass the NAICS filter.
"""
from dataclasses import dataclass

import pandas as pd

from regression_modelling.data_wrangling.sources import run_bq_explore

# columns matched against the LIKE patterns
BRAND_COLS = ("business_brand_name", "business_name")


@dataclass(frozen=True)
class StoreQuery:
    name: str
    name_patterns: tuple          # LOWER(col) LIKE these, e.g. ("%7-ele%", "%circle k%")
    naics_codes: tuple = ()        # discovered via naics_code_mix; filters count/state/point pulls


STORE_QUERIES = {
    "convenience_stores": StoreQuery(
        name="convenience_stores",
        name_patterns=("%7-ele%", "%circle k%"),
        naics_codes=("445131",),
    ),
    "liquor_stores": StoreQuery(
        name="liquor_stores",
        name_patterns=("%liquor%",),
    ),
    "gas_stations": StoreQuery(
        name="gas_stations",
        name_patterns=("%gas station%",),
    ),
}


def _name_predicate(patterns, cols=BRAND_COLS) -> str:
    clauses = [f"LOWER({c}) LIKE '{p}'" for p in patterns for c in cols]
    return "(" + " OR ".join(clauses) + ")"


def _naics_predicate(codes) -> str:
    if not codes:
        return "TRUE"
    return "naics_6_digit_primary_code IN (" + ", ".join(f"'{c}'" for c in codes) + ")"


def _resolve(store) -> StoreQuery:
    return store if isinstance(store, StoreQuery) else STORE_QUERIES[store]


def _params(spec: StoreQuery) -> dict:
    return {
        "name_predicate": _name_predicate(spec.name_patterns),
        "naics_predicate": _naics_predicate(spec.naics_codes),
    }


def naics_code_mix(store) -> pd.DataFrame:
    """Which primary NAICS codes does this brand's name match resolve to?"""
    return run_bq_explore("store_naics_codes", **_params(_resolve(store)))


def sic_code_mix(store) -> pd.DataFrame:
    """Which primary SIC codes does this brand's name match resolve to?"""
    return run_bq_explore("store_sic_codes", **_params(_resolve(store)))


def store_counts(store) -> pd.DataFrame:
    """National store counts per business_name (NAICS + name filtered)."""
    return run_bq_explore("store_counts", **_params(_resolve(store)))


def state_counts(store) -> pd.DataFrame:
    """Store counts per state (NAICS + name filtered)."""
    return run_bq_explore("store_state_counts", **_params(_resolve(store)))


def store_points(store) -> pd.DataFrame:
    """Store parcel geometries (WKT) for mapping (NAICS + name filtered)."""
    return run_bq_explore("store_points", **_params(_resolve(store)))

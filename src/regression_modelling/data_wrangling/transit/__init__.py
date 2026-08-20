"""GTFS transit predictors for the regression_modelling task.

Pipeline (see docs/adr/0002-gtfs-transit-ingestion.md, docs/transit_eda_plan.md):

    raw zip(s)                      feeds.read_stop_features / load_city_stops
      -> per-stop features          (span, overnight, trips/day, route types)
      -> colocation                 (nearest-facility distance, H1/H3 flags)
      -> build.build_transit(city)  (sjoin stops -> geoid, aggregate to BG)
      -> data/interim/sources/transit.parquet  (registry cache, backend="file")

Only `build_transit` / `build_all_transit` are meant to be called out-of-band to
materialize the cache; downstream, the `transit` FeatureSource reads it like any other.
"""
from regression_modelling.data_wrangling.transit.build import (
    build_transit, build_all_transit,
)

__all__ = ["build_transit", "build_all_transit"]

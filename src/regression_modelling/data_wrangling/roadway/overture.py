"""Load Overture Maps intersection (connector) points for a city's bounding box.

Chosen instead of raw OpenStreetMap for the permeability metric (R3, exploratory) because
Overture is OSM-lineage but relicensed CDLA-Permissive 2.0 (no ODbL share-alike) — see
docs/features/roadway_plan.md §5. Public, unauthenticated S3 bucket; queried in-process via
DuckDB's httpfs + spatial extensions (no local download of the (huge) global parquet).

Overture's `transportation` theme ships `segment` (roads) and `connector` (topology nodes)
as separate GeoParquet types. A connector's "degree" (how many distinct road segments meet
there) isn't a native column — we derive it by unnesting each segment's `connectors` array
and counting distinct segment ids per connector_id. degree>=3 = a real junction (degree 1 =
dead end, degree 2 = a mid-block vertex/curve, not an intersection).
"""
from __future__ import annotations

import geopandas as gpd

from regression_modelling.config import roadway_overture_parquet

# Pinned release for reproducibility (bump deliberately; check availability first with
# `SELECT * FROM (SHOW ...)` style bucket listing, e.g. via the AWS CLI or a HEAD request,
# since not every dated prefix under s3://overturemaps-us-west-2/release/ is guaranteed to
# stay live indefinitely).
OVERTURE_RELEASE = "2026-09-23.0"
_BASE = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=transportation"
_SEGMENT_PATH = f"{_BASE}/type=segment/*"
_CONNECTOR_PATH = f"{_BASE}/type=connector/*"

# A connector must be referenced by this many distinct road segments to count as a "real"
# intersection (excludes dead ends [1] and simple pass-through vertices [2]).
MIN_JUNCTION_DEGREE = 3

_EQUAL_AREA_CRS = "EPSG:5070"


def _connect():
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")
    con.execute("SET s3_region='us-west-2';")
    return con


def load_city_intersections(city: str, bbox: tuple[float, float, float, float],
                            refresh: bool = False) -> gpd.GeoDataFrame:
    """Junction points (degree >= MIN_JUNCTION_DEGREE) within a city's bbox, in EPSG:5070.

    ``bbox`` = (xmin, ymin, xmax, ymax) in EPSG:4326 (e.g. a city boundary's total_bounds).
    Cached to data/interim/roadway/overture/{city}.parquet.
    """
    cache = roadway_overture_parquet(city)
    if cache.exists() and not refresh:
        return gpd.read_parquet(cache)

    xmin, ymin, xmax, ymax = bbox
    con = _connect()
    query = f"""
        WITH seg AS (
            SELECT id AS segment_id, unnest(connectors).connector_id AS connector_id
            FROM read_parquet('{_SEGMENT_PATH}', hive_partitioning=1)
            WHERE subtype = 'road'
              AND bbox.xmin <= {xmax} AND bbox.xmax >= {xmin}
              AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}
        ),
        deg AS (
            SELECT connector_id, COUNT(DISTINCT segment_id) AS degree
            FROM seg GROUP BY connector_id
        )
        SELECT c.id AS connector_id, d.degree,
               ST_X(c.geometry) AS lon, ST_Y(c.geometry) AS lat
        FROM read_parquet('{_CONNECTOR_PATH}', hive_partitioning=1) c
        JOIN deg d ON c.id = d.connector_id
        WHERE d.degree >= {MIN_JUNCTION_DEGREE}
          AND c.bbox.xmin <= {xmax} AND c.bbox.xmax >= {xmin}
          AND c.bbox.ymin <= {ymax} AND c.bbox.ymax >= {ymin}
    """
    df = con.execute(query).fetch_df()
    con.close()

    geom = gpd.points_from_xy(df["lon"], df["lat"])
    out = gpd.GeoDataFrame(df[["connector_id", "degree"]], geometry=geom, crs="EPSG:4326")
    out = out.to_crs(_EQUAL_AREA_CRS)

    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"roadway[overture]: {city} -> {len(out):,} junctions (degree>={MIN_JUNCTION_DEGREE}) -> {cache}")
    return out

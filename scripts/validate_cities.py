"""Data-integrity check for the city crime sources.

Loads every configured city's raw download, maps categories, and prints a per-city
count table. This needs the gitignored files under data/raw/city_crime/, so it is a
script rather than a test: run it after adding a city or refreshing a download.

    poetry run python scripts/validate_cities.py            # all cities
    poetry run python scripts/validate_cities.py denver dc  # a subset
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from crime_blockgroup_mapping.config import DATA_DIR              # noqa: E402
from crime_blockgroup_mapping.constants import CITIES             # noqa: E402
from crime_blockgroup_mapping.crime import (                      # noqa: E402
    load_crime_data, map_crime_categories)

CATEGORIES = ("murder", "rape", "robbery", "assault",
              "burglary", "larceny", "mvt", "vandal", "fire")

# Detroit is the regression canary: it uses none of the newer loader features, so a
# change in its row count means a shared code path moved.
REGRESSION_BASELINE = {"detroit": 85365}


def check(key):
    cfg = CITIES[key]
    print("=" * 72)
    print(f"{key.upper()}  ({'property-only' if cfg.property_only else 'full coverage'})")
    if not (DATA_DIR / cfg.crime_csv).exists():
        print(f"  SKIP: missing {cfg.crime_csv}")
        return None
    gdf = load_crime_data(cfg)
    mapped = map_crime_categories(gdf, cfg)
    counts = mapped.crime_category.value_counts()

    problems = []
    if key in REGRESSION_BASELINE and len(gdf) != REGRESSION_BASELINE[key]:
        problems.append(f"regression: {len(gdf):,} rows != {REGRESSION_BASELINE[key]:,}")
    if not cfg.bg_zip.exists():
        problems.append(f"missing boundary file {cfg.bg_zip.name}")
    for need in ("larceny", "burglary", "mvt"):
        if counts.get(need, 0) == 0:
            problems.append(f"no {need} rows")
    if not cfg.property_only:
        for need in ("murder", "rape", "robbery", "assault"):
            if counts.get(need, 0) == 0:
                problems.append(f"full-coverage city has no {need} rows")
    return {"key": key, "geo": len(gdf), "mapped": int(counts.sum()),
            "counts": {c: int(counts.get(c, 0)) for c in CATEGORIES},
            "problems": problems}


def main(keys):
    unknown = [k for k in keys if k not in CITIES]
    if unknown:
        sys.exit(f"unknown cities: {unknown}\nknown: {sorted(CITIES)}")

    results = [r for r in (check(k) for k in keys) if r]

    print("\n" + "=" * 72)
    print(f"{'city':<15}{'geocoded':>10}{'mapped':>10}   " +
          " ".join(f"{c[:5]:>7}" for c in CATEGORIES))
    print("-" * 72)
    for r in results:
        print(f"{r['key']:<15}{r['geo']:>10,}{r['mapped']:>10,}   " +
              " ".join(f"{r['counts'][c]:>7,}" for c in CATEGORIES))

    failed = [r for r in results if r["problems"]]
    if failed:
        print("\nPROBLEMS")
        for r in failed:
            for p in r["problems"]:
                print(f"  {r['key']}: {p}")
        sys.exit(1)
    print(f"\nOK - {len(results)} cities validated")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CITIES))
# ADR 0001 — Task-oriented module restructure

- **Status:** Accepted
- **Date:** 2026-08-12
- **Context skill:** improve-codebase-architecture (grill-with-docs)

## Context

`src/` was organized as `core/` (shared geo + config) + `prediction/` (the model
pipeline) + loose `eval_utils.py` / `plot_utils.py`. This has three problems:

1. **Ambiguous names.** `core` and `prediction` say nothing about purpose and
   collect unrelated logic.
2. **A shared surface with no home.** `geo_utils.py` mixes genuinely shared
   block-group assembly with population/rate normalization; both the model pipeline
   and the carrier-eval side-quest reach into it. `eval_utils.py` even imports
   `src.core.config` — the only `src.`-rooted import in the repo, a seam leak.
3. **The side-quest is entangled.** Carrier evaluation of the *existing* model has no
   dedicated space, so its score-reconstruction math sits in shared/loose files.

The repo's real unit of work is a **statistical task** (build the POC model; evaluate
the existing model). Structure should follow that.

## Decision

Reorganize `src/` as **one folder per statistical task**, with a single shared
foundation module. No ambiguous names.

- **`crime_blockgroup_mapping`** (shared) — city registry + crime taxonomy,
  block-group crime assembly, population/rate normalization, and general plot
  helpers. Replaces `core`. The single module both tasks import from.
- **`regression_modelling`** — POC model rebuild. Sub-modules: `data_wrangling`,
  `feature_engineering`, `distributions`, `models` (incl. Moran's I diagnostic),
  `bias_testing`, `logging`.
- **`carrier_eval`** — side-quest evaluating the existing model; owns evals ingestion
  and the score-reconstruction math (`compute_weighted_scores`,
  `extract_national_rates`).

Cross-cutting conventions:
- `config.py` holds **paths only**; a per-module `constants.py` holds constants,
  dataclasses, and registries.
- **Notebooks** live outside `src/` in `notebooks/`, mirroring the module tree.
- **Data** stays outside `src/`, documented in a living `data/README.md` catalog.

Scope split of today's shared surface (groups):
- **A** city registry + crime taxonomy → `crime_blockgroup_mapping`
- **B** BG-crime assembly → `crime_blockgroup_mapping`
- **C** population + rate normalization → `crime_blockgroup_mapping`
- **D** score reconstruction math → `carrier_eval`

Cleanups folded in: fix the `src.core.config` import leak; delete dead
`sjoin_crimes_to_city` and `aggregate_by_bg` (deletion test — unused).

## Consequences

**Positive**
- Each module has one goal; names describe purpose.
- One clear seam: both tasks depend only on `crime_blockgroup_mapping`, not on each
  other.
- The side-quest is quarantined; future side-quests get peer folders.
- Locality: constants and paths are predictable per module.

**Negative / costs**
- A large mechanical refactor: every `core.*` / `prediction.*` import (in `src/` and
  notebooks) must be rewritten. No test suite exists, so validation is
  `py_compile` + import checks + running pipeline functions.
- `crime_blockgroup_mapping` carries plotting helpers, slightly beyond "build the BG
  crime table"; accepted to avoid duplicating base map code across tasks.

## Open items (not decided here)

- Whether `regression_modelling`'s population source should stop depending on the
  existing NeighborhoodScout `.sav` (group C's `load_model_data` is used only to
  obtain `population`). Flagged for a later ADR.
- Detailed POI/facility ingestion re-architecture (single `FacilitySpec`, generic
  build, BG-count and radius steps) — a follow-on task within `data_wrangling`.

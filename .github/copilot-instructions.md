# Copilot instructions — crime-idx-2026

Proof of concept for a **block-group (BG) crime-risk model** for insurance. It predicts
weighted crime rates per BG from national predictors, is evaluated **leave-one-city-out
(LOCO)** on city incident data (20 cities for property crime `wprop`, 10 for total crime
`wtotal`), and is benchmarked against the incumbent agency-scale model.

Read before non-trivial work:
- `docs/CONTEXT.md`: glossary and metric vocabulary. **Names there are load-bearing.**
- `docs/architecture.md`: module map, data flow, and what each `models/` file owns.
- `docs/adr/`: decisions (0003 CV protocol, 0004 bias testing, 0007 target and level
  strategy, 0008 module boundaries, 0010 predictor selection).
- `docs/features/*.md`: feature definitions, hypotheses, backlog.

## Work tracking

- **Open work lives in GitHub Issues** (`gh issue list`), labelled `modelling`,
  `predictors`, `data-quality`, `compliance` and `parked`. Check open issues at session start.
- Log new follow-ups as issues (`gh issue create`). Don't leave them only in chat or in session todos.
- Reference issues as `#N` in commits; close them from the commit or PR that finishes them.

## Environment and commands

- Python ≥3.12, Poetry. If needed: `export PATH="/snap/bin:$HOME/.local/bin:$PATH"`.
- Tests: `poetry run pytest -q` (`tests/conftest.py` puts `src/` on the path).
- Scripts: `PYTHONPATH=src poetry run python path/to/script.py`.
- Execute a notebook **from its own directory**:
  `poetry run jupyter nbconvert --to notebook --execute --inplace NB.ipynb --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=5400`.
  LOCO notebooks are slow (`02` ~8 min, `05` ~18 min), so don't re-execute for markdown-only edits.
- BigQuery and GCS are read through `regression_modelling.data_wrangling`. Pulls cache to
  Parquet under `data/interim` (`refresh=False` by default).

## Layout and conventions

- `src/` has **one module per statistical task**, plus one shared foundation:
  - `crime_blockgroup_mapping`: the shared foundation (cities, crime taxonomy, BG counts and rates, weighted scores);
  - `regression_modelling`: the POC model;
  - `carrier_eval`: a side-quest.
  - No task→task imports. No `core` / `utils` / `prediction` modules.
- `config.py` holds paths only. Constants, dataclasses and registries live in each module's `constants.py`.
- **Notebooks are drivers.** Reusable logic goes in `src/`. Notebooks live in `notebooks/`,
  mirroring the module tree. Figures are saved to `docs/images/<notebook-number>/`.
- Join key: the 12-char `geoid`. Normalize to it before any merge.
- SQL lives under `regression_modelling/data_wrangling/sql/{build,pull,explore}/` and is templated with
  `{bq_project}`, `{idap_project}` and similar placeholders. Sources are declared in the `FEATURE_SOURCES` registry.
- Predictors: model-form columns come from `*_MODEL_TRANSFORMS` (`{col}_log` etc.).
  Named sets are resolved by `predictor_set(name, target)` (current: `selected_v1`; a `+agency` suffix adds
  the lagged agency anchor). Notebooks refer to sets by name, never by inline lists.
- City pools come only from `target_pool(target)`. Never hardcode city counts.
- `models/` split (ADR 0008):
  - `dataset` → `training` (`run_cv`, estimators, `TUNED_GBM_PARAMS`) → `results.FoldRun`;
  - `metrics`, `diagnostics`, `attribution`, `levelling`, `figures` read a `FoldRun`;
  - `tuning.run_loco_parallel` is the fast LOCO.
  - Keep `metrics` free of `training` imports.
- Edit notebooks with `nbformat` or format-preserving JSON edits to avoid huge diffs.

## Domain rules (do not violate)

- **Never use jobs or employment predictors** (LODES `c000`, jobs counts or ratios). A state DOI rejected them.
- **Never use protected attributes or close proxies** as predictors. Protected attributes
  live only in `bias_testing` (ADR 0004). Bias checks soft-flag predictors; they never drop them automatically.
- **No OSM / Overture / ODbL data.** Use public-domain or attribution-only sources only.
- 7 primary crimes. Composites exclude vandalism. A weighted score is an equal-representation average of
  relative risks vs national `*_pt_u` rates (`docs/weightage_methodology.md`).
- Target: `log1p` of the weighted rate per 1,000 residents. Zero/NaN-population BGs are dropped.
- Evaluation is LOCO (random k-fold leaks spatially). Headline metrics:
  - `r2_pooled` (all held-out BGs);
  - `r2_city` (equal-weight per-city mean R²);
  - within-city Spearman;
  - `recall@10` (within city; chance = 0.10).
  - A correlation is never called "R²".
- `*_lag6` columns are **spatial** lags (pooled over the 6 nearest neighbouring BGs), **not**
  time lags. `_nbr` columns are adjacent-BG means or sums.
- Check definitions in code or docs; don't infer them from column names.

## Working style

- Discuss requirements before large implementations. Keep explanations concise.
- Commit only when asked. Include the `Co-authored-by: Copilot` trailer.
- Tests: a few focused cases in `tests/` for new behaviour. Run the full suite before committing.
- Update `docs/architecture.md`, `docs/CONTEXT.md` or an ADR when a module, concept or decision changes.

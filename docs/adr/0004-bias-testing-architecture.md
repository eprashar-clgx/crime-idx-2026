# ADR 0004 — Bias-testing architecture: separate protected-attribute table, conditional-association diagnostic

- **Status:** Accepted
- **Date:** 2026-08-27
- **Related:** ADR 0001 (`bias_testing` sub-module), ADR 0003 (filtered geoid set), `docs/eda_plan.md`

## Context

`bias_testing` exists to check that predictors correlate with **crime** and **not** with
**protected attributes** (e.g. race). We are importing a batch of new ACS columns, some of
which are legitimate predictors and some of which are **protected-attribute flags** used
only for this check. Two design questions fall out:

1. **Where do protected attributes live** so they can never leak into the model as
   predictors?
2. **What is the actual test**, given that in US urban data residential segregation makes
   **almost every** predictor correlate with race to some degree — a naive "drop if
   `|corr(X, race)| > t`" rule would nuke nearly every feature and is the wrong instrument.

## Decision

### Protected attributes live in a separate bias-testing-only table (Approach 2)

- ACS columns are split at ingestion: **predictors** flow through the normal registry
  (`FEATURE_SOURCES` → `assemble_features` → `PREDICTOR_COLS`); **protected-attribute
  flags** go to a **separate bias-testing table** owned by `bias_testing` (e.g. cached
  under `data/interim/bias/`), keyed by `geoid`.
- Protected attributes **never** enter `FEATURE_SOURCES`, `assemble_features`, or
  `PREDICTOR_COLS`. Leakage is impossible **by construction**, not by discipline.
- A `PROTECTED_ATTRIBUTES` list is **owned by `bias_testing`**, not the shared predictor
  registry.
- At test time the bias table is joined by `geoid` to the model's **filtered** geoid set
  (ADR 0003) — the check runs on exactly the fitted population, not the full boundary set.

### The test is conditional association, reported for human review

For each candidate predictor `X` and protected attribute `Z`, report **both**:

1. **Raw `corr(X, Z)`** — descriptive flag of shared variance.
2. **Whether X's crime signal survives conditioning on Z** — the partial/conditional
   association of `X` with crime after partialling out `Z` (computed inside the
   bias-testing table, so `Z` never touches the fit matrix). The real question is not "is
   `X` correlated with race?" but "**is X's crime signal just a proxy for race, or does it
   survive conditioning on race?**"

**Action: soft-flag for documented human decision — never an automatic drop.** A predictor
whose crime signal collapses under conditioning, or whose raw correlation is high, is
**flagged** for a written human call: keep-with-caveat / drop / investigate. No predictor
is dropped automatically. Auto-dropping on raw correlation is statistically naive
(segregation); auto-orthogonalizing (residualizing `X` against `Z`) hides legitimate
signal and sacrifices interpretability — both were considered and rejected.

## Consequences

- `bias_testing` gains its own data path (read ACS protected columns → own `geoid`-keyed
  cache → join to model results/filtered geoids at test time) and a `PROTECTED_ATTRIBUTES`
  registry, kept out of the predictor registry.
- ACS ingestion (Step 1) must **route** columns to the predictor path vs the bias-only path
  based on the mixed classification.
- Bias output is a **report** (raw corr + conditional association + flag), consumed by a
  human, consistent with the VIF/correlation diagnostic philosophy in ADR 0003.

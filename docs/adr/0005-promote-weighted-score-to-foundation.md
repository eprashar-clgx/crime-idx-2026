# ADR 0005 — Promote weighted-score math into the shared foundation

- **Status:** Accepted
- **Date:** 2026-08-27
- **Related:** ADR 0001 (task seam), ADR 0003 (weighted rate as a regression comparator),
  `docs/weightage_methodology.md`

## Context

ADR 0003 adds the **weighted crime rate** (equal-representation average of relative risks:
local rate ÷ national `*_pt_u` rate) as a regression comparator target. That math currently
lives in **`carrier_eval`** — `compute_weighted_scores` and `extract_national_rates`
(`src/carrier_eval/scores.py`).

The architecture's core seam (ADR 0001) is that both tasks depend **only** on
`crime_blockgroup_mapping`, with **no task→task dependency**. Importing `carrier_eval` into
`regression_modelling` to reuse the weighted-score math would couple the two tasks and
break that seam.

## Decision

**Promote the weighted-score / relative-risk math into the shared foundation
`crime_blockgroup_mapping`.**

- Move `compute_weighted_scores`, `extract_national_rates`, and the national `*_pt_u` rate
  handling out of `carrier_eval.scores` and into `crime_blockgroup_mapping` (alongside the
  existing per-1,000 rate normalization — this is task-agnostic crime math and belongs with
  the rate logic).
- **`carrier_eval` imports it from the foundation** (behaviour unchanged), and
  **`regression_modelling` imports the same function** for the weighted-rate comparator
  target. No task→task coupling; the seam holds.
- The choice follows the ADR 0001 rule: *"new shared, task-agnostic logic goes to the
  foundation."* Relative-risk normalization against national rates is exactly that.

Alternatives rejected: **duplicate** the logic into `regression_modelling` (two copies to
keep in sync); **import from `carrier_eval`** (violates the seam); **defer** the weighted
rate (viable, but the user chose to keep it in scope this round).

## Consequences

- `crime_blockgroup_mapping` grows a weighted-score/relative-risk surface; `carrier_eval`
  shrinks to carrier-specific ingestion + plots and re-imports the moved functions.
- The glossary's **weighted score** noun now points at the foundation, not `carrier_eval`.
- `carrier_eval` regression tests / notebooks that import `compute_weighted_scores` /
  `extract_national_rates` must update their import path.

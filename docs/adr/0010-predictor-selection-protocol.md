# ADR 0010 — Predictor selection protocol

- **Status:** Accepted (2026-09)
- **Date:** 2026-09-30
- **Related:** ADR 0003 (LOCO as the extrapolation protocol), ADR 0007 (lagged-agency
  anchor), ADR 0008 (module boundaries; notebooks are drivers)

## Context

Scaling to 20 cities added candidate predictor families (risky-facility transit, roadway)
and more are coming from feature engineering. A first A/B showed the gaps between sets are
0.01–0.02 of `r2_oos` from single LOCO runs — noise-sized — and that a family can look
harmful without the agency anchor yet help with it (roadway partly encodes city level).
Pooled metrics are ~30% New York by row count, so a pooled gain can hide losses in most
cities. We needed a repeatable way to choose predictors that does not wreck the reporting
notebook (`02`) or overfit the LOCO folds.

## Decision

1. **Screen, then select.** Model-free *screening rules* (weak, redundant, coverage
   artifact, sparse — every threshold an explicit, tunable argument) cut the candidate list
   to a reviewed shortlist; *stepwise selection* runs only on the survivors.
2. **Ridge drives selection; LightGBM confirms.** Stepwise runs forward and backward with
   Ridge (cheap; agreement between directions is the confidence signal). LightGBM is then
   fit on the screened vs Ridge-selected set; if the smaller set costs it less than the drop
   tolerance, it is adopted as the **one** set for both targets. Otherwise both are promoted
   and the gap reported.
3. **Paired, per-city stop rule.** A drop is allowed only if pooled `r2_oos` falls by
   < 0.005 **and** `recall@10` is not worse in at least half of held-out cities. Thresholds
   are fixed before looking at results.
4. **Fixed during selection:** the target-paired agency anchor (always in — without it,
   level-confounded predictors are misjudged) and default LightGBM params with a fixed seed.
   The incumbent's division dummies are retired (they encode 3 of 9 divisions, and the
   anchor already carries city level).
5. **Select on `wprop`, confirm on `wtotal`.** The `wtotal` pool is too small to select on.
6. **Selection bias is accepted and disclosed, not removed.** No lockbox cities: holding
   out ~4 would shrink the `wtotal` pool from 10 to ~6. Bias is bounded by making few
   decisions (a screened shortlist, not the full feature space).
7. **Where it lives.** Experiments run in `02a_predictor_selection`; `02` refers to
   **promoted** predictor sets by name only. Every selection/tuning run is appended to an
   **experiment log** storing a summary row plus the run's out-of-sample predictions, so
   per-city and future metrics are recomputable without refitting. `run_cv` stays unaware
   of the log.
8. **After the freeze:** add XGBoost as a comparison estimator, rerun `02`, then tune the
   preferred model with Optuna on `r2_oos` subject to `recall@10` ≥ the untuned model.

## Considered options

- **Lockbox cities / nested CV** — rejected: the `wtotal` pool cannot spare cities, and
  nested CV multiplies an already-large run count.
- **Family-level ablation only** — too coarse; **per-feature stepwise on everything** —
  too many decisions (selection bias). Screening first is the compromise.
- **Stepwise with LightGBM** — ~20× slower and trees already down-weight useless features;
  its role is confirmation, not search.
- **Separate sets per model or per target** — doubles everything downstream; only adopted
  if LightGBM or `wtotal` clearly disagrees.

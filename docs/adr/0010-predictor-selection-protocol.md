# ADR 0010 — Predictor selection protocol

- **Status:** Accepted (2026-09); amended 2026-10 (Amendment 1)
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

## Amendment 1 — what selection actually did (2026-10)

Running the protocol on the 36 screened candidates (20 `wprop` cities) changed four of the
decisions above. The promoted result is `PREDICTOR_SETS["selected_v1"]` (34 features +
anchor) with `TUNED_GBM_PARAMS["selected_v1"]`, reported in `02_regression_prediction`.

1. **LightGBM drives selection; Ridge is the cross-check** (replaces §2). Ridge backward kept
   11 features and disagreed with LightGBM on which mattered (it kept `det_pct`, `own_pct_nbr`,
   the interstate distance). The deployed model is a GBM, so the GBM's own backward path
   decides. LOCO folds run in parallel (`tuning.run_loco_parallel`).
2. **Cumulative-drift guard** (extends §3). `StopRule.cum_tol` refuses a drop that takes the
   score below best-so-far − `cum_tol` (0.005). Without it, a run of individually tolerable
   drops can quietly erode the score.
3. **Score on city-mean, not pooled** (`StopRule.score`, `paired_delta(score=)`; amends §3).
   New York is ~32% of rows, and its BGs sit at the 86th percentile of the other cities'
   intersection density. Pooled-scored selection (34 → 14) reached pooled r² 0.451 but lost in
   15 of 20 cities (Seattle −0.15). City-mean r² (each held-out city weighted equally) asks the
   deployment question: how well do we rank a typical new city. Both r² are reported everywhere.
4. **Promote the screened set, not a stepwise subset** (amends §2's adoption rule).
   City-mean backward selection (36 → 22) matched the full set within noise on `wprop`
   (city-mean r² 0.357 vs 0.354) but failed the `wtotal` confirmation (§5); a rerun on 34
   kept 18, again with no clear gain. The
   screened set, minus `det_pct`/`det_pct_nbr` (collinear with `own_pct`/`lap_pct`; dropped by
   review), is promoted as one set for both targets. With few decisions taken, this also keeps
   selection bias (§6) small.
5. **Estimator and tuning** (resolves §8). Untuned XGBoost tied LightGBM (pooled r² 0.449 vs
   0.449), so LightGBM stays. Optuna (50 TPE trials, `tuning.tune`) maximises **city-mean**
   r² subject to `recall@10` ≥ the untuned model. A pooled objective won pooled r² (0.443
   vs 0.433) by fitting New York (r² 0.25 vs 0.19) at the other 19 cities' expense (mean 0.351
   vs 0.367), and was worse on `wtotal` (city-mean 0.309 vs 0.323). The city-mean tune lifted city-mean r² on both targets (`wprop` 0.342 → 0.358,
   `wtotal` 0.299 → 0.323). Its parameters are regularisers (bagging, L2 = 18, slower
   learning rate), so LOCO gains with 80/20 flat.
6. **Tried and parked:** within-city percentile roadway features (+0.005 city-mean r²,
   mostly New York; adds a city-relative transform at scoring time) and a vacancy ×
   risky-transit interaction (flat on `wprop`). Pre-screen exclusions stand: liens,
   `veh0_pct`, ramp count.

## Considered options

- **Lockbox cities / nested CV** — rejected: the `wtotal` pool cannot spare cities, and
  nested CV multiplies an already-large run count.
- **Family-level ablation only** — too coarse; **per-feature stepwise on everything** —
  too many decisions (selection bias). Screening first is the compromise.
- **Stepwise with LightGBM** — ~20× slower and trees already down-weight useless features;
  its role is confirmation, not search.
- **Separate sets per model or per target** — doubles everything downstream; only adopted
  if LightGBM or `wtotal` clearly disagrees.

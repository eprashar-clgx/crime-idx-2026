# ADR 0009 — Deployment artifacts deferred, with an explicit trigger

- **Status:** Accepted (2026-09)
- **Date:** 2026-09-23
- **Related:** ADR 0008 (modelling module boundaries), ADR 0003 (LOCO as the
  extrapolation protocol), `src/regression_modelling/constants.py` (GTFS vs ACS transit
  variants), `src/regression_modelling/models/experiments.py`

## Context

A common ML repo layout separates `training/` (split protocols, metrics, evals) from
`prediction/`, where a `prediction/artifacts/` folder holds serialized models (`.pkl`),
feature configuration (`.json`), and the scripts that turn them into scored rows. The
question raised: should this repo adopt that split now, as it scales from 5/9 cities to
15-20?

Two facts about the current state decided this.

**1. Nothing is serialized, anywhere.** There is no `pickle`, `joblib`, `.pkl` or JSON
dump in `src/`. Every model this repo has produced has been discarded at kernel restart.

**2. There is no "the model" to serialize.** LOCO produces one model per held-out city
(5-9 of them) and `run_holdout` produces one fit on 80% of rows. Neither is a deployable
artifact, and the thing that would be — a single fit over every eligible block group — is
never fit anywhere. The gap is not a missing folder; it is a missing concept.

The proposed layout also overlaps almost entirely with ADR 0008:

| external layout | this repo (ADR 0008) |
|---|---|
| `training/` (80-20, LOCO) | `harness` |
| `metrics` | `scorecard` |
| `evals` (feature importance, gains by decile) | `diagnostics` |
| `prediction/artifacts/` | nothing — genuinely new |

So adopting it would add exactly one concept: the **artifact seam**.

### "Predict never-seen-before block groups" is three different asks

The phrase is ambiguous and the three readings have very different costs:

1. **Unseen BGs inside a city we trained on.** Already supported — this is the stratified
   80/20 interpolation protocol (ADR 0003).
2. **An unseen city among the cities we have ingested.** Already measured — this is
   exactly what LOCO reports, and it is the honest headline.
3. **A city we hold no crime data for.** It can never appear in training or evaluation.
   This is the deployment case, and the only one that needs an artifact.

Case 3 carries a constraint that is easy to miss: **LOCO already simulates it correctly**,
so the evaluation design needs no change — but the *predictor set* does. The default
fit-set (`PREDICTOR_COLS`) uses GTFS transit predictors, which require a transit feed
ingested per city. In a city we have not ingested, those columns do not exist, so the
model behind the current headline numbers cannot physically score there.

The deployable variant already exists (`ACS_TRANSIT_PREDICTOR_COLS`: commute-by-transit
share and zero-vehicle households, available for every block group nationally). The only
spec that exercises it lives in `experiments.poc_specs()`, which currently has no callers,
so **the deployable model's accuracy is presently unmeasured**.

## Decision

**Defer the training/prediction split and the artifact seam. Do not create a
`prediction/` module or an `artifacts/` folder now.**

Reasons:

- **One consumer is a hypothetical seam.** There is no caller that needs a serialized
  model today. Building the seam before the second consumer exists is how shallow
  modules get created.
- **The predictor set is still moving.** It changed twice in the past week (DOI-approved
  features, then the lagged-agency column). An artifact freezes a contract; freezing a
  contract that is still moving buys churn, not stability.
- **Scale-up does not trigger it.** Going to 15-20 cities is more folds through the same
  training loop. It produces no artifact and needs no prediction path.

### Trigger

Revisit this ADR when **either** of the following becomes true:

1. Someone asks for scored block groups in a city that holds no crime data (case 3
   above) — whether nationwide or a single city.
2. A model needs to outlive the process that fit it, for example to be handed to another
   team or re-scored without refitting.

Until then, LOCO metrics on the deployable predictor set are the product claim, and they
are produced entirely within the training path.

### What we do now instead, because it is free

**`FoldRun` must carry everything a prediction needs, not just the scored rows** (ADR
0008, landing 2). Today the fitted scaler is created inside `fit_fold` and dies with it,
so even a pickled estimator could not reproduce a prediction: the scaler, the resolved
predictor list, and the transform specs would all be missing. If `FoldRun` holds them,
an artifact later is *serialization of something that already exists* rather than a
retrofit against in-memory assumptions.

**Measure the deployable variant.** Run the ACS-transit predictor set through the same
LOCO ladder as the GTFS set and report both, so the gap between "best fit" and
"deployable" is a known number rather than a surprise discovered at deployment time.

## Consequences

- No `prediction/` module, no `artifacts/` folder, and no serialization code for now.
- The repo keeps a single path: data to fit to metrics. Simpler to navigate and test.
- Landing 2 designs `FoldRun` to be artifact-ready without building the artifact.
- The GTFS/ACS gap becomes a reported number, which also makes the deployability
  constraint visible to anyone reading the results table.
- If the trigger fires mid-POC, the work is additive (serialize `FoldRun`, add a scoring
  entry point) rather than a restructure — which is the point of deferring rather than
  refusing.

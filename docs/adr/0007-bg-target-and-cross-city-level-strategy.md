# ADR 0007 — City-incident block-group target and the cross-city level strategy

- **Status:** Accepted (2026-09)
- **Date:** 2026-09-24
- **Related:** ADR 0003 (regression reframe, pooled/grouped CV, spatial control
  deferred), ADR 0006 (log-scale population ring),
  `notebooks/regression_modelling/models/04_bg_comparison.ipynb` (scale case),
  `02_regression_inference.ipynb`, `02_regression_prediction.ipynb`,
  `src/regression_modelling/models/cv.py`

## Context

The refreshed model is trained at the **block-group (BG) level on city-incident
crime**, targeting the within-city log crime rate rather than an absolute national
rate. Two questions this ADR settles:

1. **Why is the modelling unit a city-incident BG target (and why demean within
   city)?**
2. **Given that the target is within-city, how — if at all — do we restore the
   between-city *level* the product ultimately displays?**

### The scale case (`04_bg_comparison.ipynb`)

The existing agency-scale baseline was decomposed against the correct observed
construct (`wtotal_rate` from `compute_weighted_scores`, the same weighted
relative-risk per-1K used by the prediction, *not* raw `count/pop`). Findings on
the 5 full-coverage cities (~5,254 BGs after the pop ≥ 250 floor):

- **Between-city level is essentially solved already.** City means line up ~1:1
  with observed (between-city Pearson **0.96**), and only **4.8%** of observed
  log-variance is *between* cities.
- **Within-city variation is the entire product problem** — **95.2%** of the
  variance is *within* city, and there the existing model is only a crude sorter:
  pooled within-city R² **0.32**, worst in **Detroit (0.12**, dangerous-block
  recall@top-decile just **10%**, "called safe" **37%**).

This is why the refresh targets the **within-city, city-demeaned** log rate: it
puts all modelling capacity on the 95% of variance that matters and that the
incumbent handles worst, and it removes a between-city level the incumbent already
captures well and that would otherwise dominate a pooled fit.

### The cross-city level tension

Demeaning the target discards the between-city level on purpose. But the deployed
index (RiskMeter / Spatial API) is a *nationally-indexed absolute score* — the
national-average post-processing (interpolation → national indexing → 0–100 cap)
is a **display transform that assumes the model already carries absolute level**.
So a within-city model needs *some* per-city level restored before that transform.

The user raised the natural idea: add a per-city column (e.g. `agency_2024`)
constant across a city's BGs — analogous to the agency model's central/midatlantic
control — to absorb across-city variation. The open question was whether that level
should be **learned from crime-free structure** (to avoid the temporally-lagged-crime
bias critique, cf. Lum & Isaac feedback loops) or taken from an **observed lagged
crime anchor**.

## Decision

Adopt a **ladder of target/level variants (A → D)** with A as the primary product
form and **D (observed anchor) as the chosen cross-city leveler**; reject the
structural leveler (C′) on empirical grounds.

| Variant | Target / level construction | Role |
|---|---|---|
| **A** | Within-city **demeaned** log rate | **Primary** — the product's within-city ranking model |
| **B** | **Raw** (absolute) log rate, pooled | Reference — shows how badly pooled level dominates |
| **C′** | Demeaned + **crime-free structural** city-mean composite | **Rejected** (see below) |
| **D** | Demeaned + **observed lagged UCR agency crime** per-city anchor | **Cross-city leveler** for the display transform |

### Why C′ (structural leveler) is rejected — empirical negative result

We tested whether a crime-free structural composite (social-disorganization proxies
available in the governance-stripped permissible set: distress/vacancy/liens,
residential instability, density — **no poverty/income/race**) could learn the
between-city level via its city-means (a named theory composite, the fewest-parameter
Mundlak form appropriate for 5–10 cities).

It cannot:

- The equal-weight composite **city-mean is ~uncorrelated with city crime level**
  (r = **−0.15** wtotal / **+0.14** wprop), and per-component between-city signs are
  inconsistent (ecological fallacy — within-BG signal does not aggregate to a
  between-city level).
- Under **leave-one-city-out (LOCO)**, Model C′ absolute R² is **catastrophically
  negative** (**−1.72** wtotal, atlanta fold **−24.9**; **+0.03** wprop, detroit
  **−2.0**) versus raw Model B at **0.245 / −0.08**. The level mapping fitted on
  n−1 cities **extrapolates unpredictably** to a held-out city.

**Root cause:** with only 5–10 cities you can support ~1 between-city parameter;
between-city level is *not learnable* from structure. Any leveler estimated on the
training cities becomes a near-arbitrary constant for a new city.

### Why D (observed anchor) is the leveler

The **only viable cross-city leveler is an *observed* per-city number**, not a
learned one. Lagged UCR agency crime (per-capita, year strictly earlier than the
target) is:

- **Given** for a held-out city, not extrapolated — it sidesteps the C′ failure
  entirely.
- **Crime-on-crime ≈ 1:1**, so it needs almost no estimation to restore level.
- **City-constant**, so it re-scales cities *without* re-ordering neighborhoods —
  which sharply limits the feedback-loop / lagged-crime bias critique (the anchor
  never drives *within-city* ranking, the product's actual output).

D is therefore kept **out of the within-city ranking model (A)** and applied only as
a per-city level input feeding the national-index display transform. It remains a
marginal, optional test rather than a core predictor.

## Consequences

- **A stays the deployed ranking model**; B is a reference baseline; C′ is
  documented and not pursued.
- **Detroit is the standing benchmark** threaded from 04 into the 02 notebooks — the
  city where the incumbent fails worst is the cleanest evidence the refresh helps
  (new within-R² 0.21 / recall@top10 0.28 vs incumbent 0.12 / 0.10).
- **Model D requires data-gathering not yet done:** lagged UCR agency crime for these
  cities, normalized agency → city (the `ct_muni` crosswalk can seed the mapping),
  resolved to **one number per city**. This is the next build step.
- **Governance:** because level comes from an observed, city-constant, crime-free-of-
  neighborhood-ranking anchor, the lagged-crime bias exposure is bounded and must be
  disclosed as such wherever D feeds the index.

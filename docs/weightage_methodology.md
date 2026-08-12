# Crime Risk Score — Weighting Methodology

This document explains, in plain language and with the underlying math, how the
severity/frequency-weighted crime scores are constructed in this repo
(`compute_weighted_scores` in `src/carrier_eval/scores.py`). It also walks through a
real client example to show how the Overall score relates to the Violent and
Property scores.

> **Scope note:** The scores in this repo are a *reconstruction* of the Cotality
> Crime Risk Score concept, built from raw incident data. Exact production
> per-crime weights may differ; the concept, structure, and directional behavior
> documented here match the product.

---

## 1. The building blocks

We work at the **census block group (BG)** level. For each BG we compute, for
each crime type, a rate per 1,000 residents:

$$
r_i^{\text{(bg)}} = \frac{\text{incident count of crime } i \text{ in the BG}}{\text{population}} \times 1000
$$

The seven **primary crime types** are:

- **Violent:** murder, rape, robbery, assault
- **Property:** burglary, larceny, motor vehicle theft (MVT)

(Vandalism is intentionally excluded from the composite totals.)

---

## 2. Normalizing to the national benchmark (relative risk)

Each local rate is divided by that crime's **national rate** ($N_i$, the
`*_pt_u` reference values) to produce a unitless **relative risk**:

$$
\text{rel}_i = \frac{r_i^{\text{(bg)}}}{N_i}
$$

Interpretation:

- $\text{rel}_i = 1.0$ → the BG matches the national average for that crime.
- $\text{rel}_i = 2.0$ → twice the national rate.
- $\text{rel}_i = 0.5$ → half the national rate.

Expressed as an index, $\text{rel}_i \times 100$ is directly comparable to the
per-crime index values shown in the product (e.g., a "Homicide Crime Index" of
`67.6` corresponds to $\text{rel}_{\text{murder}} \approx 0.676$).

### National reference rates (`*_pt_u`, per 1,000)

These are supplied pre-computed from the evals dataset (read via
`extract_national_rates` in `src/carrier_eval/scores.py`), not derived in this repo:

| Crime | National rate |
|---|---|
| murder | 0.050 |
| rape | 0.375 |
| robbery | 0.606 |
| assault | 2.561 |
| **violent (total)** | **3.591** |
| burglary | 2.292 |
| larceny | 12.721 |
| mvt | 2.588 |
| **property (total)** | **17.601** |

Note that the aggregate `violent` and `property` national rates equal the sum of
their components (e.g. $0.050 + 0.375 + 0.606 + 2.561 = 3.592 \approx 3.591$;
$2.292 + 12.721 + 2.588 = 17.601$), so:

$$
N_{\text{total}} = N_{\text{violent}} + N_{\text{property}} = 3.591 + 17.601 = 21.192 \text{ per 1{,}000}
$$

---

## 3. Combining crimes into composite scores

The composite scores are **equal-representation averages** of the relative
risks (each crime contributes an equal share):

**Overall (all 7 primary crimes, 1/7 each):**

$$
\text{wtotal\_rel} = \frac{1}{7} \sum_{i=1}^{7} \text{rel}_i
$$

**Property (3 property crimes, 1/3 each):**

$$
\text{wprop\_rel} = \frac{1}{3} \sum_{i \in \{\text{burglary, larceny, mvt}\}} \text{rel}_i
$$

These are unitless "how many times the national average" measures.

---

## 4. Rescaling back to an interpretable rate

To restate the composite on a per-1,000 scale, multiply by the relevant
aggregate national rate:

$$
\text{wtotal\_rate} = \text{wtotal\_rel} \times (N_{\text{violent}} + N_{\text{property}})
$$

$$
\text{wprop\_rate} = \text{wprop\_rel} \times N_{\text{property}}
$$

The rescale multiplier is a single constant applied to every BG, so it does not
change the ranking of BGs — it only restores interpretable units.

---

## 5. Key properties (why it behaves the way it does)

1. **The composite is not a raw sum of local rates.** Normalizing to national
   rates first prevents high-volume crimes (e.g. larceny) from mechanically
   dominating the score before weighting is applied.

2. **The Overall, Violent, and Property scores are parallel roll-ups.** Each is
   computed independently from the same normalized crime data. The Overall score
   is **not** an average of the Violent and Property scores.

3. **Frequency dominates the Overall score.** Because property crime is far more
   common nationally (~17.6 vs ~3.6 per 1,000), property crime represents
   roughly **83%** of total crime volume ($17.6 / 21.2$). As a result, the
   Overall score is driven predominantly by the property profile and tends to
   track the Property score closely.

---

## 6. Worked client example

**Client question:** *"How does the Crime Risk Score overall score relate to the
Violent Crime and Property scores? We ran a property with an overall 100 score,
100 for Property, and 92 for Violent Crime. How was the overall derived — it
clearly isn't an average?"*

**Observed values:**

| Score | Value |
|---|---|
| Overall | 100 |
| Property | 100 |
| Violent | 92 |

A simple average of the two category scores would give
$(100 + 92) / 2 = 96$, not 100 — which is why the client correctly noticed it
isn't an average.

**Explanation:** The Overall score is built bottom-up from the individual crime
types, each normalized to its national benchmark, and weighted by how commonly
each crime occurs. Because property crime accounts for ~80%+ of national crime
volume, the Overall score is dominated by the (maxed-out) property component,
while the somewhat lower violent component has only a small effect. The Overall
therefore lands at 100 alongside Property, and the Violent score of 92 barely
moves it.

**Client-ready wording:**

> The Overall score is not an average of the Violent and Property scores, and it
> isn't calculated from them. The Violent, Property, and Overall scores are three
> parallel roll-ups of the same underlying data — each individual crime type is
> measured against its national benchmark, then combined. Crimes are weighted by
> how frequently they occur nationally, and property crimes are far more common
> than violent crimes (~17.6 vs ~3.6 per 1,000, so property is ~80%+ of total
> crime volume). The Overall score is therefore driven predominantly by the
> property profile and tends to track the Property score. In this property, the
> maxed-out Property score (100) pulls the Overall to 100, while the lower
> Violent score (92) has only a small effect. A very high value in a
> high-frequency category can lift the Overall above where a simple average of
> the two category scores would fall.

---

## 7. Reference: code mapping

| Concept | Column / symbol | Location |
|---|---|---|
| Local rate per 1K | `{crime}_rate` | `normalize_actuals` |
| National rate $N_i$ | `{crime}_pt_u` | `extract_national_rates` (`carrier_eval/scores.py`) |
| Relative risk $\text{rel}_i$ | `{crime}_rel` (intermediate) | `compute_weighted_scores` |
| Overall composite (unitless) | `wtotal_rel` | `compute_weighted_scores` |
| Property composite (unitless) | `wprop_rel` | `compute_weighted_scores` |
| Overall rate per 1K | `wtotal_rate` | `compute_weighted_scores` |
| Property rate per 1K | `wprop_rate` | `compute_weighted_scores` |
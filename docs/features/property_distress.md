# Property-distress predictors — definitions & statistical treatment

Covers the four block-group (BG) property-record shares built in BigQuery under
`src/regression_modelling/data_wrangling/sql/build/` and their KNN(6) spatial lags. These are
built-environment / property-record measures (not ACS demographics), in scope under the
design constraint in `hypothesis.md`.

## 1. Definitions

| Column | Numerator | Denominator | Window | Source |
|---|---|---|---|---|
| `vacant_pct` | vacant USPS delivery points | all delivery points | current | `vacancy.sql` |
| `clip_liens_pct` | properties with a tax lien (`category_type='J'`, tax types only, amount ≥ $100) | all properties in BG | 2020–2024 | `liens.sql` |
| `clip_transaction_pct` | properties with **any** recorded transaction (all deed categories) | all properties in BG | 2020–2024 | `transactions.sql` |
| `clip_foreclosure_pct` | transacted properties with a foreclosure deed (`deedcattyp='U'`) | **transacted** properties in BG | 2020–2024 | `foreclosures.sql` |

- **Property = clip**, not parcel. `NS_pcl_universe_xref.clip_list` is pipe-delimited (a condo
  or strip-mall parcel carries many clips), so builds explode it with
  `UNNEST(SPLIT(clip_list, "|"))`. The ~0.2% of clips that appear in >1 BG are assigned to the
  BG holding most of their parcel shapes, so no clip is double-counted.
- **Foreclosures are a share of transactions**, not of all properties: foreclosure is a deed
  type, so it is a subset of transactions. This separates *turnover* (`clip_transaction_pct`)
  from *distress among sales* (`clip_foreclosure_pct`).
- **Zero denominators** (e.g. a BG with no transactions) yield NULL via `SAFE_DIVIDE`, and
  `ZERO_FILL` sets them to 0 downstream. No minimum-count filter is applied yet — decide after
  distribution EDA (see §4).

## 2. Spatial lag — pooled KNN(6)

Each `*_lag6` column summarizes the BG's **6 nearest neighbouring BGs** (centroid distance,
same state, self excluded, 25 km search prefilter).

The lag is **pooled**: the neighbours' raw counts are summed before dividing.

$$
\text{lag6}(i) = 100 \times \frac{\sum_{j \in N_6(i)} \text{numerator}_j}{\sum_{j \in N_6(i)} \text{denominator}_j}
$$

It is **not** the mean of the neighbours' percentages.

### Why pooled, not mean-of-rates

A mean of rates gives every neighbour equal weight regardless of how much data backs its rate,
so a neighbour with a tiny denominator can dominate. Example, one BG's six neighbours:

| Neighbour | Foreclosures / sales | Rate |
|---|---|---|
| A | 1 / 2 | 50% |
| B–F (each) | 10 / 400 | 2.5% |

- Mean of rates: (50 + 5 × 2.5) / 6 = **10.4%** — one chance foreclosure quadruples the lag.
- Pooled: (1 + 50) / (2 + 2,000) = **2.55%** — the neighbourhood's actual foreclosure rate.

This matters most for `clip_foreclosure_pct`, whose denominator (transacted properties) can be
small. For vacancy, liens, and transactions the denominators are large, so pooling changes
little, but it is applied to all four so every lag has the same meaning.

**Trade-off accepted:** pooling weights neighbours by volume, so one very large neighbour (e.g.
a dense condo BG) contributes more. For ratios of small counts, stability is preferred over
equal per-area votes. Neighbours with a zero/NULL denominator contribute nothing (correct: they
carry no information), rather than being silently dropped from an average.

## 3. Model form

- All four shares and their lags are right-skewed by assumption → `log1p` into `{col}_log`
  (`PROPERTY_MODEL_TRANSFORMS` in `constants.py`), zero-filled via `ZERO_FILL`.
- Agency-level roll-up (`data_wrangling/agency.py`) population-weights the raw shares, then
  applies `log1p` (aggregate raw → transform).

## 4. Open decisions (after the rebuild + distribution EDA)

1. **Small-denominator filtering / shrinkage for the BG's own foreclosure rate** — e.g. a
   minimum number of transactions, or empirical-Bayes shrinkage toward the pooled lag.
2. **Transform check** — a 5-year transaction share may not be skewed enough to need `log1p`.
3. **Collinearity** — `clip_transaction_pct` (turnover) vs ACS `moved1yr_pct`; check
   correlation / VIF before keeping both.
4. **Coefficient re-read** — `clip_foreclosure_pct_lag6` was the strongest property predictor
   under the old (all-properties) definition; its meaning changed, so re-interpret after refit.

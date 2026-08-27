"""Constants for `bias_testing` — protected attributes live HERE, never in the
predictor registry (`FEATURE_SOURCES`/`PREDICTOR_COLS`). Per ADR 0004, these columns
are used only to check that predictors carry a crime signal that is not merely a proxy
for a protected attribute; leakage into the fit is impossible by construction.

`config.py` holds paths only — this file holds the registry (see docs/CONTEXT.md).
"""

# Protected attributes pulled from the same ACS `.sav` as the demographic predictors, but
# routed to a separate bias-only table (data/interim/bias/) keyed by `geoid`. Column name
# in the .sav -> human description.
PROTECTED_ATTRIBUTES = {
    "md_hhinc": "Median household income",
    "wht_pct":  "Percentage of population White and not Hispanic",
}

# The columns as they appear in bg_acs.sav (source of truth for the loader).
PROTECTED_COLS = list(PROTECTED_ATTRIBUTES)

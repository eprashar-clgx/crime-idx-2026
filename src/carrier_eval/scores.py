"""Score-reconstruction math for carrier evals (group D).

The weighted-score math lives in the shared foundation now (ADR 0005:
`crime_blockgroup_mapping.scores`) so `regression_modelling` can build the
`log(wtotal_rate)` target without importing this task. This module re-exports it and
keeps the carrier-specific `EVALS_PATH` default for `extract_national_rates`.
"""
from crime_blockgroup_mapping.scores import compute_weighted_scores
from crime_blockgroup_mapping.scores import extract_national_rates as _extract_national_rates

from carrier_eval.config import EVALS_PATH

__all__ = ["compute_weighted_scores", "extract_national_rates"]


def extract_national_rates(path=None):
    """Extract national `*_pt_u` rates from the carrier evals parquet (defaults to EVALS_PATH)."""
    return _extract_national_rates(path or EVALS_PATH)

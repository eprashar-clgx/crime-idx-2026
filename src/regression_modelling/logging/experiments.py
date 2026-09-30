"""Experiment log — a durable record of every predictor-selection run (ADR 0010).

One logged run = one summary row + its out-of-sample predictions, each written to its own
parquet under ``config.experiments_dir()``::

    experiments/runs/{run_id}.parquet         # 1 row: what was fit and how it scored
    experiments/predictions/{run_id}.parquet  # geoid, holdout_city, y_pred

One file per run (rather than an appended table) so parallel candidate fits never race on
a write. ``run_cv`` knows nothing about this module: callers log a finished ``FoldRun``.

Predictions are stored without the pool's columns; ``load_run`` re-joins them to a pool
on ``geoid`` so any metric in ``models.metrics`` can be recomputed later.
"""
from __future__ import annotations

import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from regression_modelling import config
from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.metrics import city_scores
from regression_modelling.models.results import FoldRun


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else config.experiments_dir()


def _git_state() -> tuple[str, bool]:
    """(short SHA, dirty?) for tracked files; ("unknown", False) outside a repo."""
    try:
        cwd = Path(__file__).resolve().parent
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=cwd, capture_output=True, text=True,
                               check=True).stdout.strip() != ""
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


def summarize(run: FoldRun) -> dict:
    """Headline scores for the summary row (unrounded; `lograte` runs only)."""
    pooled = city_scores(run, net=0.10).loc["POOLED"]
    r25 = city_scores(run, net=0.25).loc["POOLED", "recall"]
    s = run.scored
    spearman = float(s["y_pred"].corr(s[_rate_col(run.mode, run.category)],
                                      method="spearman"))
    return {"n": int(pooled["n"]), "r2_oos": float(pooled["r2_oos"]),
            "recall10": float(pooled["recall"]), "recall25": float(r25),
            "spearman": spearman}


def log_run(run: FoldRun, name: str, *, estimator: str, params: dict | None = None,
            save_predictions: bool = True, tags: dict | None = None,
            root: Path | None = None) -> str:
    """Write one run to the log and return its ``run_id``.

    ``name`` groups related runs (e.g. ``"backward_ridge_wprop"``); ``tags`` is free-form
    context (step number, the predictor added/dropped, ...). Set ``save_predictions=False``
    for throw-away candidate fits — the summary row is still kept.
    """
    root = _root(root)
    run_id = new_run_id()
    sha, dirty = _git_state()
    row = {
        "run_id": run_id, "name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "category": run.category, "mode": run.mode, "split": run.split,
        "estimator": estimator, "n_predictors": len(run.predictors),
        "predictors": json.dumps(list(run.predictors)),
        "params": json.dumps(params or {}, default=str),
        "tags": json.dumps(tags or {}, default=str),
        "git_sha": sha, "git_dirty": dirty, "has_predictions": save_predictions,
        **summarize(run),
    }
    (root / "runs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(root / "runs" / f"{run_id}.parquet", index=False)
    if save_predictions:
        (root / "predictions").mkdir(parents=True, exist_ok=True)
        run.scored[["geoid", "holdout_city", "y_pred"]].to_parquet(
            root / "predictions" / f"{run_id}.parquet", index=False)
    return run_id


def load_runs(name: str | None = None, root: Path | None = None) -> pd.DataFrame:
    """All summary rows (optionally one ``name``), oldest first; ``predictors`` decoded."""
    files = sorted((_root(root) / "runs").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if name is not None:
        df = df[df["name"] == name]
    df = df.assign(predictors=df["predictors"].map(json.loads))
    return df.sort_values("timestamp").reset_index(drop=True)


def load_predictions(run_id: str, root: Path | None = None) -> pd.DataFrame:
    path = _root(root) / "predictions" / f"{run_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no saved predictions for run {run_id!r} ({path})")
    return pd.read_parquet(path)


def load_run(run_id: str, pool: pd.DataFrame, root: Path | None = None) -> FoldRun:
    """Rebuild a ``FoldRun`` from logged predictions joined back onto ``pool`` by geoid."""
    runs = load_runs(root=root)
    meta = runs.loc[runs["run_id"] == run_id]
    if meta.empty:
        raise KeyError(f"run {run_id!r} not in the experiment log")
    meta = meta.iloc[0]
    preds = load_predictions(run_id, root)
    scored = preds.merge(pool.drop(columns=["y_pred", "holdout_city"], errors="ignore"),
                         on="geoid", how="left", validate="one_to_one")
    return FoldRun(scored=scored, mode=meta["mode"], split=meta["split"],
                   category=meta["category"], predictors=list(meta["predictors"]))

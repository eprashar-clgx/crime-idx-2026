"""Report figures for the block-group model (02_regression_prediction).

Every function returns a matplotlib ``Figure``; `save` writes PNGs for the deck. Maps
join held-out predictions to Census cartographic-boundary block groups and rank BGs
WITHIN the city, so the observed and predicted panels share one colour scale whatever
the city's level.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from regression_modelling.models.dataset import _rate_col
from regression_modelling.models.diagnostics import FEATURE_FAMILIES, feature_family
from regression_modelling.models.results import FoldRun

FAMILY_COLORS = dict(zip([f for f, _ in FEATURE_FAMILIES] + ["demographic"],
                         plt.get_cmap("tab10").colors))
HIT_COLORS = {"hit": "#b2182b", "missed": "#f4a582", "false alarm": "#4393c3",
              "other": "#f0f0f0"}


def save(fig, name: str, out_dir: str | Path, dpi: int = 200) -> Path:
    """Write ``fig`` to ``out_dir/name.png`` (directory created) and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def _label(city: str) -> str:
    return {"dc": "Washington DC", "new_york": "New York"}.get(
        city, city.replace("_", " ").title())


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #
def metric_bars(card: pd.DataFrame, metrics=("r2_pooled", "r2_city", "recall@10"),
                title: str = ""):
    """Grouped bars: one group per metric, one bar per scorecard row (model)."""
    m = card[list(metrics)]
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) * max(1, len(m) / 3) + 2, 4.5))
    width = 0.8 / len(m)
    x = np.arange(len(metrics))
    colors = plt.get_cmap("Blues")(np.linspace(0.3, 0.9, len(m)))
    if any("incumbent" in str(i).lower() for i in m.index):
        colors[0] = (0.6, 0.6, 0.6, 1)
    for k, (name, row) in enumerate(m.iterrows()):
        bars = ax.bar(x + (k - (len(m) - 1) / 2) * width, row.values, width,
                      label=name, color=colors[k])
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=1)
    ax.set_xticks(x, [c.replace("_", " ") for c in metrics])
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def city_r2_bars(per_city: pd.DataFrame, model_col: str, base_col: str | None = None,
                 highlight: dict | None = None, title: str = "",
                 labels: tuple[str, str] = ("model", "incumbent")):
    """Per-held-out-city r2, sorted, optionally against a baseline column. ``highlight``
    maps city -> colour for the cities called out in the deck."""
    d = per_city.sort_values(model_col)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(d) + 1.2))
    y = np.arange(len(d))
    colors = [(highlight or {}).get(c, "#2c7fb8") for c in d.index]
    ax.barh(y, d[model_col], color=colors, label=labels[0])
    if base_col is not None:
        ax.scatter(d[base_col], y, color="black", marker="|", s=120, zorder=3,
                   label=labels[1])
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_yticks(y, [_label(c) for c in d.index])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("held-out r² (log rate)")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# SHAP                                                                        #
# --------------------------------------------------------------------------- #
def shap_bar(importance: pd.Series, top: int = 20, title: str = "",
             xlabel: str = "mean |SHAP| (log-rate units)"):
    """Horizontal bar of the ``top`` features, coloured by family."""
    imp = importance.head(top)[::-1]
    fams = [feature_family(c) for c in imp.index]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=[FAMILY_COLORS[f] for f in fams])
    used = list(dict.fromkeys(reversed(fams)))
    ax.legend(handles=[Patch(color=FAMILY_COLORS[f], label=f) for f in used],
              loc="lower right", fontsize=8, frameon=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def family_bar(fam: pd.Series, title: str = ""):
    """Family-level grouped mean |SHAP|, as a share of the total."""
    share = (fam / fam.sum()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(share) + 1.2))
    ax.barh(share.index, share.values, color=[FAMILY_COLORS[f] for f in share.index])
    for y, v in enumerate(share.values):
        ax.text(v + 0.005, y, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlabel("share of grouped mean |SHAP|")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def family_by_city(fam_city: pd.DataFrame, cities: list[str], title: str = ""):
    """Stacked shares of grouped |SHAP| by family for selected held-out cities."""
    share = fam_city.loc[cities]
    share = share.div(share.sum(axis=1), axis=0)
    order = share.mean().sort_values(ascending=False).index
    labels = [_label(c) for c in cities]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(cities) + 1.8))
    left = np.zeros(len(cities))
    for f in order:
        ax.barh(labels, share[f], left=left, color=FAMILY_COLORS[f], label=f)
        left += share[f].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of grouped mean |SHAP|")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.set_title(title)
    fig.tight_layout()
    return fig


def shap_beeswarm(shap_df: pd.DataFrame, X: pd.DataFrame, top: int = 15, title: str = ""):
    """``shap`` beeswarm on held-out SHAP values (colour = the feature's own value)."""
    import shap
    cols = [c for c in shap_df.columns if c not in ("_base", "holdout_city")]
    plt.figure()            # summary_plot draws on the current figure
    shap.summary_plot(shap_df[cols].to_numpy(), X.loc[shap_df.index, cols],
                      max_display=top, show=False, plot_size=(9, 0.38 * top + 1.5))
    fig = plt.gcf()
    fig.axes[0].set_title(title)
    return fig


# --------------------------------------------------------------------------- #
# City maps and scatter                                                       #
# --------------------------------------------------------------------------- #
def city_frame(run: FoldRun, city: str, net: float = 0.10) -> pd.DataFrame:
    """Held-out rows of one city with within-city deciles and a worst-``net`` outcome:
    ``hit`` (observed and predicted worst), ``missed`` (observed only), ``false alarm``
    (predicted only), ``other``."""
    rate_col = _rate_col(run.mode, run.category)
    d = run.scored[run.scored["holdout_city"] == city].copy()
    obs_p = d[rate_col].astype(float).rank(pct=True)
    pred_p = d["y_pred"].astype(float).rank(pct=True)
    d["obs_log"] = np.log1p(d[rate_col].astype(float))
    d["obs_decile"] = np.ceil(obs_p * 10).clip(1, 10).astype(int)
    d["pred_decile"] = np.ceil(pred_p * 10).clip(1, 10).astype(int)
    o, p = obs_p >= 1 - net, pred_p >= 1 - net
    d["top_outcome"] = np.select([o & p, o & ~p, ~o & p], ["hit", "missed", "false alarm"],
                                 "other")
    return d


def city_geometries(city: str, geoids):
    """Census cartographic-boundary polygons for ``geoids`` (EPSG:3857), quietly."""
    from crime_blockgroup_mapping.boundaries import load_state_block_groups
    from crime_blockgroup_mapping.constants import CITIES
    with contextlib.redirect_stdout(io.StringIO()):
        bg = load_state_block_groups(CITIES[city])
    bg = bg[bg["geoid"].isin(set(geoids))]
    return bg[["geoid", "geometry"]].to_crs(epsg=3857)


def _basemap(ax):
    try:
        import contextily as cx
        # CartoDB tiles now return an "API key required" watermark; Esri's grey canvas
        # is keyless.
        cx.add_basemap(ax, source=cx.providers.Esri.WorldGrayCanvas, attribution=False)
    except Exception:          # offline: maps still render without tiles
        pass


def city_fit_map(run: FoldRun, city: str, net: float = 0.10, geoms=None,
                 label: str | None = None):
    """Observed decile | predicted decile | worst-``net`` hit/miss map for one held-out
    city. Deciles are within-city, so the two left panels are directly comparable."""
    d = city_frame(run, city, net)
    geoms = geoms if geoms is not None else city_geometries(city, d["geoid"])
    g = geoms.merge(d, on="geoid", how="inner")
    cmap = plt.get_cmap("RdYlBu_r", 10)
    hit = d["top_outcome"].value_counts()
    n_hit = int(hit.get("hit", 0))
    n_top = n_hit + int(hit.get("missed", 0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    for ax, col, ttl in [(axes[0], "obs_decile", "Observed crime rate (decile)"),
                         (axes[1], "pred_decile", "Predicted, city held out (decile)")]:
        g.plot(ax=ax, column=col, cmap=cmap, vmin=0.5, vmax=10.5, alpha=0.85,
               edgecolor="white", linewidth=0.1)
        ax.set_title(ttl)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.5, 10.5))
    fig.subplots_adjust(bottom=0.12, wspace=0.05)
    cax = fig.add_axes([0.16, 0.07, 0.38, 0.022])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=range(1, 11))
    cb.set_label("within-city decile (10 = highest)")

    g.plot(ax=axes[2], color=g["top_outcome"].map(HIT_COLORS), alpha=0.9,
           edgecolor="white", linewidth=0.1)
    axes[2].legend(handles=[Patch(color=HIT_COLORS[k], label=f"{k} ({hit.get(k, 0)})")
                            for k in ("hit", "missed", "false alarm")],
                   loc="lower left", fontsize=9, framealpha=0.9)
    axes[2].set_title(f"Worst {net:.0%} of blocks: {n_hit}/{n_top} caught "
                      f"({n_hit / max(n_top, 1):.0%})")
    for ax in axes:
        _basemap(ax)
        ax.set_axis_off()
    fig.suptitle(label or _label(city), fontsize=15, y=0.98)
    return fig


def pred_vs_obs(run: FoldRun, cities: list[str], ncols: int | None = None):
    """Predicted vs observed log1p(rate) per held-out city, titled with that city's r2,
    Spearman and level error (mean predicted minus mean observed). The dashed line is
    y = x: a cloud shifted off it is a level miss, a cloud spread around it is noise."""
    ncols = ncols or len(cities)
    nrows = int(np.ceil(len(cities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    for ax, city in zip(axes.flat, cities):
        d = city_frame(run, city)
        y, p = d["obs_log"], d["y_pred"].astype(float)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rho = y.rank().corr(p.rank())
        err = p.mean() - y.mean()
        ax.scatter(p, y, s=6, alpha=0.35, color="#2c7fb8", edgecolor="none")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{_label(city)}\nr² {r2:.2f} · Spearman {rho:.2f} · level {err:+.2f}",
                     fontsize=10)
        ax.set_xlabel("predicted log1p(rate)")
        ax.set_ylabel("observed log1p(rate)")
    for ax in list(axes.flat)[len(cities):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig

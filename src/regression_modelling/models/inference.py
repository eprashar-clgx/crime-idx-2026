"""Inference: standardized OLS coefficients and their diagnostics (ADR 0008).

The counterpart to `training`. Where `training` asks *"how well does this score a city
we have never seen?"*, this module asks *"what does the fitted relationship say?"* — so
it fits on the full estimation sample and reports coefficients, standard errors and
residual structure rather than held-out skill.

That full-sample fit is deliberate, not an oversight: there is no leakage concern
because nothing here is used to make an out-of-sample claim. Anything that predicts a
held-out city belongs in `training`, behind the fold-local scaler.

Renamed from `model.py`, which had grown a second, dead copy of half this file. The
duplicate Moran's I (`spatial_moran`) was dropped rather than kept: it took centroids in
EPSG:4326 and ran KNN on raw degrees, so its neighbour sets were latitude-distorted.
`residual_spatial` below reprojects to EPSG:3857 first and is the one to use.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from regression_modelling.constants import PREDICTOR_COLS


def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column (mean 0, sd 1) so coefficients are directly comparable.

    Standardized on the full estimation sample. This is an *inferential* fit (we want
    the coefficients), not a held-out prediction task, so there is no train/test leakage
    concern — contrast `training.LinearEstimator`, whose scaler is fold-local precisely
    because it does make out-of-sample claims. Zero-variance columns (e.g. a between-city
    level feature that is constant within a city once demeaned) map to all-zeros rather
    than NaN, so they enter the design harmlessly (pinv drops them).
    """
    sd = X.std(ddof=0).replace(0, 1.0)
    return (X - X.mean()) / sd


def fit_ols(df: pd.DataFrame, target: str,
            predictors=PREDICTOR_COLS, robust: str = "HC3"):
    """Fit standardized OLS on the full sample. Returns (result, result_robust, design)."""
    d = df.dropna(subset=[target] + list(predictors)).copy()
    X = sm.add_constant(standardize(d[list(predictors)]))
    y = d[target].to_numpy()
    result = sm.OLS(y, X).fit()
    return result, result.get_robustcov_results(cov_type=robust), d


def coef_table(result, result_robust, predictors=PREDICTOR_COLS) -> pd.DataFrame:
    """Tidy coefficient table: classical vs HC3 robust SE / t / p.

    Because predictors are standardized and the target is a log outcome, each coef ≈
    change in log-outcome per +1 SD of the predictor; pct_effect ≈ (exp(coef)-1)*100 is
    the approx % change in the expected outcome per +1 SD.
    """
    names = ["const"] + list(predictors)
    tab = pd.DataFrame({
        "coef":   np.asarray(result.params),
        "se":     np.asarray(result.bse),
        "p":      np.asarray(result.pvalues),
        "se_HC3": np.asarray(result_robust.bse),
        "t_HC3":  np.asarray(result_robust.tvalues),
        "p_HC3":  np.asarray(result_robust.pvalues),
    }, index=names)
    tab["pct_effect"] = (np.exp(tab["coef"]) - 1) * 100
    tab["sig"] = pd.cut(tab["p_HC3"], [-0.01, .001, .01, .05, 1.01],
                        labels=["***", "**", "*", ""])
    return tab.round(4)


def fit_summary(result) -> pd.Series:
    """Headline fit statistics for one estimation."""
    return pd.Series({
        "n":        int(result.nobs),
        "r2":       round(result.rsquared, 4),
        "r2_adj":   round(result.rsquared_adj, 4),
        "f_pvalue": result.f_pvalue,
        "aic":      round(result.aic, 1),
    })


def residual_spatial(city: str, geoids, resid, k: int = 8):
    """Attach within-city BG geometry to residuals (by geoid) and compute Moran's I in one
    geometry pass — so the residual MAP and the spatial-autocorrelation TEST share the load.

    Returns (gdf, moran): `gdf` is a GeoDataFrame (EPSG:4326 geometry) carrying a `resid`
    column for a choropleth; `moran` is the esda.Moran on KNN(k) centroid weights. A
    significant positive Moran's I ⇒ residuals cluster in space ⇒ the (non-spatial) fit is
    still missing structure ⇒ the case for a spatial variable / spatial-lag / error model.

    Centroids are taken in EPSG:3857, not in degrees: KNN on unprojected lat/lon stretches
    east-west distance by ~cos(latitude), which quietly reshapes the neighbour set.
    """
    from crime_blockgroup_mapping.constants import CITIES
    from crime_blockgroup_mapping.boundaries import (
        load_state_block_groups, label_bgs_within_city, load_city_boundary,
    )
    from libpysal.weights import KNN
    from esda.moran import Moran

    cfg = CITIES[city]
    bg = load_state_block_groups(cfg)
    bg = label_bgs_within_city(bg, load_city_boundary(cfg))
    bg = bg[bg["within_city"]][["geoid", "geometry"]].copy()

    r = pd.DataFrame({"geoid": np.asarray(geoids), "resid": np.asarray(resid, dtype=float)})
    gdf = bg.merge(r, on="geoid", how="inner")

    proj = gdf.to_crs(3857)                                   # metric CRS for honest centroids
    coords = np.c_[proj.geometry.centroid.x, proj.geometry.centroid.y]
    w = KNN.from_array(coords, k=k)
    w.transform = "r"
    return gdf, Moran(gdf["resid"].to_numpy(), w)

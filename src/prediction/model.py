"""Per-city OLS: learn standardized predictor→crime coefficients + diagnostics.

Sequence per model:
  1. standardized OLS  (coef comparable across predictors)
  2. classical SE / t / p
  3. HC3 robust SE / t / p  (heteroskedasticity-consistent)
  4. residual diagnostics   (resid-vs-fitted, Q-Q, Breusch-Pagan, Jarque-Bera)
  5. Moran's I on residuals  (spatial autocorrelation → is OLS mis-specified?)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

from prediction.dataset import PREDICTOR_COLS


# ------------------------------------------------------------------ core fit --
def standardize(X: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column (mean 0, sd 1) so coefficients are directly comparable.

    NOTE: standardized on the full estimation sample. This is an *inferential*
    fit (we want the coefficients), not a held-out prediction task, so there is
    no train/test leakage concern here.
    """
    return (X - X.mean()) / X.std(ddof=0)


def fit_ols(df: pd.DataFrame, target: str,
            predictors=PREDICTOR_COLS, robust: str = "HC3"):
    """Fit standardized OLS. Returns (result, result_robust, design_frame)."""
    d = df.dropna(subset=[target] + list(predictors)).copy()
    X = sm.add_constant(standardize(d[predictors]))
    y = d[target].to_numpy()
    result = sm.OLS(y, X).fit()
    result_robust = result.get_robustcov_results(cov_type=robust)
    return result, result_robust, d


def coef_table(result, result_robust, predictors=PREDICTOR_COLS) -> pd.DataFrame:
    """Tidy coefficient table: classical vs HC3 robust SE / t / p.

    Because predictors are standardized and the target is log(count+1), each
    coef ≈ change in log-count per +1 SD of the predictor; pct_effect ≈
    (exp(coef)-1)*100 is the approx % change in expected count per +1 SD.
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
    return pd.Series({
        "n":        int(result.nobs),
        "r2":       round(result.rsquared, 4),
        "r2_adj":   round(result.rsquared_adj, 4),
        "f_pvalue": result.f_pvalue,
        "aic":      round(result.aic, 1),
    })


# ---------------------------------------------------------------- diagnostics --
def residual_diagnostics(result, title: str = ""):
    """Plot resid-vs-fitted, Q-Q, histogram; print Breusch-Pagan + Jarque-Bera."""
    resid, fitted = result.resid, result.fittedvalues
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].scatter(fitted, resid, s=8, alpha=.35)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set_xlabel("fitted"); ax[0].set_ylabel("residual")
    ax[0].set_title("Residuals vs Fitted")
    stats.probplot(resid, dist="norm", plot=ax[1]); ax[1].set_title("Normal Q-Q")
    ax[2].hist(resid, bins=40, color="slateblue", edgecolor="white")
    ax[2].set_title("Residual histogram")
    fig.suptitle(title); fig.tight_layout(); plt.show()

    bp_p = het_breuschpagan(resid, result.model.exog)[1]
    jb_p = jarque_bera(resid)[1]
    print(f"Breusch-Pagan (heteroskedasticity): p = {bp_p:.4g}"
          f"  → {'heteroskedastic (use HC3)' if bp_p < .05 else 'homoskedastic'}")
    print(f"Jarque-Bera  (normality):           p = {jb_p:.4g}"
          f"  → {'non-normal residuals' if jb_p < .05 else 'approx normal'}")
    return resid


def spatial_moran(city: str, design_frame: pd.DataFrame, resid, k: int = 8):
    """Moran's I on residuals via KNN weights over BG centroids (rejoined on geoid).

    Significant positive I ⇒ residual clustering ⇒ plain OLS is missing spatial
    structure ⇒ escalate to a spatial lag / error model or add spatial features.
    """
    from core.config import CITIES
    from core import geo_utils as geo
    from libpysal.weights import KNN
    from esda.moran import Moran

    cfg = CITIES[city]
    bg = geo.load_state_block_groups(cfg)
    bg = geo.label_bgs_within_city(bg, geo.load_city_boundary(cfg))
    bg = bg[bg["within_city"]][["geoid", "geometry"]].copy()
    bg["cx"] = bg.geometry.centroid.x
    bg["cy"] = bg.geometry.centroid.y

    r = pd.DataFrame({"geoid": design_frame["geoid"].to_numpy(),
                      "resid": np.asarray(resid)})
    m = bg.merge(r, on="geoid", how="inner")

    w = KNN.from_array(m[["cx", "cy"]].to_numpy(), k=k)
    w.transform = "r"
    mi = Moran(m["resid"].to_numpy(), w)
    print(f"Moran's I on residuals (k={k}): I = {mi.I:.4f}, p = {mi.p_sim:.4g}"
          f"  → {'SPATIAL autocorrelation present' if mi.p_sim < .05 else 'no sig. spatial autocorrelation'}")
    return mi


# --------------------------------------------------------------------- driver --
def fit_and_report(df: pd.DataFrame, city: str, target: str = "cl_total_logcount",
                   predictors=PREDICTOR_COLS, spatial: bool = True):
    """Full sequence for one city/target. Returns dict of artifacts."""
    print(f"\n{'='*72}\n{city.upper()} — {target}\n{'='*72}")
    result, robust, d = fit_ols(df, target, predictors)
    print(fit_summary(result).to_string(), "\n")
    tab = coef_table(result, robust, predictors)
    print(tab.to_string())

    resid = residual_diagnostics(result, title=f"{city.title()} — {target}")
    mi = spatial_moran(city, d, resid) if spatial else None
    return {"result": result, "robust": robust, "coef": tab, "moran": mi, "design": d}
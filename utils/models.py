"""Logistic models with clustered SEs, spline contrasts, and random-effects pooling."""

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from .common import SPLINE_DF


def spline(var: str) -> str:
    """Natural cubic spline (same function space as rcs), knots at quantiles.

    Centered so it is identifiable beside the intercept; uncentered, the
    clustered covariance is singular and the SEs come out NaN.
    """
    return f"cr({var}, df={SPLINE_DF}, constraints='center')"


def drop_constant_outcome_groups(df: pd.DataFrame, outcome: str, by: str) -> pd.DataFrame:
    """Drop groups whose outcome never varies (e.g. a unit with no deaths).

    Their fixed effect goes to +/-infinity and the Hessian turns singular; under group fixed
    effects they carry no information on the other coefficients (conditional logit drops them too).
    """
    return df[df.groupby(by)[outcome].transform("nunique") > 1]


def _fit(model, df: pd.DataFrame, cluster: list, **kw):
    sub = df.loc[model.data.row_labels]
    groups = np.column_stack([pd.factorize(sub[c])[0] for c in cluster])
    cov = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
    with warnings.catch_warnings():  # convergence is recorded in mle_retvals, not spammed
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = model.fit(**cov, **kw)
        if not getattr(res, "mle_retvals", {}).get("converged", True):
            # Newton's absolute step tolerance stalls on hundreds of fixed-effect dummies; finish from where it stopped
            res = model.fit(**cov, method="lbfgs", start_params=res.params, maxiter=2000, disp=0)
    return res


def fit_logit(df: pd.DataFrame, outcome: str, terms: list, cluster: list):
    """Logistic regression on complete cases with SEs clustered on every column in `cluster`.

    Two columns give two-way clustering (statsmodels takes a 2-column `groups`).
    """
    return _fit(smf.logit(f"{outcome} ~ " + " + ".join(terms), df), df, cluster, disp=0, maxiter=200)


def fit_ols(df: pd.DataFrame, outcome: str, terms: list, cluster: list):
    """Linear regression with the same clustering (balance tests: a baseline covariate on the exposure)."""
    return _fit(smf.ols(f"{outcome} ~ " + " + ".join(terms), df), df, cluster)


def e_value(odds_ratio: float) -> float:
    """VanderWeele & Ding: the weakest association an unmeasured confounder would need with both
    exposure and outcome (risk-ratio scale) to explain the estimate away. 1 means none needed."""
    r = max(odds_ratio, 1 / odds_ratio)
    return r + np.sqrt(r * (r - 1))


def spline_contrasts(res, var: str, grid, ref: float) -> pd.DataFrame:
    """Log-OR (and SE) of `var` at each grid value vs `ref`, all else held equal.

    Design rows differ only in `var`, so every other term cancels in the
    difference; the SE is the delta method on the clustered covariance.
    """
    first = res.model.data.row_labels[0]  # first complete case: a row with an NA covariate has no design row
    rows = pd.concat([res.model.data.frame.loc[[first]]] * (len(grid) + 1), ignore_index=True)
    rows[var] = list(grid) + [ref]
    X = np.asarray(build_design_matrices([res.model.data.design_info], rows)[0])
    D = X[:-1] - X[-1]
    se = np.sqrt(np.einsum("ij,jk,ik->i", D, res.cov_params().values, D))
    return pd.DataFrame({"x": list(grid), "estimate": D @ res.params.values, "se": se})


def random_effects(est, se) -> dict:
    """Pool site log-ORs: Paule-Mandel tau^2 with a t interval on k-1 df (DerSimonian-Laird's
    z interval is too narrow with 3-4 sites), plus the fixed-effect estimate. One site passes through.

    Returns log-scale estimate, lo, hi, tau2, fixed, fixed_lo, fixed_hi.
    """
    est, se = np.asarray(est, float), np.asarray(se, float)
    if len(est) == 1:
        lo, hi = est[0] - 1.96 * se[0], est[0] + 1.96 * se[0]
        return {"estimate": est[0], "lo": lo, "hi": hi, "tau2": 0.0, "fixed": est[0], "fixed_lo": lo, "fixed_hi": hi}
    with warnings.catch_warnings():  # per-site CIs need nobs for t; we use only the pooled rows
        warnings.simplefilter("ignore", UserWarning)
        re = combine_effects(est, se**2, method_re="pm", use_t=True)
        r = re.summary_frame().loc["random effect"]
    fe = combine_effects(est, se**2, method_re="pm").summary_frame().loc["fixed effect"]
    return {"estimate": r["eff"], "lo": r["ci_low"], "hi": r["ci_upp"], "tau2": float(re.tau2),
            "fixed": fe["eff"], "fixed_lo": fe["ci_low"], "fixed_hi": fe["ci_upp"]}

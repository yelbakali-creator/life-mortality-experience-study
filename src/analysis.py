"""Actual-to-expected analysis, credibility and Poisson GLM."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

FULL_CREDIBILITY_DEATHS = 1082   # limited fluctuation: 90% probability, +/-5% error

AGE_BINS = [20, 30, 40, 50, 60, 120]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60+"]


def add_bands(exp):
    exp = exp.copy()
    exp["age_band"] = pd.cut(exp["age"], AGE_BINS, right=False, labels=AGE_LABELS).astype(str)
    exp["duration_band"] = np.where(exp["duration"] >= 3, "3+", exp["duration"].astype(str))
    exp["sa_band"] = np.where(exp["sum_assured"] > 300_000, ">300k", "<=300k")
    exp["actual_amount"] = exp["death"] * exp["sum_assured"]
    exp["expected_amount"] = exp["expected"] * exp["sum_assured"]
    return exp


def poisson_ci(deaths, expected, level=0.95):
    """Exact Poisson confidence interval for the A/E ratio."""
    alpha = 1 - level
    deaths = np.asarray(deaths, dtype=float)
    lower = np.where(deaths > 0, chi2.ppf(alpha / 2, 2 * deaths) / 2, 0.0)
    upper = chi2.ppf(1 - alpha / 2, 2 * (deaths + 1)) / 2
    return lower / expected, upper / expected


def ae_table(exp, by):
    """A/E by count and by amount for one or several grouping variables."""
    g = exp.groupby(by, observed=True).agg(
        exposure_years=("exposure", "sum"),
        actual=("death", "sum"),
        expected=("expected", "sum"),
        actual_amount=("actual_amount", "sum"),
        expected_amount=("expected_amount", "sum"),
    ).reset_index()
    g["ae_count"] = g["actual"] / g["expected"]
    g["ci_low"], g["ci_high"] = poisson_ci(g["actual"], g["expected"])
    g["ae_amount"] = g["actual_amount"] / g["expected_amount"]
    g["credibility"] = np.minimum(1.0, np.sqrt(g["actual"] / FULL_CREDIBILITY_DEATHS))
    # Credibility-weighted assumption, blended with 100% of the reference table
    g["assumption"] = g["credibility"] * g["ae_count"] + (1 - g["credibility"]) * 1.0
    return g


def fit_glm(exp):
    """Poisson GLM on aggregated cells, offset by log(expected deaths).

    exp(coefficient) = multiplicative effect on the reference table, relative
    to the base cell (age 40-49, male, non-smoker, policy year 3+, SA <= 300k).
    exp(intercept) = level of mortality of the base cell versus the table.
    """
    cells = exp.groupby(["age_band", "sex", "smoker", "duration_band", "sa_band"],
                        observed=True).agg(deaths=("death", "sum"),
                                           expected=("expected", "sum")).reset_index()
    cells = cells[cells["expected"] > 0]
    formula = ("deaths ~ C(age_band, Treatment('40-49')) + C(sex, Treatment('M'))"
               " + C(smoker, Treatment('N')) + C(duration_band, Treatment('3+'))"
               " + C(sa_band, Treatment('<=300k'))")
    model = smf.glm(formula, data=cells, family=sm.families.Poisson(),
                    offset=np.log(cells["expected"])).fit()
    ci = model.conf_int()
    factors = pd.DataFrame({
        "term": model.params.index,
        "factor": np.exp(model.params.values),
        "ci_low": np.exp(ci[0].values),
        "ci_high": np.exp(ci[1].values),
        "p_value": model.pvalues.values,
    })
    return model, factors

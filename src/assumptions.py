"""Mortality bases used in the study.

Two bases are defined:

* The REFERENCE table: a Gompertz-Makeham law standing in for an industry
  table. It is the "Expected" basis of the experience study.
* The TRUE mortality of the simulated portfolio: the reference table adjusted
  by hidden multipliers (insured-lives effect, smoker status, select period,
  sum assured). The study never reads these values; it must recover them
  from the data, which is how the method is validated.
"""
import numpy as np

# Gompertz-Makeham parameters (A, B, c): mu(x) = A + B * c**x
MAKEHAM = {
    "M": (0.00020, 0.000030, 1.10),
    "F": (0.00010, 0.000018, 1.10),
}

# Hidden multipliers applied to the reference force of mortality
TRUE_INSURED_LEVEL = 0.85                       # insured lives vs reference population
TRUE_SMOKER = {"N": 1.00, "S": 1.90}
TRUE_DURATION = {0: 0.60, 1: 0.75, 2: 0.90}     # select period; 1.00 from policy year 3
TRUE_HIGH_SA = 0.85                             # large sums assured: healthier lives
HIGH_SA_THRESHOLD = 300_000


def reference_mu(age, sex):
    """Force of mortality of the reference table at (possibly fractional) age."""
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex)
    a = np.where(sex == "M", MAKEHAM["M"][0], MAKEHAM["F"][0])
    b = np.where(sex == "M", MAKEHAM["M"][1], MAKEHAM["F"][1])
    c = np.where(sex == "M", MAKEHAM["M"][2], MAKEHAM["F"][2])
    return a + b * c ** age


def true_multiplier(smoker, duration, sum_assured):
    """Hidden multiplier of the portfolio's true mortality versus the reference."""
    smoker = np.asarray(smoker)
    duration = np.asarray(duration)
    sum_assured = np.asarray(sum_assured)
    m = np.full(smoker.shape, TRUE_INSURED_LEVEL, dtype=float)
    m *= np.where(smoker == "S", TRUE_SMOKER["S"], TRUE_SMOKER["N"])
    dur_factor = np.ones(duration.shape)
    for d, f in TRUE_DURATION.items():
        dur_factor = np.where(duration == d, f, dur_factor)
    m *= dur_factor
    m *= np.where(sum_assured > HIGH_SA_THRESHOLD, TRUE_HIGH_SA, 1.0)
    return m

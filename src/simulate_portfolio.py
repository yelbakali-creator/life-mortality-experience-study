"""Simulate an individual life insurance portfolio, one row per policy.

Each policy has a date of birth, sex, smoker status, sum assured and issue
date. Policies are projected month by month from issue until death, lapse or
the end of the data extract. The output looks like a real administration
extract: dates and an exit reason, no exposure and no ages pre-computed.
"""
import numpy as np
import pandas as pd

from .assumptions import reference_mu, true_multiplier

EXTRACT_END = pd.Timestamp("2023-12-31")
FIRST_ISSUE = pd.Timestamp("2012-01-01")


def _lapse_monthly(duration):
    """Monthly lapse probability: 8% a year in the first two policy years, 5% after."""
    annual = np.where(duration < 2, 0.08, 0.05)
    return 1 - (1 - annual) ** (1 / 12)


def simulate_portfolio(n_policies=100_000, seed=2026):
    rng = np.random.default_rng(seed)
    n_months = (EXTRACT_END.year - FIRST_ISSUE.year) * 12 + EXTRACT_END.month - FIRST_ISSUE.month + 1

    # Static policy characteristics
    issue_month = rng.integers(0, n_months - 6, n_policies)          # no issues in the last 6 months
    issue_day = rng.integers(1, 29, n_policies)
    month_starts = pd.period_range(FIRST_ISSUE, EXTRACT_END, freq="M").to_timestamp()
    issue_date = pd.DatetimeIndex(month_starts.values[issue_month]) + pd.to_timedelta(issue_day - 1, unit="D")

    sex = np.where(rng.random(n_policies) < 0.55, "M", "F")
    smoker_rate = np.where(sex == "M", 0.17, 0.13)
    smoker = np.where(rng.random(n_policies) < smoker_rate, "S", "N")
    issue_age = np.clip(rng.normal(40, 10, n_policies), 20, 65)
    dob = issue_date - pd.to_timedelta(np.round(issue_age * 365.25), unit="D")
    sum_assured = np.clip(np.round(rng.lognormal(np.log(150_000), 0.7, n_policies), -3), 25_000, 2_000_000)

    # Monthly projection
    status = np.zeros(n_policies, dtype=np.int8)        # 0 not issued / in force, 1 death, 2 lapse
    exit_month = np.full(n_policies, -1)
    dob_days = dob.values.astype("datetime64[D]").astype(np.int64)

    for m in range(n_months):
        active = (issue_month <= m) & (status == 0)
        if not active.any():
            continue
        idx = np.flatnonzero(active)
        mid_month = np.datetime64(month_starts[m] + pd.Timedelta(days=14), "D").astype(np.int64)
        age = (mid_month - dob_days[idx]) / 365.25
        duration = (m - issue_month[idx]) // 12
        mu = reference_mu(age, sex[idx]) * true_multiplier(smoker[idx], duration, sum_assured[idx])
        p_death = 1 - np.exp(-mu / 12)
        p_lapse = _lapse_monthly(duration)
        u = rng.random(idx.size)
        died = u < p_death
        lapsed = (~died) & (u < p_death + p_lapse)
        status[idx[died]] = 1
        status[idx[lapsed]] = 2
        exit_month[idx[died | lapsed]] = m

    # Exit dates: a random day within the exit month, never before the issue date
    exited = status > 0
    exit_date = pd.Series(pd.NaT, index=range(n_policies), dtype="datetime64[ns]")
    days_in_month = month_starts.days_in_month.values
    em = exit_month[exited]
    offsets = (rng.random(em.size) * days_in_month[em]).astype(int)
    dates = month_starts.values[em] + offsets.astype("timedelta64[D]")
    dates = np.maximum(dates, issue_date.values[exited] + np.timedelta64(1, "D"))
    exit_date[exited] = dates

    return pd.DataFrame({
        "policy_id": np.arange(1, n_policies + 1),
        "sex": sex,
        "smoker": smoker,
        "date_of_birth": dob.normalize(),
        "issue_date": issue_date.normalize(),
        "sum_assured": sum_assured.astype(int),
        "exit_date": exit_date.values,
        "exit_reason": np.select([status == 1, status == 2], ["death", "lapse"], "in force"),
    })

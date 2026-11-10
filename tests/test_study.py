"""Validation tests: exposure engine reconciliation and edge cases, A/E machinery."""
import numpy as np
import pandas as pd

from src.analysis import add_bands, ae_table, poisson_ci
from src.exposure import compute_exposure
from src.simulate_portfolio import simulate_portfolio


def test_deaths_reconcile_with_extract():
    pol = simulate_portfolio(20_000, seed=3)
    exp = compute_exposure(pol)
    raw = ((pol["exit_reason"] == "death") & pol["exit_date"].between("2019-01-01", "2023-12-31")).sum()
    assert exp["death"].sum() == raw


def test_death_on_first_day_of_window_is_counted():
    pol = pd.DataFrame({"policy_id": [1], "sex": ["M"], "smoker": ["N"], "date_of_birth": [pd.Timestamp("1970-06-15")],
                        "issue_date": [pd.Timestamp("2015-03-01")], "sum_assured": [100_000],
                        "exit_date": [pd.Timestamp("2019-01-01")], "exit_reason": ["death"]})
    exp = compute_exposure(pol)
    assert exp["death"].sum() == 1 and exp["exposure"].sum() > 0


def test_exposure_never_exceeds_window():
    pol = simulate_portfolio(5_000, seed=4)
    per_policy = compute_exposure(pol).groupby("policy_id")["exposure"].sum()
    assert per_policy.max() <= 1826 / 365.25 + 1e-9          # 1 Jan 2019 to 31 Dec 2023 inclusive


def test_poisson_interval_contains_point_estimate():
    lo, hi = poisson_ci(np.array([50.0]), np.array([60.0]))
    assert lo[0] < 50 / 60 < hi[0]


def test_ae_table_totals():
    exp = add_bands(compute_exposure(simulate_portfolio(5_000, seed=5)))
    t = ae_table(exp, "sex")
    assert t["actual"].sum() == exp["death"].sum()
    assert abs(t["expected"].sum() - exp["expected"].sum()) < 1e-6

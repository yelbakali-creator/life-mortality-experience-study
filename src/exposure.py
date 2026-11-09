"""Central exposure calculation from policy-level dates.

For each policy, the time spent in force inside the observation window is cut
into segments at every birthday and every policy anniversary. Each segment
therefore has a single attained age (age last birthday) and a single policy
year. A death is counted in the segment where it occurs.

Expected deaths use the reference force of mortality at mid-age:
    expected = central exposure (years) * mu_ref(age + 0.5)
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .assumptions import reference_mu

OBS_START = date(2019, 1, 1)
OBS_END = date(2023, 12, 31)          # inclusive


def _add_years(d, k):
    """Same calendar day k years later; 29 February falls back to 28 February."""
    try:
        return d.replace(year=d.year + k)
    except ValueError:
        return d.replace(year=d.year + k, day=28)


def _completed_years(origin, d):
    """Whole years elapsed between origin and d (age last birthday, policy year)."""
    years = d.year - origin.year
    if _add_years(origin, years) > d:
        years -= 1
    return years


def compute_exposure(policies, obs_start=OBS_START, obs_end=OBS_END):
    window_end = obs_end + timedelta(days=1)        # exclusive bound
    rows = []
    for p in policies.itertuples(index=False):
        dob = p.date_of_birth.date()
        issue = p.issue_date.date()
        exit_d = None if pd.isna(p.exit_date) else p.exit_date.date()

        start = max(issue, obs_start)
        # The exit day itself is in force: a death on day d is exposed on day d
        end = min(exit_d + timedelta(days=1), window_end) if exit_d is not None else window_end
        if end <= start:
            continue
        died_in_window = p.exit_reason == "death" and exit_d is not None and obs_start <= exit_d <= obs_end

        # Breakpoints: birthdays and policy anniversaries strictly inside (start, end)
        cuts = {start, end}
        for origin in (dob, issue):
            k = _completed_years(origin, start) + 1
            nxt = _add_years(origin, k)
            while nxt < end:
                cuts.add(nxt)
                k += 1
                nxt = _add_years(origin, k)
        cuts = sorted(cuts)

        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            rows.append((
                p.policy_id, p.sex, p.smoker, p.sum_assured,
                _completed_years(dob, a),            # attained age
                _completed_years(issue, a),          # policy year (0 = first year)
                (b - a).days / 365.25,               # central exposure in years
                int(died_in_window and i == len(cuts) - 2),
            ))

    exp = pd.DataFrame(rows, columns=["policy_id", "sex", "smoker", "sum_assured",
                                      "age", "duration", "exposure", "death"])
    exp["expected"] = exp["exposure"] * reference_mu(exp["age"] + 0.5, exp["sex"])
    return exp

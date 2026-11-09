"""Run the full mortality experience study: data, exposure, A/E, GLM, charts."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import assumptions as true_basis
from src.analysis import add_bands, ae_table, fit_glm
from src.exposure import compute_exposure
from src.simulate_portfolio import simulate_portfolio
from src.assumptions import reference_mu

ROOT = Path(__file__).parent
DATA, TABLES, FIGS = ROOT / "data", ROOT / "results" / "tables", ROOT / "results" / "figures"
for folder in (DATA, TABLES, FIGS):
    folder.mkdir(parents=True, exist_ok=True)

NAVY, RED, GREY = "#1F3864", "#C0392B", "#7F8C8D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def clean_term(term):
    if term == "Intercept":
        return "Base level (age 40-49, male, non-smoker, year 3+, SA <= 300k)"
    var, level = re.match(r"C\((\w+).*\)\[T\.(.+)\]", term).groups()
    names = {"age_band": "Age", "sex": "Sex", "smoker": "Smoker", "duration_band": "Policy year",
             "sa_band": "Sum assured"}
    return f"{names[var]}: {level}"


def main():
    # 1. Data
    policies = simulate_portfolio(n_policies=500_000, seed=2026)
    policies.to_csv(DATA / "policies.csv", index=False)

    # 2. Exposure, with a reconciliation check against the raw extract
    exp = add_bands(compute_exposure(policies))
    raw_deaths = ((policies["exit_reason"] == "death")
                  & policies["exit_date"].between("2019-01-01", "2023-12-31")).sum()
    assert exp["death"].sum() == raw_deaths, "Deaths do not reconcile with the extract"

    # 3. A/E tables
    total = ae_table(exp.assign(portfolio="Total"), "portfolio")
    tables = {by: ae_table(exp, by) for by in ["age_band", "sex", "smoker", "duration_band", "sa_band"]}
    total.to_csv(TABLES / "ae_total.csv", index=False)
    for by, t in tables.items():
        t.to_csv(TABLES / f"ae_by_{by}.csv", index=False)

    # 4. Poisson GLM and validation against the hidden true basis
    model, factors = fit_glm(exp)
    factors["term"] = factors["term"].map(clean_term)
    factors.to_csv(TABLES / "glm_factors.csv", index=False)
    truth = {
        "Base level (age 40-49, male, non-smoker, year 3+, SA <= 300k)": true_basis.TRUE_INSURED_LEVEL,
        "Smoker: S": true_basis.TRUE_SMOKER["S"],
        "Policy year: 0": true_basis.TRUE_DURATION[0],
        "Policy year: 1": true_basis.TRUE_DURATION[1],
        "Policy year: 2": true_basis.TRUE_DURATION[2],
        "Sum assured: >300k": true_basis.TRUE_HIGH_SA,
        "Sex: F": 1.0,
    }
    validation = factors[factors["term"].isin(truth)].copy()
    validation["true_value"] = validation["term"].map(truth)
    validation["within_95_ci"] = validation["true_value"].between(validation["ci_low"], validation["ci_high"])
    validation.to_csv(TABLES / "glm_validation.csv", index=False)

    # 5. Charts
    t = tables["age_band"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(t["age_band"], t["ae_count"], color=NAVY, width=0.6)
    ax.errorbar(t["age_band"], t["ae_count"], yerr=[t["ae_count"] - t["ci_low"], t["ci_high"] - t["ae_count"]],
                fmt="none", ecolor="black", capsize=4, lw=1)
    ax.axhline(1, color=RED, ls="--", lw=1, label="100% of reference table")
    for x, (v, n) in enumerate(zip(t["ae_count"], t["actual"])):
        ax.text(x, 0.05, f"{v:.0%}\n{n} deaths", ha="center", color="white", fontsize=8)
    ax.set_ylabel("Actual / Expected (by count)")
    ax.set_title("A/E by attained age, with 95% confidence intervals")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "ae_by_age.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    for ax, by, title in zip(axes, ["sex", "smoker", "duration_band", "sa_band"],
                             ["Sex", "Smoker status", "Policy year", "Sum assured"]):
        t = tables[by]
        labels = t[by].replace({"F": "Female", "M": "Male", "N": "Non-smoker", "S": "Smoker"})
        ax.bar(labels, t["ae_count"], color=NAVY, width=0.6, label="By count")
        ax.scatter(labels, t["ae_amount"], color=RED, zorder=3, label="By amount")
        ax.errorbar(labels, t["ae_count"], yerr=[t["ae_count"] - t["ci_low"], t["ci_high"] - t["ae_count"]],
                    fmt="none", ecolor="black", capsize=3, lw=1)
        ax.axhline(1, color=GREY, ls="--", lw=1)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    axes[0].set_ylabel("Actual / Expected")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("A/E by segment (bars: by count with 95% CI; dots: by amount)")
    fig.tight_layout()
    fig.savefig(FIGS / "ae_by_segment.png", dpi=150)
    plt.close(fig)

    by_age = exp.groupby(["sex", "age"]).agg(deaths=("death", "sum"), exposure=("exposure", "sum")).reset_index()
    by_age = by_age[by_age["exposure"] > 500]
    fig, ax = plt.subplots(figsize=(7, 4))
    for sex, color, label in [("M", NAVY, "Male"), ("F", RED, "Female")]:
        d = by_age[by_age["sex"] == sex]
        ages = np.arange(d["age"].min(), d["age"].max() + 1)
        ax.scatter(d["age"], d["deaths"] / d["exposure"], color=color, s=12, alpha=0.6, label=f"{label}: observed")
        ax.plot(ages, reference_mu(ages + 0.5, sex), color=color, lw=1.5, label=f"{label}: reference table")
    ax.set_yscale("log")
    ax.set_xlabel("Attained age")
    ax.set_ylabel("Mortality rate (log scale)")
    ax.set_title("Observed crude mortality rates vs reference table")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "observed_vs_expected_rates.png", dpi=150)
    plt.close(fig)

    v = validation.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(v))
    ax.errorbar(v["factor"], y, xerr=[v["factor"] - v["ci_low"], v["ci_high"] - v["factor"]],
                fmt="o", color=NAVY, ecolor=NAVY, capsize=3, label="GLM estimate (95% CI)")
    ax.scatter(v["true_value"], y, marker="x", color=RED, s=60, zorder=3, label="True hidden value")
    ax.set_yticks(y, [s.split(" (")[0] for s in v["term"]])
    ax.set_xlabel("Multiplier on reference mortality")
    ax.set_title("Model validation: the GLM recovers the hidden assumptions")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "glm_validation.png", dpi=150)
    plt.close(fig)

    # 6. Console summary
    tot = total.iloc[0]
    print(f"Policies: {len(policies):,} | Exposure: {tot.exposure_years:,.0f} years | "
          f"Deaths: {int(tot.actual):,} | Expected: {tot.expected:,.1f}")
    print(f"A/E by count: {tot.ae_count:.1%} (95% CI {tot.ci_low:.1%} - {tot.ci_high:.1%}) | "
          f"A/E by amount: {tot.ae_amount:.1%}")
    for by, t in tables.items():
        print(f"\n{by}\n" + t[[by, "actual", "expected", "ae_count", "ci_low", "ci_high", "ae_amount",
                                 "credibility", "assumption"]].round(3).to_string(index=False))
    print("\nGLM validation\n" + validation.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

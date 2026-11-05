"""
00_power_analysis.py  (v2 - no statsmodels dependency)
========================================================
Prospective power analysis to determine the minimum N (seeds) needed for each
hypothesis test, BEFORE running the enhanced multi-seed campaign.

NOTE: This version uses only scipy + numpy (no statsmodels) because
statsmodels does not yet ship a prebuilt wheel for very new Python versions
(e.g. 3.15) and fails to build from source without a C/Fortran compiler
toolchain on Windows. The power calculation below is done manually using
the standard paired-t-test power formula, which is exactly what
statsmodels.stats.power.TTestPower does internally anyway.

Why this matters for publication:
  Appendix C.12's "Methodological Limitation" box in the current thesis notes
  that N=8 was chosen retrospectively, not from a prospective power analysis.
  Reviewers at Applied Energy / IEEE Trans. Smart Grid will ask for this.
  Running this script FIRST and reporting its output in the paper closes that
  gap cleanly.

Usage:
    pip install scipy numpy
    python 00_power_analysis.py
"""
import numpy as np
from scipy import stats
from scipy.optimize import brentq


def paired_ttest_power(n, effect_size_d, alpha=0.05):
    """Power of a two-sided paired t-test with n pairs, Cohen's d effect
    size, at significance level alpha. Standard formula (same one
    statsmodels.stats.power.TTestPower uses internally)."""
    df = n - 1
    ncp = effect_size_d * np.sqrt(n)  # non-centrality parameter
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # power = P(|T| > t_crit) under noncentral t with noncentrality ncp
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return power


def n_for_power(effect_size_d, alpha=0.05, target_power=0.80, n_max=500):
    """Solve for the smallest n such that power >= target_power."""
    def f(n):
        return paired_ttest_power(int(np.ceil(n)), effect_size_d, alpha) - target_power

    # Scan upward until we bracket the root (power crosses target_power)
    n_lo, n_hi = 3, 4
    while paired_ttest_power(n_hi, effect_size_d, alpha) < target_power and n_hi < n_max:
        n_lo, n_hi = n_hi, n_hi * 2
    if n_hi >= n_max:
        return n_max  # effect too small to reach target power within n_max
    n_needed = brentq(f, n_lo, n_hi)
    return int(np.ceil(n_needed))


def rank_biserial_to_cohens_d(r):
    """Rough conversion for paired designs: d ~ 2r / sqrt(1 - r^2)."""
    return (2 * r) / np.sqrt(1 - r ** 2)


print("=" * 70)
print("PROSPECTIVE POWER ANALYSIS — for the enhanced statistical campaign")
print("(scipy-only implementation, no statsmodels required)")
print("=" * 70)

# Effect sizes observed in the CURRENT (N=8) results — used here only to
# estimate what N would be needed to confirm them at 80% power.
observed_effects = {
    "H2 - Office (personalisation)":        0.833,   # Table C.10b, large effect
    "H2 - Residential (personalisation)":   0.556,
    "H4 - Residential (FedProx vs FedAvg)": 0.611,
    "H4 - Commercial (FedProx vs FedAvg)":  0.333,
    "H4 - Mixed-use (FedProx vs FedAvg)":   0.444,
}

print(f"\n{'Comparison':<42}{'r (obs.)':<10}{'d (approx)':<12}{'N needed (80% power)'}")
print("-" * 90)
results = {}
for name, r in observed_effects.items():
    d = rank_biserial_to_cohens_d(r)
    # Wilcoxon correction factor (Lehmann): required N under Wilcoxon ~= N_ttest / 0.955
    n_ttest = n_for_power(d)
    n_wilcoxon = int(np.ceil(n_ttest / 0.955))
    results[name] = n_wilcoxon
    print(f"{name:<42}{r:<10.3f}{d:<12.3f}N ~ {n_wilcoxon}")

print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)
max_n = max(results.values())
print(f"""
The largest required N across all comparisons is N ~ {max_n} seeds
(driven by the smallest observed effect, H4-Commercial).

Practical recommendation:
  - Target N = 20 seeds per archetype for H1, H2, H4 (multi_seed_*_test.py).
    This exceeds the requirement for all "large" effects (Office H2) and gets
    close for the "medium" H4 effects -- if H4 remains non-significant at
    N=20, that itself is informative (the true effect is likely too small to
    matter practically, which is a legitimate, reportable finding).
  - If compute allows, N = 30 is the more conventional "safe" target and
    directly answers any reviewer question about power.
  - Report this analysis in the paper's Methods section (Section 5.4 /
    Statistical Protocol) as: "An a priori power analysis (Table X) using
    effect sizes from a preliminary N=8 pilot indicated N=20-30 seeds were
    required to detect the observed effects at 80% power; the confirmatory
    campaign was run at N={{N_YOU_CHOOSE}}."

This directly resolves the "Methodological Limitation: No A Priori Power
Analysis Was Conducted" box currently in Appendix C.12 of the thesis.
""")

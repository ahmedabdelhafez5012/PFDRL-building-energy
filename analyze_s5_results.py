"""Reproduces Tables E.1 and E.2 of the thesis (Scenario S5, operator-acceptance survey).
Usage: python analyze_s5_results.py path\\to\\responses.xlsx   (or .csv)
The responses file is private and is NOT part of the repository."""
import sys
import numpy as np
import pandas as pd
from scipy import stats
path = sys.argv[1]
df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding="utf-8-sig")
q = [c for c in df.columns if str(c).startswith("Q")]
X = df[q].apply(pd.to_numeric, errors="coerce")
n = len(df)
print("N =", n, "| missing ratings:", int(X.isna().sum().sum()))
print("\nTable E.1 (one-sample t-test against the scale midpoint 3)")
for c in q:
    t, p = stats.ttest_1samp(X[c], 3)
    print("%-6s mean %.2f  SD %.2f  t %.2f  p %.2g" % (str(c)[:3], X[c].mean(), X[c].std(ddof=1), t, p))
rm = X.mean(axis=1)
t, p = stats.ttest_1samp(rm, 3)
print("Overall mean %.2f  SD %.2f  t %.2f  p %.2g  (%.1f%% of the scale maximum)" % (rm.mean(), rm.std(ddof=1), t, p, rm.mean() / 5 * 100))
print("Wilcoxon signed-rank on respondent means vs 3: p = %.4g" % stats.wilcoxon(rm - 3).pvalue)
print("\nRespondent level: mean >= 4 for %d of %d; ratings of 4 or 5: %d of %d" % ((rm >= 4).sum(), n, (X >= 4).sum().sum(), X.size))
print("Identical ratings on all six items: %d (all 5: %d)" % ((X.nunique(axis=1) == 1).sum(), (X == 5).all(axis=1).sum()))
for i in range(0, len(q) - 1, 2):
    print("%s identical to %s for every respondent: %s" % (str(q[i])[:2], str(q[i + 1])[:2], bool((X[q[i]] == X[q[i + 1]]).all())))
g = df["Prior BMS/EMS Experience"].astype(str).str.strip()
a, b = rm[g == "Yes"], rm[g == "No"]
u = stats.mannwhitneyu(a, b, alternative="two-sided")
print("\nTable E.2: prior BMS/EMS n=%d mean %.2f (SD %.2f) | none n=%d mean %.2f (SD %.2f) | U=%.1f p=%.4f" % (len(a), a.mean(), a.std(ddof=1), len(b), b.mean(), b.std(ddof=1), u.statistic, u.pvalue))
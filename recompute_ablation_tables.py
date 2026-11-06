"""Recomputes the B1-B4 ablation tables of the thesis from the raw per-seed results (Tables C.13, C.13b and C.14a).
Run from the repository root:  python recompute_ablation_tables.py > ABLATION_TABLES.md"""
import json
import numpy as np
from scipy import stats
A = ["residential", "office", "commercial", "mixed_use"]
L = {"residential": "Residential", "office": "Office", "commercial": "Commercial", "mixed_use": "Mixed-use"}
def load(d): return {a: {k: np.array(v) for k, v in json.load(open(f"results/{d}/{a}_ablation.json")).items()} for a in A}
O, N = load("ablation_b1_b4"), load("ablation_b1_b4_normalised")
def fp(p): return "p<0.0001" if p < 1e-4 else ("p<0.001" if p < 1e-3 else "p=%.3f" % p)
w = lambda x, y: stats.wilcoxon(x, y).pvalue
print("## Table C.13 - mean held-out cost per episode, mean (sample SD), 30 seeds\n\n| Archetype | B1 | B2 | B3 | B4 |\n|---|---|---|---|---|")
for a in A: print("| %s | %s |" % (L[a], " | ".join("%.4f (%.4f)" % (O[a][b].mean(), O[a][b].std(ddof=1)) for b in ("B1", "B2", "B3", "B4"))))
print("\n## Table C.13b - paired Wilcoxon signed-rank p-values (exact, two-sided, 30 pairs)\n\n| Archetype | B4 vs B1 | B4 vs B2 | B4 vs B3 | B2 vs B1 | B3 vs B1 |\n|---|---|---|---|---|---|")
for a in A:
    x = O[a]; print("| %s | %s |" % (L[a], " | ".join(fp(w(x[p], x[q])) for p, q in (("B4", "B1"), ("B4", "B2"), ("B4", "B3"), ("B2", "B1"), ("B3", "B1")))))
print("\n## Table C.14a - B4-B1 cost gap, original vs reward-normalised (mean gap, Wilcoxon p)\n\n| Archetype | original | normalised |\n|---|---|---|")
for a in A:
    go = O[a]["B4"] - O[a]["B1"]; gn = N[a]["B4"] - N[a]["B1"]
    print("| %s | %+.3f (%s) | %+.3f (%s) |" % (L[a], go.mean(), fp(stats.wilcoxon(go).pvalue), gn.mean(), fp(stats.wilcoxon(gn).pvalue)))
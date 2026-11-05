import json, numpy as np
from scipy.stats import wilcoxon
for a in ["residential", "office", "commercial", "mixed_use"]:
    d = json.load(open("results/ablation_b1_b4/" + a + "_ablation.json"))
    n = min(len(d["B1"]), len(d["B2"]), len(d["B3"]))
    b1 = np.array(d["B1"][:n]); b2 = np.array(d["B2"][:n]); b3 = np.array(d["B3"][:n])
    r21 = round(100 * np.mean((b2 - b1) / b1), 2); p21 = round(wilcoxon(b2, b1).pvalue, 4)
    r32 = round(100 * np.mean((b3 - b2) / b2), 2); p32 = round(wilcoxon(b3, b2).pvalue, 4)
    print(a, "n =", n, "| B2 vs B1 %:", r21, "p =", p21, "| B3 vs B2 %:", r32, "p =", p32)
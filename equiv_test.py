import argparse, glob, json, os
import numpy as np
from scipy import stats
ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="results/fed_multi/frac0.2"); ap.add_argument("--margin", type=float, default=2.0)
a = ap.parse_args()
def load(folder):
    d = {}
    for f in glob.glob(os.path.join(folder, "*_s*.json")):
        c, s = os.path.basename(f)[:-5].rsplit("_s", 1); d.setdefault(c, {})[int(s)] = json.load(open(f))
    return d
d = load(a.folder); keys = sorted(next(iter(d["indep"].values())).keys()); archs = sorted(set(k.split("|")[0] for k in keys))
print("Equivalence analysis vs independent training | folder:", a.folder, "| margin: +/-%.1f%%" % a.margin)
print("(paired per-seed % difference of archetype-mean cost; 90% CI = two one-sided tests at 5%)")
for c in sorted(d):
    if c in ("indep", "b1", "noop"): continue
    ss = sorted(set(d[c]) & set(d["indep"]))
    for arch in archs + ["ALL"]:
        ks = [k for k in keys if (arch == "ALL" or k.startswith(arch + "|")) and all(k in d[c][s] for s in ss)]
        if not ks: continue
        diff = np.array([(np.mean([d[c][s][k]["cost"] for k in ks]) - np.mean([d["indep"][s][k]["cost"] for k in ks])) / np.mean([d["indep"][s][k]["cost"] for k in ks]) * 100 for s in ss])
        n = len(diff); m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.95, n - 1)
        lo, hi = m - t * se, m + t * se
        verdict = "EQUIVALENT (within +/-%.1f%%)" % a.margin if (lo > -a.margin and hi < a.margin) else ("federation worse by more than the margin cannot be ruled out" if hi >= a.margin else "inconclusive")
        print("%-9s %-12s n=%d  mean %+6.2f%%  90%% CI [%+6.2f, %+6.2f]  max plausible saving %5.2f%%  -> %s" % (c, arch, n, m, lo, hi, max(0.0, -lo), verdict))
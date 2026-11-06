import glob, json, os
import numpy as np
from scipy.stats import spearmanr
F = "results/fed_multi/frac0.2"; N = "results/fed_multi/norm.json"
norm = json.load(open(N))
def std_of(key):
    arch, bid = key.split("|")
    for p, v in norm.items():
        if ("%s_%s_" % (arch, bid)) in p: return v[1]
    return None
def load(cond):
    out = {}
    for f in glob.glob(os.path.join(F, cond + "_s*.json")):
        s = int(os.path.basename(f)[:-5].rsplit("_s", 1)[1]); out[s] = json.load(open(f))
    return out
cr, ir, cn = load("central_raw"), load("indep_raw"), load("central_norm")
ss = sorted(set(cr) & set(ir) & set(cn)); keys = sorted(cr[ss[0]].keys())
rows = []
for k in keys:
    pen = np.nanmean([(cr[s][k]["cost"] - ir[s][k]["cost"]) / ir[s][k]["cost"] * 100 for s in ss])
    fix = np.nanmean([(cn[s][k]["cost"] - cr[s][k]["cost"]) / cr[s][k]["cost"] * 100 for s in ss])
    rows.append((std_of(k), k, pen, fix))
rows.sort(key=lambda r: r[0])
print("building (sorted by reward SD)             reward SD   pooling penalty %   normalisation effect %")
for sd, k, pen, fix in rows: print("%-40s %8.3f %12.2f %18.2f" % (k, sd, pen, fix))
rho, p = spearmanr([np.log(r[0]) for r in rows], [r[2] for r in rows]); print("\nSpearman rho (log reward SD vs pooling penalty): %.2f  p = %.4f  (n = %d buildings, %d seeds)" % (rho, p, len(rows), len(ss)))
rho2, p2 = spearmanr([np.log(r[0]) for r in rows], [r[3] for r in rows]); print("Spearman rho (log reward SD vs normalisation effect): %.2f  p = %.4f" % (rho2, p2))
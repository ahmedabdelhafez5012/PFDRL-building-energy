"""
dp_summary.py
Analysis only (no training). Reads the per-seed results written by exp_dp.py and produces:
  1. an equivalence (TOST) analysis of the privacy noise: dp_eps<E> versus clipping only
     (same configuration, same training budget), paired per-seed % differences, 90% CI;
  2. the summary figure: discomfort change versus independent training against the
     expected noise-to-signal ratio, one point per configuration and epsilon.

Usage:
  python dp_summary.py --margin 2            # cost margin in %, as in equiv_test.py
  python dp_summary.py --folder results/fed_dp_all_server_head_ft400/frac0.2
"""
import argparse, glob, json, math, os
import numpy as np
from scipy import stats

D_FULL, D_HEAD = 9667, 195          # shared parameters: whole actor+critic / final layers only
CONFIGS = [  # folder, label, d, clients per federation, marker
    ("results/fed_dp/frac0.2",                         "A: whole model, 3 clients, client-side noise", D_FULL, 3, "o"),
    ("results/fed_dp_server/frac0.2",                  "Whole model, 3 clients, trusted aggregator",   D_FULL, 3, "s"),
    ("results/fed_dp_all/frac0.2",                     "Whole model, 12 clients, client-side noise",   D_FULL, 12, "D"),
    ("results/fed_dp_head/frac0.2",                    "Control: last layer, 3 clients, client-side",  D_HEAD, 3, "x"),
    ("results/fed_dp_all_server_head/frac0.2",         "Last layer, 12 clients, trusted aggregator",   D_HEAD, 12, "^"),
    ("results/fed_dp_all_server_head_ft400/frac0.2",   "B: last layer, 12 clients, aggregator + fine-tune", D_HEAD, 12, "*"),
    ("results/fed_dp_res16_server_head_ft400/frac0.2", "C: last layer, 16 residential, aggregator + fine-tune", D_HEAD, 16, "P"),
    ("results/fed_dp_res16tz_server_head_ft400/frac0.2", "C': as C, corrected residential HVAC timing", D_HEAD, 16, "h"),
]


def load(folder):
    d = {}
    for f in glob.glob(os.path.join(folder, "*_s*.json")):
        c, s = os.path.basename(f)[:-5].rsplit("_s", 1); d.setdefault(c, {})[int(s)] = json.load(open(f))
    return d


def baseline_dir(folder):
    """Configurations on the 12-building set read baselines from fed_multi; res16 stores its own."""
    return folder if "res16" in folder else "results/fed_multi/frac0.2"


def per_seed(x, y, metric, keys):
    ss = sorted(set(x) & set(y)); out = []
    for s in ss:
        xm = np.mean([x[s][k][metric] for k in keys]); ym = np.mean([y[s][k][metric] for k in keys])
        out.append(100 * (xm - ym) / ym)
    return np.array(out)


def keys_of(d):
    any_cond = next(iter(d.values())); any_seed = next(iter(any_cond.values()))
    return sorted(k for k in any_seed if not k.startswith("_"))


def tost(folder, eps, margin):
    d = load(folder)
    a, b = "dp_eps%g" % eps, "fedclip"
    if a not in d or b not in d: print("missing", a, "or", b, "in", folder); return
    keys = keys_of(d); archs = sorted(set(k.split("|")[0] for k in keys))
    print("\nEquivalence of the privacy noise | %s vs %s | %s" % (a, b, folder))
    print("paired per-seed %% difference, 90%% CI (= TOST at 5%%); cost margin +/-%.1f%%" % margin)
    for metric in ("cost", "disc"):
        for arch in archs + ["ALL"]:
            ks = [k for k in keys if arch == "ALL" or k.startswith(arch + "|")]
            v = per_seed(d[a], d[b], metric, ks); n = len(v)
            if n < 3: continue
            m = v.mean(); se = v.std(ddof=1) / math.sqrt(n); t = stats.t.ppf(0.95, n - 1)
            lo, hi = m - t * se, m + t * se
            smallest = max(abs(lo), abs(hi))          # smallest symmetric margin for which equivalence holds
            if metric == "cost":
                verdict = "EQUIVALENT within +/-%.1f%%" % margin if (lo > -margin and hi < margin) else "not shown at +/-%.1f%%" % margin
            else:
                verdict = "no pre-specified margin; equivalence holds only for margins >= %.1f%%" % smallest
            print("  %-10s %-12s n=%2d  mean %+7.2f%%  90%% CI [%+7.2f, %+7.2f]  -> %s" % (metric, arch, n, m, lo, hi, verdict))


def nsr(z, d, n, mode):
    return z * math.sqrt(d) / (math.sqrt(n) if mode == "client" else n)


def figure(out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for folder, label, dsh, n, mk in CONFIGS:
        if not os.path.isdir(folder): continue
        d = load(folder); base = load(baseline_dir(folder))
        if "indep" not in base: continue
        keys = [k for k in keys_of(d) if all(k in base["indep"][s] for s in base["indep"])]
        xs, ys, es = [], [], []
        for c in sorted(d):
            if not c.startswith("dp_eps"): continue
            meta = next(iter(d[c].values()))["_meta"]
            v = per_seed(d[c], base["indep"], "disc", keys)
            x = nsr(meta["z"], dsh, n, meta["mode"])
            xs.append(x); ys.append(v.mean()); es.append(v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0)
            rows.append((label, c[6:], len(v), x, v.mean()))
        o = np.argsort(xs)
        ax.errorbar(np.array(xs)[o], np.array(ys)[o], yerr=np.array(es)[o], marker=mk, ms=7, capsize=2, lw=1, label=label)
    ax.set_xscale("log"); ax.axhline(0, color="grey", lw=0.8); ax.axvline(1, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Expected noise-to-signal ratio on the averaged update (log scale)")
    ax.set_ylabel("Thermal discomfort vs independent training (%)")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print("\nFigure saved to", out_png)
    print("%-55s %6s %5s %9s %12s" % ("configuration", "eps", "seeds", "NSR", "discomfort %"))
    for r in sorted(rows, key=lambda r: r[3]): print("%-55s %6s %5d %9.1f %+12.1f" % r)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results/fed_dp_all_server_head_ft400/frac0.2")
    ap.add_argument("--eps", type=float, default=8); ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--fig", default="results/dp_nsr_figure.png")
    a = ap.parse_args()
    tost(a.folder, a.eps, a.margin)
    figure(a.fig)

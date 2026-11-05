"""
13_resolve_open_items.py  (no torch; reads only your existing result files)
Resolves two open questions from the audit:
  (1) Which mu_hat* did the N=30 H2 run use, and where does the repeated 3.424 (Tables C.7 / C.10) come from?
  (2) Do the stored N=30 results agree with the numbers printed in the thesis?
Run from the folder that holds results/ :   python 13_resolve_open_items.py
"""
import json, os, numpy as np
ext = "results/extended_n/h2_results.json"; s34 = "results/multiseed_s3_s4/s3_multiseed.json"
names = ["residential", "office", "commercial", "mixed-use"]
thesis_C10 = {"residential": (3.424, 3.468), "office": (542.674, 554.247), "commercial": (321.814, 325.740), "mixed-use": (1317.192, 1273.853)}
thesis_C7  = {"residential": (0.20, 3.424), "office": (0.05, 525.554), "commercial": (0.00, 339.042), "mixed-use": (0.80, 1259.015)}
if os.path.exists(ext):
    h = json.load(open(ext)); print("== H2 file:", ext, "(n =", len(h[f"{names[0]}_pers"]), ")")
    for a in names:
        p, n = np.mean(h[f"{a}_pers"]), np.mean(h[f"{a}_nonpers"]); tp, tn = thesis_C10[a]
        print(f"  {a:12s} file: pers={p:10.3f} nonpers={n:10.3f} | thesis C.10: {tp:10.3f} / {tn:10.3f} | {'MATCH' if abs(p-tp)<0.002 and abs(n-tn)<0.002 else 'DIFFERENT'}")
else: print("missing", ext)
if os.path.exists(s34):
    s = json.load(open(s34)); print("\n== S3 N=5 file:", s34)
    for a in names:
        means = {mu: np.mean(v) for mu, v in s[a].items()}; best = min(means, key=means.get); sd = np.std(s[a][best])
        mu_t, c_t = thesis_C7[a]
        print(f"  {a:12s} file: mu_hat*={best:>5s} cost={means[best]:10.3f} (sd {sd:.3f}) | thesis C.7: mu_hat*={mu_t} cost={c_t} | {'MATCH' if abs(float(best)-mu_t)<1e-9 and abs(means[best]-c_t)<0.002 else 'DIFFERENT'}")
else: print("missing", s34)
print("\nIf residential pers matches C.10 but S3's residential cost at mu_hat* is not 3.424, then C.7's residential cell was copied from C.10 and must be replaced by the S3 file's value.")
print("Which mu_hat* H2 used: check the MU_HAT_STAR dict in the copy of 02_extended_n_multiseed.py you ran (the uploaded copy has 0.80/0.00/0.01/0.01).")

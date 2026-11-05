import json, numpy as np
h = json.load(open("results/extended_n/h2_results.json"))
s = json.load(open("results/multiseed_s3_s4/s3_multiseed.json"))
p = h["residential_pers"]; n = h["residential_nonpers"]; s3 = s["residential"]["0.2"]
print("H2 residential pers: n =", len(p), "unique =", len(set(np.round(p, 6))), "mean =", round(float(np.mean(p)), 4), "sd =", round(float(np.std(p)), 4))
print("H2 pers first 8:", [round(x, 3) for x in p[:8]])
print("H2 pers last 8 :", [round(x, 3) for x in p[-8:]])
print("H2 residential nonpers: n =", len(n), "unique =", len(set(np.round(n, 6))), "mean =", round(float(np.mean(n)), 4))
print("S3 residential mu=0.2 values:", [round(x, 4) for x in s3], "mean =", round(float(np.mean(s3)), 4), "sd =", round(float(np.std(s3)), 4))
print("H2 pers mean of first 5 seeds:", round(float(np.mean(p[:5])), 4))
for a in ["office", "commercial", "mixed-use"]:
    q = h[a + "_pers"]
    print(a, "pers unique =", len(set(np.round(q, 6))), "of", len(q))
print("S3 residential mean per mu:", {mu: round(float(np.mean(v)), 3) for mu, v in s["residential"].items()})
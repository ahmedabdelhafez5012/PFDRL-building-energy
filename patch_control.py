def sub(s, old, new, name):
    n = s.count(old)
    print(name, "->", n, "match(es)")
    return s.replace(old, new) if n == 1 else s
s = open("exp_multi.py", encoding="utf-8").read()
s = sub(s, 'if cond == "indep": groups = [[c] for c in cl]; shared = False', 'if cond in ("indep", "indep3x"): groups = [[c] for c in cl]; shared = False', "condition")
s = sub(s, 'ags, _ = train_group(gcl, seed, R, steps, norm, True, shared and len(gcl) > 1)', 'ags, _ = train_group(gcl, seed, R, steps * (3 if cond == "indep3x" else 1), norm, True, shared and len(gcl) > 1)', "3x steps")
s = sub(s, 'ap.add_argument("--conditions", nargs="+", default=["indep", "fed_arch", "fed_all"])', 'ap.add_argument("--conditions", nargs="+", default=["indep", "fed_arch", "fed_all"]); ap.add_argument("--only_arch", default="")', "arg")
s = sub(s, 'lazy(); cl = clients(a.k); norm = get_norm(cl); steps = max(100, int(round(a.steps * a.data_frac)))', 'lazy(); cl = clients(a.k)\n    if a.only_arch: cl = [c for c in cl if c["arch"] == a.only_arch]\n    norm = get_norm(cl); steps = max(100, int(round(a.steps * a.data_frac)))', "filter")
s = sub(s, 'ks = [k for k in keys if a == "ALL" or k.startswith(a + "|")]', 'ks = [k for k in keys if (a == "ALL" or k.startswith(a + "|")) and all(k in d[c][s] for s in ss)]\n                if not ks: continue', "analyze filter")
block = '''    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder)
        if "indep3x" in d and "fed_arch" in d:
            print("\\n== control: federation within archetype vs independent training with 3x steps |", folder)
            ss = sorted(set(d["indep3x"]) & set(d["fed_arch"]) & set(d["indep"]))
            ks = sorted(set(d["indep3x"][ss[0]].keys()) & set(d["fed_arch"][ss[0]].keys())) if ss else []
            for a in sorted(set(k.split("|")[0] for k in ks)):
                kk = [k for k in ks if k.startswith(a + "|")]
                x = [np.mean([d["fed_arch"][s][k]["cost"] for k in kk]) for s in ss]; y = [np.mean([d["indep3x"][s][k]["cost"] for k in kk]) for s in ss]
                m, p = pw(x, y); print("fed_arch vs indep3x", a, "(%d buildings, n=%d seeds)" % (len(kk), len(ss)), "cost %:", round(m, 2), "p =", round(p, 4))
                z = [np.mean([d["indep3x"][s][k]["cost"] for k in kk]) for s in ss]; w = [np.mean([d["indep"][s][k]["cost"] for k in kk]) for s in ss]
                m2, p2 = pw(z, w); print("indep3x vs indep", a, "cost %:", round(m2, 2), "p =", round(p2, 4))
'''
s = sub(s, '    cd = load(os.path.join(OUT, "cold"))', block + '    cd = load(os.path.join(OUT, "cold"))', "control block")
open("exp_multi.py", "w", encoding="utf-8").write(s)
e = open("equiv_test.py", encoding="utf-8").read()
e = sub(e, 'ks = [k for k in keys if arch == "ALL" or k.startswith(arch + "|")]', 'ks = [k for k in keys if (arch == "ALL" or k.startswith(arch + "|")) and all(k in d[c][s] for s in ss)]\n        if not ks: continue', "equiv filter")
open("equiv_test.py", "w", encoding="utf-8").write(e)
print("done")
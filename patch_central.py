def sub(s, old, new, name):
    n = s.count(old)
    print(name, "->", n, "match(es)")
    return s.replace(old, new) if n == 1 else s
s = open("exp_multi.py", encoding="utf-8").read()
central = '''def run_central(cl, seed, R, steps, norm):
    """One agent trained on the pooled, interleaved experience of all buildings (centralised pooling)."""
    from ddpg_agent import ReplayBuffer
    T.manual_seed(seed); np.random.seed(seed)
    envs = [make_env(c["path"], seed * 11 + 3 + k, norm) for k, c in enumerate(cl)]
    ag = new_agent(envs[0], R * steps * len(cl)); buf = ReplayBuffer()
    states = [e.reset() for e in envs]; warm = min(WARMUP, steps // 4)
    for r in range(R):
        for i in range(steps):
            for k, env in enumerate(envs):
                rand = (r == 0 and i < warm)
                a = np.random.uniform(-1, 1, env.action_dim) if rand else ag.act(states[k])
                s2, rew, d, _ = env.step(a); buf.push(states[k], a, rew, s2, d); states[k] = env.reset() if d else s2
                if len(buf) > BATCH and not rand: step_train(ag, buf.sample(BATCH))
    res = {}
    for c in cl:
        cost, disc = ev(ag, c["path"], seed); res[c["arch"] + "|" + c["id"]] = {"cost": cost, "disc": disc}
    return res
def run_condition(cond, cl, seed, R, steps, norm):
    if cond == "central_raw": return run_central(cl, seed, R, steps, None)
    if cond == "central_norm": return run_central(cl, seed, R, steps, norm)
    if cond == "indep_raw": norm = None
'''
s = sub(s, 'def run_condition(cond, cl, seed, R, steps, norm):\n', central, "run_condition")
s = sub(s, 'if cond in ("indep", "indep3x"): groups = [[c] for c in cl]; shared = False', 'if cond in ("indep", "indep3x", "indep_raw"): groups = [[c] for c in cl]; shared = False', "indep_raw")
s = sub(s, 'x, y = np.array(x), np.array(y); p = wilcoxon(x, y).pvalue', 'x, y = np.array(x, dtype=float), np.array(y, dtype=float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; p = wilcoxon(x, y).pvalue', "nan-safe pw")
block = '''    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder)
        if "central_raw" in d and "indep_raw" in d:
            print("\\n== reward-scale test: centralised pooling vs independent training |", folder)
            for a_, b_ in (("central_raw", "indep_raw"), ("central_norm", "indep"), ("central_norm", "central_raw"), ("indep_raw", "indep")):
                if a_ not in d or b_ not in d: continue
                ss = sorted(set(d[a_]) & set(d[b_])); keys = sorted(d[a_][ss[0]].keys()); archs = sorted(set(k.split("|")[0] for k in keys))
                for arch in archs + ["ALL"]:
                    ks = [k for k in keys if arch == "ALL" or k.startswith(arch + "|")]
                    x = [np.mean([d[a_][s][k]["cost"] for k in ks]) for s in ss]; y = [np.mean([d[b_][s][k]["cost"] for k in ks]) for s in ss]
                    m, p = pw(x, y); print("%-13s vs %-12s %-12s n=%d seeds  cost %%: %7.2f  p = %.4f" % (a_, b_, arch, len(ss), m, p))
'''
s = sub(s, '    cd = load(os.path.join(OUT, "cold"))', block + '    cd = load(os.path.join(OUT, "cold"))', "analysis block")
open("exp_multi.py", "w", encoding="utf-8").write(s)
print("done")
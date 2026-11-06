import argparse, glob, json, os
import numpy as np
from scipy.stats import wilcoxon
MAN = "data/multi/manifest.json"; OUT = "results/fed_multi"
BATCH = 128; WARMUP = 800; MU_PROX = 0.05
T = None
def lazy():
    global T
    if T is None:
        import torch
        T = torch
def clients(k):
    m = json.load(open(MAN)); out = []; cnt = {}
    for c in m:
        cnt[c["arch"]] = cnt.get(c["arch"], 0)
        if cnt[c["arch"]] < k: out.append(c); cnt[c["arch"]] += 1
    return out
def make_env(path, seed, norm=None):
    from environment import BuildingEnergyEnv
    e = BuildingEnergyEnv(data_path=path, episode_len=96, seed=seed)
    if norm is None: return e
    m, s = norm[path]
    class W:
        def __init__(self): self.e = e
        def reset(self): return self.e.reset()
        def step(self, a):
            x, r, d, i = self.e.step(a); return x, (r - m) / s, d, i
        def __getattr__(self, n): return getattr(self.e, n)
    return W()
def get_norm(cl):
    p = os.path.join(OUT, "norm.json"); out = json.load(open(p)) if os.path.exists(p) else {}
    for c in cl:
        if c["path"] in out: continue
        env = make_env(c["path"], 12345); rs = []
        for _ in range(20):
            env.reset(); d = False
            while not d:
                _, r, d, _ = env.step(np.random.uniform(-1, 1, env.action_dim)); rs.append(r)
        out[c["path"]] = [float(np.mean(rs)), float(np.std(rs) + 1e-8)]
    os.makedirs(OUT, exist_ok=True); json.dump(out, open(p, "w")); return out
def new_agent(env, total):
    from ddpg_agent import DDPGAgent
    return DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4, noise_decay_steps=int(total * 0.6))
def plist(ag): return list(ag.actor.parameters()) + list(ag.critic.parameters())
def flat(ag): return T.cat([p.detach().flatten() for p in plist(ag)])
def setflat(ag, v):
    i = 0
    for p in plist(ag):
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n
    ag.actor_target.load_state_dict(ag.actor.state_dict()); ag.critic_target.load_state_dict(ag.critic.state_dict())
def step_train(ag, b, ref=None):
    import torch.nn.functional as F
    s, a, r, s2, d = b
    with T.no_grad(): y = r + ag.gamma * (1 - d) * ag.critic_target(s2, ag.actor_target(s2))
    cp = list(ag.critic.parameters()); ap = list(ag.actor.parameters()); na = sum(p.numel() for p in ap)
    loss = F.mse_loss(ag.critic(s, a), y)
    if ref is not None:
        i = 0; pen = 0.0
        for p in cp:
            n = p.numel(); pen = pen + ((p - ref[na + i:na + i + n].view_as(p)) ** 2).sum(); i += n
        loss = loss + 0.5 * MU_PROX * pen
    ag.critic_opt.zero_grad(); loss.backward(); ag.critic_opt.step()
    al = -ag.critic(s, ag.actor(s)).mean()
    if ref is not None:
        i = 0; pen = 0.0
        for p in ap:
            n = p.numel(); pen = pen + ((p - ref[i:i + n].view_as(p)) ** 2).sum(); i += n
        al = al + 0.5 * MU_PROX * pen
    ag.actor_opt.zero_grad(); al.backward(); ag.actor_opt.step()
    ag.soft_update(ag.critic_target, ag.critic); ag.soft_update(ag.actor_target, ag.actor)
def rnd(ag, env, buf, steps, warm, ref=None):
    s = env.reset()
    for i in range(steps):
        a = np.random.uniform(-1, 1, env.action_dim) if i < warm else ag.act(s)
        s2, r, d, _ = env.step(a); buf.push(s, a, r, s2, d); s = s2
        if len(buf) > BATCH and i >= warm: step_train(ag, buf.sample(BATCH), ref)
        if d: s = env.reset()
def ev(ag, path, seed, mode="agent"):
    from baseline_b1_rule_based import rule_based_action
    env = make_env(path, 9000 + seed); c = []; q = []
    for _ in range(15):
        s = env.reset(); cc = dd = 0.0; d = False
        while not d:
            a = np.zeros(2) if mode == "noop" else (rule_based_action(s, env.price_scale) if mode == "b1" else ag.act(s, explore=False))
            s, _, d, i = env.step(a); cc += i["cost"]; dd += i["discomfort"]
        c.append(cc); q.append(dd)
    return float(np.mean(c)), float(np.mean(q))
def train_group(cl, seed, R, steps, norm, prox, shared):
    from ddpg_agent import ReplayBuffer
    T.manual_seed(seed); np.random.seed(seed)
    envs = [make_env(c["path"], seed * 11 + 3 + k, norm) for k, c in enumerate(cl)]
    ags = [new_agent(e, R * steps) for e in envs]; bufs = [ReplayBuffer() for _ in cl]
    g = flat(ags[0]).clone()
    if shared:
        for ag in ags: setflat(ag, g)
    for r in range(R):
        vs = []
        for ag, env, buf in zip(ags, envs, bufs):
            rnd(ag, env, buf, steps, min(WARMUP, steps // 4) if r == 0 else 0, g if (shared and prox) else None); vs.append(flat(ag))
        if shared:
            g = T.stack(vs).mean(0)
            for ag in ags: setflat(ag, g)
    return ags, g
def run_central(cl, seed, R, steps, norm):
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
    res = {}
    if cond in ("indep", "indep3x", "indep_raw"): groups = [[c] for c in cl]; shared = False
    elif cond == "fed_arch":
        archs = sorted(set(c["arch"] for c in cl)); groups = [[c for c in cl if c["arch"] == a] for a in archs]; shared = True
    else: groups = [cl]; shared = True
    for gcl in groups:
        ags, _ = train_group(gcl, seed, R, steps * (3 if cond == "indep3x" else 1), norm, True, shared and len(gcl) > 1)
        for c, ag in zip(gcl, ags):
            cost, disc = ev(ag, c["path"], seed); res[c["arch"] + "|" + c["id"]] = {"cost": cost, "disc": disc}
    return res
def cold_one(hold, cl, seed, R, steps, cs, norm):
    from ddpg_agent import ReplayBuffer
    peers = [c for c in cl if c["arch"] == hold["arch"] and c["id"] != hold["id"]]
    if not peers: return None
    _, g = train_group(peers, seed, R, steps, norm, True, True)
    env = make_env(hold["path"], seed * 23 + 7, norm); out = {}
    for name, init in (("warm_start", True), ("scratch", False)):
        T.manual_seed(seed); np.random.seed(seed)
        ag = new_agent(env, cs); buf = ReplayBuffer()
        if init: setflat(ag, g)
        rnd(ag, make_env(hold["path"], seed * 23 + 7, norm), buf, cs, 0 if init else min(WARMUP, cs // 4))
        out[name] = ev(ag, hold["path"], seed)[0]
    return out
def load(folder):
    d = {}
    for f in glob.glob(os.path.join(folder, "*_s*.json")):
        c, s = os.path.basename(f)[:-5].rsplit("_s", 1); d.setdefault(c, {})[int(s)] = json.load(open(f))
    return d
def pw(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; p = wilcoxon(x, y).pvalue if len(x) >= 6 and np.any(x != y) else float("nan")
    return 100 * np.mean((x - y) / y), p
def analyze():
    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder); print("\n==", folder)
        if "indep" not in d: print("no indep"); continue
        keys = sorted(next(iter(d["indep"].values())).keys()); archs = sorted(set(k.split("|")[0] for k in keys))
        for c in sorted(d):
            if c in ("indep", "b1", "noop"): continue
            ss = sorted(set(d[c]) & set(d["indep"]))
            for a in archs + ["ALL"]:
                ks = [k for k in keys if (a == "ALL" or k.startswith(a + "|")) and all(k in d[c][s] for s in ss)]
                if not ks: continue
                x = [np.mean([d[c][s][k]["cost"] for k in ks]) for s in ss]; y = [np.mean([d["indep"][s][k]["cost"] for k in ks]) for s in ss]
                dx = np.array([np.mean([d[c][s][k]["disc"] for k in ks]) for s in ss]); dy = np.array([np.mean([d["indep"][s][k]["disc"] for k in ks]) for s in ss])
                m, p = pw(x, y); print(c, a, "(%d buildings, n=%d seeds)" % (len(ks), len(ss)), "cost vs indep %:", round(m, 2), "p =", round(p, 4), "| discomfort %:", round(float(np.mean((dx - dy) / dy) * 100), 1))
        for r in ("b1", "noop"):
            if r in d:
                ss = sorted(set(d["indep"]) & set(d[r]))
                for a in archs:
                    ks = [k for k in keys if k.startswith(a + "|")]
                    x = [np.mean([d["indep"][s][k]["cost"] for k in ks]) for s in ss]; y = [np.mean([d[r][s][k]["cost"] for k in ks]) for s in ss]
                    print("indep vs", r, a, "cost %:", round(pw(x, y)[0], 2))
    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder)
        if "indep3x" in d and "fed_arch" in d:
            print("\n== control: federation within archetype vs independent training with 3x steps |", folder)
            ss = sorted(set(d["indep3x"]) & set(d["fed_arch"]) & set(d["indep"]))
            ks = sorted(set(d["indep3x"][ss[0]].keys()) & set(d["fed_arch"][ss[0]].keys())) if ss else []
            for a in sorted(set(k.split("|")[0] for k in ks)):
                kk = [k for k in ks if k.startswith(a + "|")]
                x = [np.mean([d["fed_arch"][s][k]["cost"] for k in kk]) for s in ss]; y = [np.mean([d["indep3x"][s][k]["cost"] for k in kk]) for s in ss]
                m, p = pw(x, y); print("fed_arch vs indep3x", a, "(%d buildings, n=%d seeds)" % (len(kk), len(ss)), "cost %:", round(m, 2), "p =", round(p, 4))
                z = [np.mean([d["indep3x"][s][k]["cost"] for k in kk]) for s in ss]; w = [np.mean([d["indep"][s][k]["cost"] for k in kk]) for s in ss]
                m2, p2 = pw(z, w); print("indep3x vs indep", a, "cost %:", round(m2, 2), "p =", round(p2, 4))
    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder)
        if "central_raw" in d and "indep_raw" in d:
            print("\n== reward-scale test: centralised pooling vs independent training |", folder)
            for a_, b_ in (("central_raw", "indep_raw"), ("central_norm", "indep"), ("central_norm", "central_raw"), ("indep_raw", "indep")):
                if a_ not in d or b_ not in d: continue
                ss = sorted(set(d[a_]) & set(d[b_])); keys = sorted(d[a_][ss[0]].keys()); archs = sorted(set(k.split("|")[0] for k in keys))
                for arch in archs + ["ALL"]:
                    ks = [k for k in keys if arch == "ALL" or k.startswith(arch + "|")]
                    x = [np.mean([d[a_][s][k]["cost"] for k in ks]) for s in ss]; y = [np.mean([d[b_][s][k]["cost"] for k in ks]) for s in ss]
                    m, p = pw(x, y); print("%-13s vs %-12s %-12s n=%d seeds  cost %%: %7.2f  p = %.4f" % (a_, b_, arch, len(ss), m, p))
    cd = load(os.path.join(OUT, "cold"))
    ids = sorted(cd)
    if ids:
        archs = sorted(set(i.split("__")[0] for i in ids))
        for a in archs:
            hs = [i for i in ids if i.startswith(a + "__")]; ss = sorted(set.intersection(*[set(cd[h]) for h in hs]))
            w = [np.mean([cd[h][s]["warm_start"] for h in hs]) for s in ss]; sc = [np.mean([cd[h][s]["scratch"] for h in hs]) for s in ss]
            m, p = pw(w, sc); print("cold", a, "(%d held-out buildings, n=%d seeds)" % (len(hs), len(ss)), "warm vs scratch %:", round(m, 2), "p =", round(p, 4))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train"); ap.add_argument("--k", type=int, default=3); ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--data_frac", type=float, default=1.0); ap.add_argument("--cold_steps", type=int, default=600)
    ap.add_argument("--conditions", nargs="+", default=["indep", "fed_arch", "fed_all"]); ap.add_argument("--only_arch", default="")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.mode == "analyze": return analyze()
    lazy(); cl = clients(a.k)
    if a.only_arch: cl = [c for c in cl if c["arch"] == a.only_arch]
    norm = get_norm(cl); steps = max(100, int(round(a.steps * a.data_frac)))
    print("clients:", [(c["arch"], c["id"]) for c in cl], flush=True)
    if a.mode == "train":
        f = os.path.join(OUT, "frac%g" % a.data_frac); os.makedirs(f, exist_ok=True)
        for seed in range(a.n_seeds):
            for ref in ("b1", "noop"):
                p = os.path.join(f, "%s_s%d.json" % (ref, seed))
                if not os.path.exists(p):
                    json.dump({c["arch"] + "|" + c["id"]: dict(zip(("cost", "disc"), ev(None, c["path"], seed, ref))) for c in cl}, open(p, "w"))
            for cond in a.conditions:
                p = os.path.join(f, "%s_s%d.json" % (cond, seed))
                if os.path.exists(p): continue
                print("seed", seed, cond, flush=True); json.dump(run_condition(cond, cl, seed, a.rounds, steps, norm), open(p, "w"))
    else:
        f = os.path.join(OUT, "cold"); os.makedirs(f, exist_ok=True)
        for seed in range(a.n_seeds):
            for h in cl:
                p = os.path.join(f, "%s_s%d.json" % (h["arch"] + "__" + h["id"], seed))
                if os.path.exists(p): continue
                r = cold_one(h, cl, seed, a.rounds, steps, a.cold_steps, norm)
                if r is None: continue
                print("seed", seed, "cold", h["arch"], h["id"], flush=True); json.dump(r, open(p, "w"))
main()
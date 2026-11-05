import argparse, json, os, glob
import numpy as np
from scipy.stats import wilcoxon
PATHS = {"residential": "data/residential_archetype_15min.csv", "office": "data/office_archetype_15min.csv",
         "commercial": "data/commercial_archetype_15min.csv", "mixed_use": "data/mixeduse_archetype_15min.csv"}
ARCHS = list(PATHS)
BATCH = 128; WARMUP = 800; MU_PROX = 0.05; OUT = "results/fed_lean"
T = None
def lazy():
    global T
    if T is None:
        import torch
        T = torch
def make_env(arch, seed, norm=None):
    from environment import BuildingEnergyEnv
    e = BuildingEnergyEnv(data_path=PATHS[arch], episode_len=96, seed=seed)
    if norm is None: return e
    m, s = norm[arch]
    class W:
        def __init__(self): self.e = e
        def reset(self): return self.e.reset()
        def step(self, a):
            x, r, d, i = self.e.step(a); return x, (r - m) / s, d, i
        def __getattr__(self, n): return getattr(self.e, n)
    return W()
def get_norm():
    p = os.path.join(OUT, "norm.json")
    if os.path.exists(p): return json.load(open(p))
    out = {}
    for a in ARCHS:
        env = make_env(a, 12345); rs = []
        for _ in range(20):
            env.reset(); d = False
            while not d:
                _, r, d, _ = env.step(np.random.uniform(-1, 1, env.action_dim)); rs.append(r)
        out[a] = [float(np.mean(rs)), float(np.std(rs) + 1e-8)]
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
def ev(ag, arch, seed, mode="agent"):
    from baseline_b1_rule_based import rule_based_action
    env = make_env(arch, 9000 + seed); c = []; q = []
    for _ in range(15):
        s = env.reset(); cc = dd = 0.0; d = False
        while not d:
            a = np.zeros(2) if mode == "noop" else (rule_based_action(s, env.price_scale) if mode == "b1" else ag.act(s, explore=False))
            s, _, d, i = env.step(a); cc += i["cost"]; dd += i["discomfort"]
        c.append(cc); q.append(dd)
    return {"cost": float(np.mean(c)), "disc": float(np.mean(q))}
def indep(seed, R, steps, norm):
    from ddpg_agent import ReplayBuffer
    out = {}
    for k, a in enumerate(ARCHS):
        T.manual_seed(seed); np.random.seed(seed)
        env = make_env(a, seed * 11 + 3 + k, norm); ag = new_agent(env, R * steps); buf = ReplayBuffer()
        for r in range(R): rnd(ag, env, buf, steps, min(WARMUP, steps // 4) if r == 0 else 0)
        out[a] = ev(ag, a, seed)
    return out
def fed(seed, R, steps, norm, prox, archs=ARCHS):
    from ddpg_agent import ReplayBuffer
    T.manual_seed(seed); np.random.seed(seed)
    envs = [make_env(a, seed * 11 + 3 + ARCHS.index(a), norm) for a in archs]
    ags = [new_agent(e, R * steps) for e in envs]; bufs = [ReplayBuffer() for _ in archs]
    g = flat(ags[0]).clone()
    for ag in ags: setflat(ag, g)
    for r in range(R):
        vs = []
        for ag, env, buf in zip(ags, envs, bufs):
            rnd(ag, env, buf, steps, min(WARMUP, steps // 4) if r == 0 else 0, g if prox else None); vs.append(flat(ag))
        g = T.stack(vs).mean(0)
        for ag in ags: setflat(ag, g)
    return ags, g
def cold(seed, hold, R, steps, cs, norm):
    from ddpg_agent import ReplayBuffer
    others = [a for a in ARCHS if a != hold]
    _, g = fed(seed, R, steps, norm, True, others)
    env = make_env(hold, seed * 23 + 7, norm); out = {}
    z = new_agent(env, 1); setflat(z, g); out["zero_shot"] = ev(z, hold, seed)
    for name, init in (("warm_start", True), ("scratch", False)):
        T.manual_seed(seed); np.random.seed(seed)
        ag = new_agent(env, cs); buf = ReplayBuffer()
        if init: setflat(ag, g)
        rnd(ag, make_env(hold, seed * 23 + 7, norm), buf, cs, 0 if init else min(WARMUP, cs // 4))
        out[name] = ev(ag, hold, seed)
    return out
def load(folder):
    d = {}
    for f in glob.glob(os.path.join(folder, "*_s*.json")):
        c, s = os.path.basename(f)[:-5].rsplit("_s", 1); d.setdefault(c, {})[int(s)] = json.load(open(f))
    return d
def pw(x, y):
    x, y = np.array(x), np.array(y); p = wilcoxon(x, y).pvalue if len(x) >= 6 and np.any(x != y) else float("nan")
    return 100 * np.mean((x - y) / y), p
def analyze():
    for folder in sorted(glob.glob(os.path.join(OUT, "frac*"))):
        d = load(folder); print("\n==", folder)
        if "indep" not in d: print("no indep results"); continue
        for c in sorted(d):
            if c in ("indep", "b1", "noop"): continue
            ss = sorted(set(d[c]) & set(d["indep"]))
            for a in ARCHS:
                x = [d[c][s][a]["cost"] for s in ss]; y = [d["indep"][s][a]["cost"] for s in ss]
                dx = np.array([d[c][s][a]["disc"] for s in ss]); dy = np.array([d["indep"][s][a]["disc"] for s in ss])
                m, p = pw(x, y); print(c, a, "n =", len(ss), "cost vs indep %:", round(m, 2), "p =", round(p, 4), "discomfort vs indep %:", round(float(np.mean((dx - dy) / dy) * 100), 1))
        for r in ("b1", "noop"):
            if r in d:
                for a in ARCHS:
                    ss = sorted(set(d["indep"]) & set(d[r]))
                    x = [d["indep"][s][a]["cost"] for s in ss]; y = [d[r][s][a]["cost"] for s in ss]
                    print("indep vs", r, a, "cost %:", round(pw(x, y)[0], 2))
    cd = load(os.path.join(OUT, "cold"))
    for h in sorted(cd):
        ss = sorted(cd[h]); w = [cd[h][s]["warm_start"]["cost"] for s in ss]; sc = [cd[h][s]["scratch"]["cost"] for s in ss]; z = [cd[h][s]["zero_shot"]["cost"] for s in ss]
        m, p = pw(w, sc); print("cold", h, "n =", len(ss), "warm vs scratch %:", round(m, 2), "p =", round(p, 4), "| zero-shot vs scratch %:", round(pw(z, sc)[0], 2))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train"); ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=15); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--data_frac", type=float, default=1.0); ap.add_argument("--cold_steps", type=int, default=600)
    ap.add_argument("--holdouts", nargs="+", default=ARCHS)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    if a.mode == "analyze": return analyze()
    lazy(); norm = get_norm(); steps = max(100, int(round(a.steps * a.data_frac)))
    if a.mode == "train":
        f = os.path.join(OUT, "frac%g" % a.data_frac); os.makedirs(f, exist_ok=True)
        for seed in range(a.n_seeds):
            for ref in ("b1", "noop"):
                p = os.path.join(f, "%s_s%d.json" % (ref, seed))
                if not os.path.exists(p): json.dump({x: ev(None, x, seed, ref) for x in ARCHS}, open(p, "w"))
            for c in ("indep", "fedavg", "fedprox"):
                p = os.path.join(f, "%s_s%d.json" % (c, seed))
                if os.path.exists(p): continue
                print("seed", seed, c, flush=True)
                if c == "indep": res = indep(seed, a.rounds, steps, norm)
                else:
                    ags, _ = fed(seed, a.rounds, steps, norm, c == "fedprox"); res = {x: ev(g, x, seed) for x, g in zip(ARCHS, ags)}
                json.dump(res, open(p, "w"))
    else:
        f = os.path.join(OUT, "cold"); os.makedirs(f, exist_ok=True)
        for seed in range(a.n_seeds):
            for h in a.holdouts:
                p = os.path.join(f, "%s_s%d.json" % (h, seed))
                if os.path.exists(p): continue
                print("seed", seed, "cold", h, flush=True)
                json.dump(cold(seed, h, a.rounds, a.steps, a.cold_steps, norm), open(p, "w"))
main()
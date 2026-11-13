"""
exp_dp.py
Corrected differential-privacy experiment (closes Limitation L6 and L10-ii).

Builds on exp_multi.py (federated actor AND critic, parameter-space proximal term)
and adds client-level DP done properly:
  * each round, every client computes its update  delta_i = w_i - w_global
  * delta_i is clipped to L2 norm C
  * Gaussian noise is added with a SEEDED generator
  * privacy loss is composed over all R rounds with RDP accounting
    (Gaussian mechanism, full participation, Mironov 2017 conversion)

Privacy unit: one building's entire dataset (client-level DP).
Two threat models (--dp_mode):
  client : every client noises its own clipped update before sending it.
           Protects against an untrusted server. Adjacency = replacing one
           client's data, so the L2 sensitivity of a release is 2C.
  server : clients send clipped updates; a trusted aggregator adds one noise
           draw to the SUM. Protects the released global model against
           everyone except the aggregator. Add/remove adjacency, sensitivity C.

Conditions written per seed (results/fed_dp/frac<f>/<cond>_s<seed>.json):
  fedclip          clipping only, no noise  (separates clipping from noise)
  dp_eps<E>        clipping + noise calibrated to a composed (E, delta) over R rounds
The independent baseline and un-clipped federation (indep, fed_arch) are read
from results/fed_multi/frac<f>/ produced by exp_multi.py with the same seeds.

Usage (same budget as the thesis multi-building run):
  python exp_dp.py --mode train --k 3 --n_seeds 8 --data_frac 0.2 --eps 16 8 4 1
  python exp_dp.py --mode analyze --data_frac 0.2
  python exp_dp.py --mode sigma --rounds 10 --eps 16 8 4 1     # print calibrated noise only
"""
import argparse, glob, json, math, os
import numpy as np
import exp_multi as M

OUT = "results/fed_dp"
ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]


def residential_clients(n=16):
    """The real residential clients of Appendix C.15 (data/residential_scale_clients)."""
    import re as _re
    fs = glob.glob(os.path.join("data", "residential_scale_clients", "residential_client*_15min.csv"))
    fs = sorted(fs, key=lambda x: int(_re.search(r"client(\d+)_", x).group(1)))[:n]
    if len(fs) < n: raise SystemExit("Found %d residential client files, need %d (run 10_residential_scale_federation.py first)." % (len(fs), n))
    return [{"arch": "residential", "id": "client%d" % i, "path": f_} for i, f_ in enumerate(fs)]


# ---------------------------------------------------------------- accounting
def rdp_to_eps(z, rounds, delta, sens):
    """Composed epsilon for R full-participation Gaussian releases.
    z = noise std / C ; sens = L2 sensitivity / C (2 for client mode, 1 for server mode).
    RDP of one release at order a: a * sens^2 / (2 z^2)."""
    if z <= 0: return math.inf
    return min(rounds * a * sens ** 2 / (2 * z ** 2) + math.log(1 / delta) / (a - 1) for a in ALPHAS)


def calibrate_z(eps, rounds, delta, sens):
    """Smallest noise multiplier z giving composed epsilon <= eps (bisection)."""
    if math.isinf(eps): return 0.0
    lo, hi = 1e-3, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if rdp_to_eps(mid, rounds, delta, sens) > eps: lo = mid
        else: hi = mid
    return hi


# ---------------------------------------------------------------- training
def shared_mask(ag, share):
    """Boolean mask over the flat parameter vector (actor then critic).
    share='all'  : every parameter is federated.
    share='head' : only the final linear layer (weight + bias) of the actor and of the critic."""
    T = M.T
    ap = list(ag.actor.parameters()); cp = list(ag.critic.parameters()); flags = []
    for ps in (ap, cp):
        for j, p_ in enumerate(ps):
            on = share == "all" or j >= len(ps) - 2
            flags.append(T.full((p_.numel(),), on, dtype=T.bool))
    return T.cat(flags)


def step_train_masked(ag, b, ref, mask):
    """exp_multi.step_train with the proximal penalty restricted to the shared coordinates."""
    import torch.nn.functional as F
    T = M.T
    s_, a_, r_, s2, d_ = b
    with T.no_grad(): y = r_ + ag.gamma * (1 - d_) * ag.critic_target(s2, ag.actor_target(s2))
    ap = list(ag.actor.parameters()); cp = list(ag.critic.parameters()); na = sum(p_.numel() for p_ in ap)
    def pen(ps, off):
        i = 0; tot = 0.0
        for p_ in ps:
            n_ = p_.numel(); m_ = mask[off + i:off + i + n_].view_as(p_)
            if m_.any(): tot = tot + (((p_ - ref[off + i:off + i + n_].view_as(p_)) ** 2) * m_).sum()
            i += n_
        return tot
    loss = F.mse_loss(ag.critic(s_, a_), y) + 0.5 * M.MU_PROX * pen(cp, na)
    ag.critic_opt.zero_grad(); loss.backward(); ag.critic_opt.step()
    al = -ag.critic(s_, ag.actor(s_)).mean() + 0.5 * M.MU_PROX * pen(ap, 0)
    ag.actor_opt.zero_grad(); al.backward(); ag.actor_opt.step()
    ag.soft_update(ag.critic_target, ag.critic); ag.soft_update(ag.actor_target, ag.actor)


def rnd_masked(ag, env, buf, steps, warm, ref, mask):
    """exp_multi.rnd using step_train_masked."""
    s_ = env.reset()
    for i in range(steps):
        a_ = np.random.uniform(-1, 1, env.action_dim) if i < warm else ag.act(s_)
        s2, r_, d_, _ = env.step(a_); buf.push(s_, a_, r_, s2, d_); s_ = s2
        if len(buf) > M.BATCH and i >= warm: step_train_masked(ag, buf.sample(M.BATCH), ref, mask)
        if d_: s_ = env.reset()


def train_group_dp(cl, seed, R, steps, norm, clip, z, mode, share="all", finetune=0):
    """Federation of the clients in cl with clipped (and optionally noised) updates of the shared part.
    Non-shared parameters never leave the client. After the last round each client may fine-tune
    locally for `finetune` steps (post-processing of its own data, no further release)."""
    from ddpg_agent import ReplayBuffer
    T = M.T
    T.manual_seed(seed); np.random.seed(seed)
    gen = T.Generator().manual_seed(10_000 + seed)            # seeded noise (fixes L10-ii)
    envs = [M.make_env(c["path"], seed * 11 + 3 + k, norm) for k, c in enumerate(cl)]
    ags = [M.new_agent(e, R * steps + finetune) for e in envs]; bufs = [ReplayBuffer() for _ in cl]
    mask = shared_mask(ags[0], share)
    g = M.flat(ags[0]).clone()
    for ag in ags:                                             # common start for the shared part only
        v = M.flat(ag); v[mask] = g[mask]; M.setflat(ag, v)
    n = len(cl); clipped_frac = []
    for r in range(R):
        deltas = []
        for ag, env, buf in zip(ags, envs, bufs):
            warm = min(M.WARMUP, steps // 4) if r == 0 else 0
            if share == "all": M.rnd(ag, env, buf, steps, warm, g)   # identical to exp_multi
            else: rnd_masked(ag, env, buf, steps, warm, g, mask)     # proximal pull on shared coords only
            d = (M.flat(ag) - g)[mask]
            nrm = float(d.norm())
            clipped_frac.append(nrm > clip)
            d = d * min(1.0, clip / (nrm + 1e-12))
            if mode == "client" and z > 0:
                d = d + T.normal(0.0, z * clip, size=d.shape, generator=gen)
            deltas.append(d)
        s = T.stack(deltas).sum(0)
        if mode == "server" and z > 0:
            s = s + T.normal(0.0, z * clip, size=s.shape, generator=gen)
        g[mask] = g[mask] + s / n
        for ag in ags:
            v = M.flat(ag); v[mask] = g[mask]; M.setflat(ag, v)
    if finetune > 0:
        for ag, env, buf in zip(ags, envs, bufs):
            M.rnd(ag, env, buf, finetune, 0, None)
    return ags, float(np.mean(clipped_frac))


def run(cond, cl, seed, R, steps, norm, clip, z, mode, group, share="all", finetune=0):
    archs = sorted(set(c["arch"] for c in cl)); res = {}; meta = {}
    groups = [(a, [c for c in cl if c["arch"] == a]) for a in archs] if group == "arch" else [("all", cl)]
    for a, gcl in groups:
        ags, cf = train_group_dp(gcl, seed, R, steps, norm, clip, z, mode, share, finetune)
        for c in gcl: meta[c["arch"]] = {"share_of_updates_clipped": cf}
        for c, ag in zip(gcl, ags):
            cost, disc = M.ev(ag, c["path"], seed); res[c["arch"] + "|" + c["id"]] = {"cost": cost, "disc": disc}
    res["_meta"] = {"cond": cond, "z": z, "clip": clip, "mode": mode, "rounds": R, "group": group, "share": share, "finetune": finetune, **meta}
    return res


# ---------------------------------------------------------------- analysis
def out_dir(frac, group, mode, share="all", finetune=0, clients="multi"):
    tag = ("" if clients == "multi" else "_" + clients) + \
          ("" if group == "arch" else "_all") + ("" if mode == "client" else "_server") + \
          ("" if share == "all" else "_head") + ("" if finetune == 0 else "_ft%d" % finetune)
    return os.path.join(OUT + tag, "frac%g" % frac)


def analyze(frac, group="arch", mode="client", share="all", finetune=0, clients="multi"):
    dp_dir = out_dir(frac, group, mode, share, finetune, clients)
    ref_dir = os.path.join(M.OUT, "frac%g" % frac) if clients == "multi" else dp_dir   # res16 stores its own baselines
    ref = M.load(ref_dir); dp = M.load(dp_dir)
    dp = {c: v for c, v in dp.items() if c == "fedclip" or c.startswith("dp_eps")}
    if "indep" not in ref: print("No indep results in", ref_dir, "- run exp_multi.py first."); return
    conds = sorted(dp, key=lambda c: (c != "fedclip", -float(c[6:]) if c.startswith("dp_eps") else 0))
    if "fedclip" in dp: ref = dict(ref); ref["fedclip (same budget, no noise)"] = dp["fedclip"]
    for base in ("indep", "fed_arch" if group == "arch" else "fed_all", "b1", "noop", "fedclip (same budget, no noise)"):
        if base not in ref: continue
        print("\n== cost relative to", base, "(negative = cheaper) |", dp_dir)
        for c in conds:
            if base.startswith("fedclip") and c == "fedclip": continue
            ss = sorted(set(dp[c]) & set(ref[base]))
            if not ss: continue
            keys = [k for k in dp[c][ss[0]] if not k.startswith("_")]
            archs = sorted(set(k.split("|")[0] for k in keys))
            meta = dp[c][ss[0]]["_meta"]
            print("%-10s z=%-7.3f mode=%s" % (c, meta["z"], meta["mode"]))
            for a in archs + ["ALL"]:
                ks = [k for k in keys if a == "ALL" or k.startswith(a + "|")]
                x = [np.mean([dp[c][s][k]["cost"] for k in ks]) for s in ss]
                y = [np.mean([ref[base][s][k]["cost"] for k in ks]) for s in ss]
                m, p = M.pw(x, y)
                dx = np.array([np.mean([dp[c][s][k]["disc"] for k in ks]) for s in ss]); dy = np.array([np.mean([ref[base][s][k]["disc"] for k in ks]) for s in ss])
                dd = float(np.mean((dx - dy) / np.where(dy == 0, np.nan, dy)) * 100)
                pd_ = M.pw(dx, dy)[1] if np.all(dy > 0) else float("nan")
                cf = np.mean([dp[c][s]["_meta"][a]["share_of_updates_clipped"] for s in ss]) if a != "ALL" else float("nan")
                print("   %-12s n=%2d seeds  cost %%: %+7.2f  p = %.4f  discomfort %%: %+7.1f (p = %.4f)  clipped: %s" % (a, len(ss), m, p, dd, pd_, "" if a == "ALL" else "%.0f%%" % (100 * cf)))
    n_tests = 4 * max(1, len([c for c in conds if c.startswith("dp_eps")]))
    print("\nBonferroni: with %d archetype-by-epsilon tests per baseline, use p < %.4f" % (n_tests, 0.05 / n_tests))
    print("Primary test only (eps=8 vs indep, 4 archetypes): use p < 0.0125")
    print("Note: with n paired seeds the smallest possible two-sided Wilcoxon p is 2/2^n (n=8: 0.0078, n=12: 0.0005).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "analyze", "sigma"])
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--n_seeds", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--data_frac", type=float, default=0.2)
    ap.add_argument("--eps", type=float, nargs="+", default=[64, 16, 8, 1])
    ap.add_argument("--delta", type=float, default=1e-5); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dp_mode", default="client", choices=["client", "server"])
    ap.add_argument("--no_fedclip", action="store_true")
    ap.add_argument("--clients", default="multi", choices=["multi", "res16"], help="12-building corrective set, or the 16 real residential clients of C.15")
    ap.add_argument("--share", default="all", choices=["all", "head"], help="federate the whole network or only the final layer of actor and critic")
    ap.add_argument("--finetune", type=int, default=0, help="local fine-tuning steps after the last round (no release)")
    ap.add_argument("--group", default="arch", choices=["arch", "all"], help="federate within each archetype (3 clients) or across all 12 buildings")
    a = ap.parse_args(); sens = 2.0 if a.dp_mode == "client" else 1.0
    if a.mode == "sigma":
        for e in a.eps:
            z = calibrate_z(e, a.rounds, a.delta, sens)
            print("eps=%g delta=%g R=%d mode=%s -> noise multiplier z=%.3f (check: eps=%.3f)" % (e, a.delta, a.rounds, a.dp_mode, z, rdp_to_eps(z, a.rounds, a.delta, sens)))
        return
    if a.mode == "analyze": return analyze(a.data_frac, a.group, a.dp_mode, a.share, a.finetune, a.clients)
    M.lazy(); cl = M.clients(a.k) if a.clients == "multi" else residential_clients(16); norm = M.get_norm(cl); steps = max(100, int(round(a.steps * a.data_frac)))
    f = out_dir(a.data_frac, a.group, a.dp_mode, a.share, a.finetune, a.clients); os.makedirs(f, exist_ok=True)
    plan = ([] if a.no_fedclip else [("fedclip", 0.0)]) + [("dp_eps%g" % e, calibrate_z(e, a.rounds, a.delta, sens)) for e in a.eps]
    d_shared = int(shared_mask(M.new_agent(M.make_env(cl[0]["path"], 0, norm), 1), a.share).sum())
    n_cl = len(cl) if a.group == "all" else len([c for c in cl if c["arch"] == cl[0]["arch"]])
    print("shared parameters per client: %d | clients per federation: %d" % (d_shared, n_cl))
    for name, z in plan:
        nr = z * math.sqrt(d_shared) / (math.sqrt(n_cl) if a.dp_mode == "client" else n_cl)
        print("%-10s z = %.3f  | expected noise norm on the averaged update / clipped signal C: %.1f" % (name, z, nr), flush=True)
    for seed in range(a.n_seeds):
        if a.clients != "multi":                                  # own baselines for the new client set
            for ref_ in ("b1", "noop"):
                p = os.path.join(f, "%s_s%d.json" % (ref_, seed))
                if not os.path.exists(p):
                    json.dump({c["arch"] + "|" + c["id"]: dict(zip(("cost", "disc"), M.ev(None, c["path"], seed, ref_))) for c in cl}, open(p, "w"))
            p = os.path.join(f, "indep_s%d.json" % seed)
            if not os.path.exists(p):
                print("seed", seed, "indep", flush=True)
                json.dump(M.run_condition("indep", cl, seed, a.rounds, steps, norm), open(p, "w"))
        for name, z in plan:
            p = os.path.join(f, "%s_s%d.json" % (name, seed))
            if os.path.exists(p): continue
            print("seed", seed, name, flush=True)
            json.dump(run(name, cl, seed, a.rounds, steps, norm, a.clip, z, a.dp_mode, a.group, a.share, a.finetune), open(p, "w"))


if __name__ == "__main__":
    main()

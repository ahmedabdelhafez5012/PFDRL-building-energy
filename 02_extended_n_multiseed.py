"""
02_extended_n_multiseed.py  (v2 - REAL implementation, built directly
against multi_seed_h1_test.py, multi_seed_h2_test.py, multi_seed_h4_test.py)
==============================================================================
Extends H1, H2, H4 from N=8 to N_TARGET (default 30) seeds, at the SAME
reduced per-seed budget already used and reported (15 rounds x 3,000 steps),
so results stay directly comparable to the existing Appendix C.9/C.10/C.12
numbers -- just with a larger, more defensible N.

Efficiency note (same lesson as 01_full_ablation_b1_b4.py): H4's federation
trains all four archetypes TOGETHER in one run_one_seed_h4 call. This script
therefore runs H4 once per seed (covering all four archetypes), not once per
archetype -- avoiding the ~4x redundant-retraining trap.
H1 and H2 are inherently per-archetype (no federation involved), so they are
looped over archetypes within each seed with no redundancy concern.

Resume support: safe to Ctrl+C and re-run the same command; each hypothesis
tracks its own progress in its own JSON file under --out_dir.

Run:
    python 02_extended_n_multiseed.py --hypothesis all --n_target 30
    python 02_extended_n_multiseed.py --hypothesis h4 --n_target 30   # just one
    python 02_extended_n_multiseed.py --hypothesis all --n_target 30 --skip_existing_n8
        (use this flag if you want to treat your EXISTING N=8 console results as
         already "seed 0-7" and only compute NEW seeds 8..N_TARGET-1 -- see notes below)
"""
import argparse
import json
import os
import copy
import numpy as np
import torch
from scipy.stats import kruskal, wilcoxon

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import fedprox_aggregate, fedavg_aggregate, add_gaussian_dp_noise
from baseline_b1_rule_based import rule_based_action

ROUNDS_PER_SEED = 15
STEPS_PER_ROUND = 3000
WARMUP_STEPS = 800
FINE_TUNE_STEPS = 2000
BATCH_SIZE = 128
MU_FEDPROX = 0.05
DP_EPSILON = 8.0
DP_DELTA = 1e-5
EPISODE_LEN = 96
EVAL_EPISODES = 10

MU_HAT_STAR = {"residential": 0.80, "office": 0.00, "commercial": 0.01, "mixed-use": 0.01}

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv"},
    {"name": "office", "path": "data/office_archetype_15min.csv"},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv"},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv"},
]
ARCH_NAMES = [a["name"] for a in ARCHETYPES]


def make_env(path, seed):
    return BuildingEnergyEnv(data_path=path, episode_len=EPISODE_LEN, seed=seed)


def local_train_round(agent, env, buf, steps, warmup=0, mu=1.0, personalisation_target=None):
    s = env.reset()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE), personalisation_target=personalisation_target, mu=mu)
        if done:
            s = env.reset()


def evaluate(agent, path, seed):
    env = make_env(path, seed=seed)
    costs = []
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        ep_cost = 0.0
        done = False
        while not done:
            a = agent.act(s, explore=False)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
        costs.append(ep_cost)
    return float(np.mean(costs))


def evaluate_b1(path, seed):
    env = make_env(path, seed=seed)
    costs = []
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        ep_cost = 0.0
        done = False
        while not done:
            a = rule_based_action(s, env.price_scale)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
        costs.append(ep_cost)
    return float(np.mean(costs))


# ---------------------------------------------------------------------------
# H1 - per-archetype independent training, ratio vs B1
# ---------------------------------------------------------------------------
def run_h1_one_seed(seed):
    torch.manual_seed(200 + seed)
    np.random.seed(200 + seed)
    out = {}
    for a in ARCHETYPES:
        env = make_env(a["path"], seed=seed)
        agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                           noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
        buf = ReplayBuffer()
        for rnd in range(ROUNDS_PER_SEED):
            local_train_round(agent, env, buf, STEPS_PER_ROUND, warmup=WARMUP_STEPS if rnd == 0 else 0)
        ddpg_cost = evaluate(agent, a["path"], seed=9000 + seed)
        b1_cost = evaluate_b1(a["path"], seed=9000 + seed)
        out[a["name"]] = ddpg_cost / b1_cost if b1_cost != 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# H2 - per-archetype Stage1 + personalised/non-personalised fine-tune
# ---------------------------------------------------------------------------
def train_stage1(archetype, seed):
    env = make_env(archetype["path"], seed=seed)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                       noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
    buf = ReplayBuffer()
    for rnd in range(ROUNDS_PER_SEED):
        local_train_round(agent, env, buf, STEPS_PER_ROUND, warmup=WARMUP_STEPS if rnd == 0 else 0)
    return agent


def fine_tune(archetype, base_agent, mu, seed):
    env = make_env(archetype["path"], seed=seed + 1000)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.1, noise_decay_steps=FINE_TUNE_STEPS)
    agent.actor.load_state_dict(base_agent.actor.state_dict())
    agent.critic.load_state_dict(base_agent.critic.state_dict())
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    global_critic_state = copy.deepcopy(base_agent.critic.state_dict())

    buf = ReplayBuffer()
    s = env.reset()
    for i in range(FINE_TUNE_STEPS):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > 128:
            agent.train_step(buf.sample(128), personalisation_target=global_critic_state, mu=mu)
        if done:
            s = env.reset()
    return agent


def run_h2_one_seed(seed):
    torch.manual_seed(100 + seed)
    np.random.seed(100 + seed)
    pers, nonpers = {}, {}
    for a in ARCHETYPES:
        base_agent = train_stage1(a, seed)
        mu_star = MU_HAT_STAR[a["name"]]
        p_agent = fine_tune(a, base_agent, mu_star, seed)
        np_agent = fine_tune(a, base_agent, 1.0, seed)
        pers[a["name"]] = evaluate(p_agent, a["path"], seed=9000 + seed)
        nonpers[a["name"]] = evaluate(np_agent, a["path"], seed=9000 + seed)
    return pers, nonpers


# ---------------------------------------------------------------------------
# H4 - federated (all 4 together), FedProx vs FedAvg. Trained ONCE per seed
# per strategy, covering all archetypes (avoids the 4x redundancy trap).
# ---------------------------------------------------------------------------
def run_h4_one_seed_one_strategy(strategy, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = [make_env(a["path"], seed=i * 11 + seed) for i, a in enumerate(ARCHETYPES)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim, noise_sigma=0.4,
                         noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
              for _ in ARCHETYPES]
    bufs = [ReplayBuffer() for _ in ARCHETYPES]
    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)

    for rnd in range(1, ROUNDS_PER_SEED + 1):
        client_critics = []
        for i in range(len(ARCHETYPES)):
            warmup = WARMUP_STEPS if rnd == 1 else 0
            local_train_round(agents[i], envs[i], bufs[i], STEPS_PER_ROUND, warmup=warmup)
            sd, _ = add_gaussian_dp_noise(agents[i].critic.state_dict(), epsilon=DP_EPSILON, delta=DP_DELTA)
            client_critics.append(sd)
        if strategy == "fedprox":
            new_global = fedprox_aggregate(client_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        else:
            new_global = fedavg_aggregate(client_critics)
        global_critic.load_state_dict(new_global)
        for i in range(len(ARCHETYPES)):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

    return {a["name"]: evaluate(agents[i], a["path"], seed=9000 + seed) for i, a in enumerate(ARCHETYPES)}


def run_h4_one_seed(seed):
    fp = run_h4_one_seed_one_strategy("fedprox", seed)
    fa = run_h4_one_seed_one_strategy("fedavg", seed)
    return fp, fa


# ---------------------------------------------------------------------------
# Driver: resume-capable, per-hypothesis JSON
# ---------------------------------------------------------------------------
def load_or_init(path, keys):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {k: [] for k in keys}


def save(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_h1(n_target, out_dir):
    path = os.path.join(out_dir, "h1_results.json")
    results = load_or_init(path, ARCH_NAMES)
    n_done = min(len(results[a]) for a in ARCH_NAMES)
    for seed in range(n_done, n_target):
        print(f"[H1] seed {seed+1}/{n_target}")
        vals = run_h1_one_seed(seed)
        for a in ARCH_NAMES:
            results[a].append(vals[a])
        save(path, results)

    print("\n[H1] Kruskal-Wallis across archetype ratio distributions:")
    groups = [results[a] for a in ARCH_NAMES]
    stat, p = kruskal(*groups)
    print(f"  H={stat:.4f}  p={p:.4f}  ({'archetypes DIFFER significantly' if p < 0.05 else 'consistent with H1 (parity)'})")
    for a in ARCH_NAMES:
        arr = np.array(results[a])
        print(f"  {a:12s} mean_ratio={arr.mean():.4f} (+/-{arr.std():.4f})  n={len(arr)}")


def run_h2(n_target, out_dir):
    path = os.path.join(out_dir, "h2_results.json")
    results = load_or_init(path, [f"{a}_pers" for a in ARCH_NAMES] + [f"{a}_nonpers" for a in ARCH_NAMES])
    n_done = min(len(results[f"{a}_pers"]) for a in ARCH_NAMES)
    for seed in range(n_done, n_target):
        print(f"[H2] seed {seed+1}/{n_target}")
        pers, nonpers = run_h2_one_seed(seed)
        for a in ARCH_NAMES:
            results[f"{a}_pers"].append(pers[a])
            results[f"{a}_nonpers"].append(nonpers[a])
        save(path, results)

    print("\n[H2] Wilcoxon signed-rank, personalised vs non-personalised:")
    bonf_alpha = 0.05 / len(ARCH_NAMES)
    for a in ARCH_NAMES:
        p_vals = np.array(results[f"{a}_pers"])
        np_vals = np.array(results[f"{a}_nonpers"])
        try:
            stat, p = wilcoxon(p_vals, np_vals)
        except ValueError as e:
            print(f"  {a:12s} Wilcoxon failed: {e}")
            continue
        survives = "YES" if p < bonf_alpha else "no"
        print(f"  {a:12s} pers={p_vals.mean():.3f}  nonpers={np_vals.mean():.3f}  "
              f"p={p:.4f}  Bonferroni(a={bonf_alpha:.4f})survives={survives}  n={len(p_vals)}")


def run_h4(n_target, out_dir):
    path = os.path.join(out_dir, "h4_results.json")
    results = load_or_init(path, [f"{a}_fedprox" for a in ARCH_NAMES] + [f"{a}_fedavg" for a in ARCH_NAMES])
    n_done = min(len(results[f"{a}_fedprox"]) for a in ARCH_NAMES)
    for seed in range(n_done, n_target):
        print(f"[H4] seed {seed+1}/{n_target}")
        fp, fa = run_h4_one_seed(seed)
        for a in ARCH_NAMES:
            results[f"{a}_fedprox"].append(fp[a])
            results[f"{a}_fedavg"].append(fa[a])
        save(path, results)

    print("\n[H4] Wilcoxon signed-rank, FedProx vs FedAvg:")
    bonf_alpha = 0.05 / len(ARCH_NAMES)
    for a in ARCH_NAMES:
        fp_vals = np.array(results[f"{a}_fedprox"])
        fa_vals = np.array(results[f"{a}_fedavg"])
        try:
            stat, p = wilcoxon(fp_vals, fa_vals)
        except ValueError as e:
            print(f"  {a:12s} Wilcoxon failed: {e}")
            continue
        survives = "YES" if p < bonf_alpha else "no"
        direction = "FedProx lower" if fp_vals.mean() < fa_vals.mean() else "FedAvg lower"
        print(f"  {a:12s} FedProx={fp_vals.mean():.3f}  FedAvg={fa_vals.mean():.3f}  "
              f"p={p:.4f}  Bonferroni survives={survives}  {direction}  n={len(fp_vals)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", choices=["h1", "h2", "h4", "all"], default="all")
    parser.add_argument("--n_target", type=int, default=30)
    parser.add_argument("--out_dir", default="results/extended_n")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    todo = ["h1", "h2", "h4"] if args.hypothesis == "all" else [args.hypothesis]
    if "h1" in todo:
        print("\n" + "=" * 70 + "\nH1\n" + "=" * 70)
        run_h1(args.n_target, args.out_dir)
    if "h2" in todo:
        print("\n" + "=" * 70 + "\nH2\n" + "=" * 70)
        run_h2(args.n_target, args.out_dir)
    if "h4" in todo:
        print("\n" + "=" * 70 + "\nH4\n" + "=" * 70)
        run_h4(args.n_target, args.out_dir)


if __name__ == "__main__":
    main()

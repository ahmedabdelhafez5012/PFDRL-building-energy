"""
01_full_ablation_b1_b4.py  (v3 - efficiency-fixed)
==============================================================================
CRITICAL FIX vs v2: B3 (federation) trains all four archetypes TOGETHER by
construction (that's what federation means) -- so calling it once per
--archetype flag, four separate times, retrains the SAME 4-archetype
federation four times over. This version runs ONCE per seed and evaluates
all four archetypes from that single federation/pooled-oracle training run,
cutting total wall-clock time by roughly 4x for B3 (and B4, which similarly
benefits from being trained once and evaluated on all four archetypes rather
than retrained per archetype).

Measured timing on this codebase (CPU, rounds=15, steps=3000): ~26 minutes
for B1+B2+B3 for a SINGLE archetype call under the old per-archetype design.
With this fix, B3's federation training cost is paid ONCE per seed (not once
per archetype), so the total campaign time drops from an estimated ~44 hours
to roughly ~16-19 hours for N=20 seeds across all four archetypes and all
four baselines.

Run (covers ALL FOUR archetypes per seed -- no --archetype flag needed):
    python 01_full_ablation_b1_b4.py --n_seeds 20
    python 01_full_ablation_b1_b4.py --n_seeds 20 --rounds 50 --steps 5000   # full budget
    python 01_full_ablation_b1_b4.py --n_seeds 20 --skip_b4                 # faster first pass

Resume support: safe to Ctrl+C and re-run the same command; already-completed
seeds are skipped automatically (checked per-archetype, per-baseline).
"""
import argparse
import json
import os
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import fedavg_aggregate, add_gaussian_dp_noise
from baseline_b1_rule_based import rule_based_action

ARCHETYPE_PATHS = {
    "residential": "data/residential_archetype_15min.csv",
    "office":      "data/office_archetype_15min.csv",
    "commercial":  "data/commercial_archetype_15min.csv",
    "mixed_use":   "data/mixeduse_archetype_15min.csv",
}
ALL_ARCHETYPES = list(ARCHETYPE_PATHS.keys())

DEFAULT_ROUNDS = 15
DEFAULT_STEPS_PER_ROUND = 3000
BATCH_SIZE = 128
DP_EPSILON = 8.0
DP_DELTA = 1e-5
EPISODE_LEN = 96
EVAL_EPISODES = 15
EVAL_SEED_OFFSET = 9000


def make_env(archetype, seed):
    return BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype],
                              episode_len=EPISODE_LEN, seed=seed)


def evaluate_rule_based(archetype, seed):
    env = make_env(archetype, seed=EVAL_SEED_OFFSET + seed)
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


def evaluate_agent(agent, archetype, seed):
    env = make_env(archetype, seed=EVAL_SEED_OFFSET + seed)
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


def local_train_round(agent, env, buf, steps, warmup=0):
    s = env.reset()
    for i in range(steps):
        if i < warmup:
            a = np.random.uniform(-1, 1, size=env.action_dim)
        else:
            a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE), mu=1.0)
        if done:
            s = env.reset()


# ---------------------------------------------------------------------------
# B1 - Rule-based (per archetype, no training, cheap -- fine to call 4x)
# ---------------------------------------------------------------------------
def run_b1_all_archetypes(seed):
    return {a: evaluate_rule_based(a, seed) for a in ALL_ARCHETYPES}


# ---------------------------------------------------------------------------
# B2 - Independent DDPG (inherently per-archetype, no redundancy possible --
# each archetype genuinely needs its own solo-trained agent)
# ---------------------------------------------------------------------------
def run_b2_all_archetypes(seed, rounds, steps_per_round):
    out = {}
    for a in ALL_ARCHETYPES:
        torch.manual_seed(seed); np.random.seed(seed)
        env = make_env(a, seed=seed * 11 + 3)
        agent = DDPGAgent(env.state_dim, env.action_dim,
                           noise_sigma=0.4, noise_decay_steps=int(rounds * steps_per_round * 0.6))
        buf = ReplayBuffer()
        for rnd in range(1, rounds + 1):
            warmup = 800 if rnd == 1 else 0
            local_train_round(agent, env, buf, steps_per_round, warmup=warmup)
        out[a] = evaluate_agent(agent, a, seed)
    return out


# ---------------------------------------------------------------------------
# B3 - Federated, no personalisation. TRAINED ONCE (all 4 archetypes
# together, as federation requires), evaluated on all 4.
# ---------------------------------------------------------------------------
def run_b3_all_archetypes(seed, rounds, steps_per_round):
    torch.manual_seed(seed); np.random.seed(seed)

    envs = {a: make_env(a, seed=seed * 11 + 3 + i) for i, a in enumerate(ALL_ARCHETYPES)}
    agents = {a: DDPGAgent(envs[a].state_dim, envs[a].action_dim,
                            noise_sigma=0.4, noise_decay_steps=int(rounds * steps_per_round * 0.6))
              for a in ALL_ARCHETYPES}
    bufs = {a: ReplayBuffer() for a in ALL_ARCHETYPES}
    global_critic = Critic(envs[ALL_ARCHETYPES[0]].state_dim, envs[ALL_ARCHETYPES[0]].action_dim)

    for rnd in range(1, rounds + 1):
        warmup = 800 if rnd == 1 else 0
        noised_critics = []
        for a in ALL_ARCHETYPES:
            local_train_round(agents[a], envs[a], bufs[a], steps_per_round, warmup=warmup)
            noised_sd, _ = add_gaussian_dp_noise(agents[a].critic.state_dict(),
                                                  epsilon=DP_EPSILON, delta=DP_DELTA)
            noised_critics.append(noised_sd)
        new_global = fedavg_aggregate(noised_critics)
        global_critic.load_state_dict(new_global)
        for a in ALL_ARCHETYPES:
            agents[a].critic.load_state_dict(new_global)
            agents[a].critic_target.load_state_dict(new_global)

    return {a: evaluate_agent(agents[a], a, seed) for a in ALL_ARCHETYPES}


# ---------------------------------------------------------------------------
# B4 - Centralised oracle. TRAINED ONCE on pooled data from all 4
# archetypes, evaluated on all 4 (the whole point of "centralised": one
# agent, all data, then check how it does per archetype).
# ---------------------------------------------------------------------------
def run_b4_all_archetypes(seed, rounds, steps_per_round):
    torch.manual_seed(seed); np.random.seed(seed)

    envs = {a: make_env(a, seed=seed * 11 + 3 + i) for i, a in enumerate(ALL_ARCHETYPES)}
    agent = DDPGAgent(envs[ALL_ARCHETYPES[0]].state_dim, envs[ALL_ARCHETYPES[0]].action_dim,
                       noise_sigma=0.4, noise_decay_steps=int(rounds * steps_per_round * 0.6))
    pooled_buf = ReplayBuffer(capacity=50000 * len(ALL_ARCHETYPES))

    states = {a: envs[a].reset() for a in ALL_ARCHETYPES}
    total_steps = rounds * steps_per_round
    warmup = 800

    for i in range(total_steps):
        a_name = ALL_ARCHETYPES[i % len(ALL_ARCHETYPES)]
        env = envs[a_name]
        s = states[a_name]
        if i < warmup:
            a = np.random.uniform(-1, 1, size=env.action_dim)
        else:
            a = agent.act(s)
        s2, r, done, info = env.step(a)
        pooled_buf.push(s, a, r, s2, done)
        states[a_name] = env.reset() if done else s2
        if len(pooled_buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(pooled_buf.sample(BATCH_SIZE))

    return {a: evaluate_agent(agent, a, seed) for a in ALL_ARCHETYPES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_ROUND)
    parser.add_argument("--out_dir", default="results/ablation_b1_b4")
    parser.add_argument("--skip_b4", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_paths = {a: os.path.join(args.out_dir, f"{a}_ablation.json") for a in ALL_ARCHETYPES}

    results = {}
    for a in ALL_ARCHETYPES:
        if os.path.exists(out_paths[a]):
            with open(out_paths[a]) as f:
                results[a] = json.load(f)
        else:
            results[a] = {"B1": [], "B2": [], "B3": [], "B4": []}

    def n_done(baseline):
        return min(len(results[a][baseline]) for a in ALL_ARCHETYPES)

    def save_all():
        for a in ALL_ARCHETYPES:
            with open(out_paths[a], "w") as f:
                json.dump(results[a], f, indent=2)

    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed+1}/{args.n_seeds} (rounds={args.rounds}, steps/round={args.steps}) ===")

        if seed >= n_done("B1"):
            vals = run_b1_all_archetypes(seed)
            for a in ALL_ARCHETYPES: results[a]["B1"].append(vals[a])
            print("  B1 (rule-based):        " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if seed >= n_done("B2"):
            vals = run_b2_all_archetypes(seed, args.rounds, args.steps)
            for a in ALL_ARCHETYPES: results[a]["B2"].append(vals[a])
            print("  B2 (independent DDPG):  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if seed >= n_done("B3"):
            vals = run_b3_all_archetypes(seed, args.rounds, args.steps)
            for a in ALL_ARCHETYPES: results[a]["B3"].append(vals[a])
            print("  B3 (FedAvg, no pers.):  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if not args.skip_b4 and seed >= n_done("B4"):
            vals = run_b4_all_archetypes(seed, args.rounds, args.steps)
            for a in ALL_ARCHETYPES: results[a]["B4"].append(vals[a])
            print("  B4 (centralised oracle):" + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

    print("\nDone. Per-archetype summaries:")
    for a in ALL_ARCHETYPES:
        print(f"\n[{a}]  -> {out_paths[a]}")
        for b, vals in results[a].items():
            if vals:
                arr = np.array(vals)
                print(f"  {b}: mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}")


if __name__ == "__main__":
    main()

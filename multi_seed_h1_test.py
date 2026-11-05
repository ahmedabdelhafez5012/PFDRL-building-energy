"""
multi_seed_h1_test.py
Multi-seed statistical protocol (Section 5.4) for Hypothesis H1: "PFDRL
achieves consistent (parity) relative improvement over the rule-based
baseline (B1) across the four heterogeneous building archetypes" - i.e.,
the framework does not work well on some archetypes and poorly on others.

Methodology: for each archetype, train DDPG across N_SEEDS independent
seeds, evaluate held-out cost, and compute the improvement ratio relative
to B1 (rule-based, deterministic, evaluated on the same held-out episodes):
    ratio = DDPG_cost / B1_cost   (ratio < 1 means DDPG beats B1)
"Parity" (H1) is then tested via a Kruskal-Wallis test across the four
archetypes' ratio distributions: if the four groups of ratios are NOT
significantly different from each other, this supports H1 (consistent
relative benefit); if they ARE significantly different, PFDRL's benefit
varies meaningfully by archetype, contradicting a strong reading of H1.
"""
import numpy as np
import torch
from scipy.stats import kruskal

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer
from baseline_b1_rule_based import rule_based_action

N_SEEDS = 8
ROUNDS_PER_SEED = 15
STEPS_PER_ROUND = 3000
WARMUP_STEPS = 800
BATCH_SIZE = 128
EPISODE_LEN = 96
EVAL_EPISODES = 10

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv"},
    {"name": "office", "path": "data/office_archetype_15min.csv"},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv"},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv"},
]


def make_env(path, seed):
    return BuildingEnergyEnv(data_path=path, episode_len=EPISODE_LEN, seed=seed)


def local_train_round(agent, env, buf, steps, warmup=0):
    s = env.reset()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE))
        if done:
            s = env.reset()


def evaluate_ddpg(agent, path, seed):
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


def run():
    print("=" * 78)
    print("Multi-seed H1 test: cross-archetype parity (DDPG/B1 ratio)")
    print(f"N_SEEDS={N_SEEDS}")
    print("=" * 78)

    ratios = {a["name"]: [] for a in ARCHETYPES}

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(200 + seed)
        np.random.seed(200 + seed)
        for archetype in ARCHETYPES:
            env = make_env(archetype["path"], seed=seed)
            agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                               noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
            buf = ReplayBuffer()
            for rnd in range(ROUNDS_PER_SEED):
                local_train_round(agent, env, buf, STEPS_PER_ROUND, warmup=WARMUP_STEPS if rnd == 0 else 0)

            ddpg_cost = evaluate_ddpg(agent, archetype["path"], seed=9000 + seed)
            b1_cost = evaluate_b1(archetype["path"], seed=9000 + seed)
            ratio = ddpg_cost / b1_cost if b1_cost != 0 else float("nan")
            ratios[archetype["name"]].append(ratio)
            print(f"  {archetype['name']:12s} | DDPG={ddpg_cost:10.3f} | B1={b1_cost:10.3f} | ratio={ratio:.4f}")

    print("\n" + "=" * 78)
    print("KRUSKAL-WALLIS TEST (H1: ratio distributions are similar across archetypes)")
    print("=" * 78)
    groups = [ratios[a["name"]] for a in ARCHETYPES]
    stat, p = kruskal(*groups)
    sig = "SIGNIFICANT DIFFERENCE (p<0.05) - archetypes differ, contradicts strong H1" if p < 0.05 else "not significant - consistent with H1 (parity)"
    print(f"H-statistic={stat:.4f} | p={p:.4f} | {sig}")

    print("\nPer-archetype ratio summary (ratio < 1.0 means DDPG beats B1):")
    for name in ratios:
        vals = np.array(ratios[name])
        print(f"  {name:12s} | mean={vals.mean():.4f} (+/-{vals.std():.4f}) | "
              f"beats B1 in {np.sum(vals < 1.0)}/{len(vals)} seeds")

    print("\nRaw per-seed ratios:")
    print(ratios)


if __name__ == "__main__":
    run()

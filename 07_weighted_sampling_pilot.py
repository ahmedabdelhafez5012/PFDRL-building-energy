"""
07_weighted_sampling_pilot.py
================================
PHASE 1c of the Nature-tier roadmap: tests whether upweighting
residential's representation in B4's training batches -- the practical
equivalent of "a higher effective learning rate for residential" in a
single shared network trained on pooled, interleaved data -- closes the
residual B4-B1 gap that survived both reward normalisation (Phase 1a,
gap shrank 64%) and increased network capacity (Phase 1b, NOT supported,
gap got slightly worse).

WHY NOT A LITERAL PER-ARCHETYPE LEARNING RATE: B4 is one network with one
optimiser; within a mixed training batch, all samples share the same
gradient step, so "residential's learning rate" cannot be set independently
of the other archetypes'. The closest faithful, implementable equivalent is
to change how OFTEN residential's samples appear in each batch (its
"voice" in the shared gradient), which is what this script tests.

DESIGN: replaces the single pooled ReplayBuffer (used in Phase 1a/1b) with
FOUR separate per-archetype buffers, filled via the same round-robin data
collection as before. Each training batch is then constructed with a
CONFIGURABLE per-archetype mixture ratio instead of being drawn uniformly
at random from one pooled buffer. Default (--weights equal) reproduces
Phase 1a's behaviour as a sanity check; --weights residential_boost tests
the upweighted hypothesis.

Run:
    python 07_weighted_sampling_pilot.py --n_seeds 10 --weights equal
        (sanity check: should reproduce Phase 1a's hidden=64 result)
    python 07_weighted_sampling_pilot.py --n_seeds 10 --weights residential_boost
        (test hypothesis: residential gets 40% of every batch instead of 25%)
"""
import argparse
import json
import os
import numpy as np
import torch
from scipy import stats

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer

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
EPISODE_LEN = 96
EVAL_EPISODES = 15
EVAL_SEED_OFFSET = 9000
NORM_STATS_PATH = "results/reward_norm_stats.json"
ORIGINAL_NORMALISED_DIR = "results/ablation_b1_b4_normalised"  # Phase 1a's output

WEIGHT_PRESETS = {
    # fraction of each BATCH_SIZE=128 batch drawn from each archetype's own buffer
    "equal":               {"residential": 0.25, "office": 0.25, "commercial": 0.25, "mixed_use": 0.25},
    "residential_boost":   {"residential": 0.40, "office": 0.20, "commercial": 0.20, "mixed_use": 0.20},
    "residential_boost_strong": {"residential": 0.55, "office": 0.15, "commercial": 0.15, "mixed_use": 0.15},
}


class RewardNormalisingEnv:
    def __init__(self, base_env, mean, std):
        self._env = base_env
        self._mean = mean
        self._std = std if std > 1e-8 else 1.0

    def reset(self):
        return self._env.reset()

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        return state, (reward - self._mean) / self._std, done, info

    def __getattr__(self, name):
        return getattr(self._env, name)


def load_norm_stats():
    if not os.path.exists(NORM_STATS_PATH):
        raise FileNotFoundError(f"{NORM_STATS_PATH} not found. Run 05_reward_normalisation_ablation.py first.")
    with open(NORM_STATS_PATH) as f:
        return json.load(f)


def make_env(archetype, seed, norm_stats):
    base = BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype], episode_len=EPISODE_LEN, seed=seed)
    return RewardNormalisingEnv(base, norm_stats[archetype]["mean"], norm_stats[archetype]["std"])


def evaluate_agent(agent, archetype, seed):
    env = BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype], episode_len=EPISODE_LEN, seed=EVAL_SEED_OFFSET + seed)
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


def sample_weighted_batch(buffers, weights, batch_size):
    """Draw a batch composed of a fixed proportion from each archetype's
    own buffer, then concatenate -- this is the mechanism that gives
    residential a higher "voice" in the shared gradient step."""
    s_all, a_all, r_all, s2_all, d_all = [], [], [], [], []
    for archetype, buf in buffers.items():
        n = max(1, int(round(batch_size * weights[archetype])))
        if len(buf) < n:
            continue
        s, a, r, s2, d = buf.sample(n)
        s_all.append(s); a_all.append(a); r_all.append(r); s2_all.append(s2); d_all.append(d)
    return (torch.cat(s_all), torch.cat(a_all), torch.cat(r_all), torch.cat(s2_all), torch.cat(d_all))


def run_b4_weighted(seed, rounds, steps_per_round, weights, norm_stats):
    torch.manual_seed(seed); np.random.seed(seed)
    envs = {a: make_env(a, seed=seed * 11 + 3 + i, norm_stats=norm_stats) for i, a in enumerate(ALL_ARCHETYPES)}
    agent = DDPGAgent(envs[ALL_ARCHETYPES[0]].state_dim, envs[ALL_ARCHETYPES[0]].action_dim,
                       noise_sigma=0.4, noise_decay_steps=int(rounds * steps_per_round * 0.6))
    # ONE buffer PER archetype now, instead of one pooled buffer
    buffers = {a: ReplayBuffer(capacity=50000) for a in ALL_ARCHETYPES}

    states = {a: envs[a].reset() for a in ALL_ARCHETYPES}
    total_steps = rounds * steps_per_round
    warmup = 800
    min_buffer_needed = max(1, int(round(BATCH_SIZE * min(weights.values())))) + 1

    for i in range(total_steps):
        a_name = ALL_ARCHETYPES[i % len(ALL_ARCHETYPES)]  # data collection stays round-robin (fair)
        env = envs[a_name]
        s = states[a_name]
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)
        buffers[a_name].push(s, a, r, s2, done)
        states[a_name] = env.reset() if done else s2

        if i >= warmup and all(len(buf) > min_buffer_needed for buf in buffers.values()):
            batch = sample_weighted_batch(buffers, weights, BATCH_SIZE)  # WEIGHTED, not uniform pooled
            agent.train_step(batch)

    return {a: evaluate_agent(agent, a, seed) for a in ALL_ARCHETYPES}


def load_b1_baseline(archetype, n_seeds):
    path = os.path.join(ORIGINAL_NORMALISED_DIR, f"{archetype}_ablation.json")
    with open(path) as f:
        data = json.load(f)
    return data["B1"][:n_seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_ROUND)
    parser.add_argument("--weights", choices=list(WEIGHT_PRESETS.keys()), default="residential_boost")
    parser.add_argument("--out_dir", default="results/weighted_sampling_pilot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    norm_stats = load_norm_stats()
    weights = WEIGHT_PRESETS[args.weights]
    print(f"Batch composition weights ({args.weights}): {weights}")

    out_path = os.path.join(args.out_dir, f"b4_{args.weights}.json")
    results = {a: [] for a in ALL_ARCHETYPES}
    if os.path.exists(out_path):
        with open(out_path) as f:
            results = json.load(f)

    n_done = min(len(results[a]) for a in ALL_ARCHETYPES)
    for seed in range(n_done, args.n_seeds):
        print(f"[{args.weights}] seed {seed+1}/{args.n_seeds}")
        vals = run_b4_weighted(seed, args.rounds, args.steps, weights, norm_stats)
        for a in ALL_ARCHETYPES:
            results[a].append(vals[a])
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))

    print(f"\n--- Weighting={args.weights} vs. B1, residual gap analysis ---")
    for a in ALL_ARCHETYPES:
        b1_vals = np.array(load_b1_baseline(a, args.n_seeds))
        b4_vals = np.array(results[a][:args.n_seeds])
        diff = b4_vals - b1_vals
        try:
            _, p = stats.wilcoxon(diff)
        except Exception:
            p = float("nan")
        print(f"  {a:12s} B4({args.weights})-B1 mean diff = {diff.mean():+.4f}   p = {p:.6f}   n={len(diff)}")

    if args.weights != "equal":
        h64_path = os.path.join(ORIGINAL_NORMALISED_DIR, "residential_ablation.json")
        with open(h64_path) as f:
            h64_data = json.load(f)
        h64_diff = (np.array(h64_data["B4"][:args.n_seeds]) - np.array(h64_data["B1"][:args.n_seeds])).mean()
        new_diff = (np.array(results["residential"][:args.n_seeds]) - np.array(load_b1_baseline("residential", args.n_seeds))).mean()
        print(f"\n  RESIDENTIAL residual-gap comparison: equal-weight gap={h64_diff:+.4f}  ->  {args.weights} gap={new_diff:+.4f}")
        if abs(new_diff) < abs(h64_diff) * 0.5:
            print("  -> WEIGHTED-SAMPLING HYPOTHESIS SUPPORTED: upweighting residential substantially reduced the residual gap.")
        elif abs(new_diff) < abs(h64_diff):
            print("  -> Modest improvement; sample weighting may be a minor contributing factor.")
        else:
            print("  -> NOT SUPPORTED: upweighting did not reduce the residual gap.")


if __name__ == "__main__":
    main()

"""
05_reward_normalisation_ablation.py
=====================================
PHASE 1 of the Nature-tier roadmap: tests whether the residential-specific
B4 (centralised oracle) disadvantage found in the completed ablation
(Appendix C.13: B4 significantly worse than B1/B2/B3 for residential,
p<0.0001 in all three comparisons) is caused by reward-scale heterogeneity
across archetypes, and whether normalising rewards before pooling/
aggregation fixes it.

DESIGN: a RewardNormalisingEnv WRAPPER around your existing
BuildingEnergyEnv -- does NOT modify environment.py. It intercepts
step()'s reward and rescales it using per-archetype statistics gathered
in a short calibration pass, while leaving info['cost'] and
info['discomfort'] untouched (so evaluation/reporting stays on the real,
interpretable cost scale -- only the RL training signal changes).

This lets you run the IDENTICAL B1-B4 ablation protocol as
01_full_ablation_b1_b4.py, just with normalised rewards, and directly
compare: does B4's disadvantage for residential shrink or disappear?

Run:
    python 05_reward_normalisation_ablation.py --n_seeds 30
    python 05_reward_normalisation_ablation.py --n_seeds 30 --skip_b4   # faster first pass
    python 05_reward_normalisation_ablation.py --compare_only           # just re-run the stats
        (use --compare_only once both the original (Appendix C.13) and
        normalised results exist, to print the before/after comparison
        without retraining anything)
"""
import argparse
import json
import os
import numpy as np
import torch
from scipy import stats

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
CALIBRATION_EPISODES = 20          # random-action episodes used to estimate reward mean/std per archetype
NORM_STATS_PATH = "results/reward_norm_stats.json"


# ---------------------------------------------------------------------------
# Reward normalisation wrapper
# ---------------------------------------------------------------------------
class RewardNormalisingEnv:
    """Wraps BuildingEnergyEnv; z-score normalises the reward returned by
    step() using FIXED, pre-calibrated per-archetype statistics (mean,
    std). info['cost'] and info['discomfort'] pass through unchanged.
    Using fixed (not running/online) statistics keeps the normalisation
    stable across federated rounds and avoids the calibration itself
    depending on the policy being trained, which would confound the
    comparison to the original (unnormalised) ablation.
    """
    def __init__(self, base_env, mean, std):
        self._env = base_env
        self._mean = mean
        self._std = std if std > 1e-8 else 1.0

    def reset(self):
        return self._env.reset()

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        normalised_reward = (reward - self._mean) / self._std
        return state, normalised_reward, done, info

    def __getattr__(self, name):
        # forward anything else (state_dim, action_dim, price_scale, ...)
        return getattr(self._env, name)


def calibrate_reward_stats(archetype, seed):
    """Run CALIBRATION_EPISODES with random actions to estimate this
    archetype's raw reward mean/std, BEFORE any training happens."""
    env = BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype], episode_len=EPISODE_LEN, seed=seed)
    rewards = []
    for _ in range(CALIBRATION_EPISODES):
        s = env.reset()
        done = False
        while not done:
            a = np.random.uniform(-1, 1, size=env.action_dim)
            s, r, done, info = env.step(a)
            rewards.append(r)
    return float(np.mean(rewards)), float(np.std(rewards))


def get_or_compute_norm_stats():
    """Calibrate once, reuse across all seeds (this is a property of the
    archetype/environment, not of any particular seed)."""
    if os.path.exists(NORM_STATS_PATH):
        with open(NORM_STATS_PATH) as f:
            stats_dict = json.load(f)
        print("Loaded existing reward normalisation stats:", stats_dict)
        return stats_dict

    os.makedirs(os.path.dirname(NORM_STATS_PATH), exist_ok=True)
    stats_dict = {}
    for a in ALL_ARCHETYPES:
        mean, std = calibrate_reward_stats(a, seed=12345)
        stats_dict[a] = {"mean": mean, "std": std}
        print(f"[calibration] {a}: reward mean={mean:.4f}  std={std:.4f}")
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats_dict, f, indent=2)
    return stats_dict


def make_env(archetype, seed, norm_stats=None):
    base = BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype], episode_len=EPISODE_LEN, seed=seed)
    if norm_stats is None:
        return base
    return RewardNormalisingEnv(base, norm_stats[archetype]["mean"], norm_stats[archetype]["std"])


def evaluate_rule_based(archetype, seed):
    """Evaluation is ALWAYS on raw cost (info['cost']), never on the
    normalised reward -- normalisation only affects training signal."""
    env = BuildingEnergyEnv(data_path=ARCHETYPE_PATHS[archetype], episode_len=EPISODE_LEN, seed=EVAL_SEED_OFFSET + seed)
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


def local_train_round(agent, env, buf, steps, warmup=0):
    s = env.reset()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)   # r is NORMALISED if env is wrapped
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE), mu=1.0)
        if done:
            s = env.reset()


# ---------------------------------------------------------------------------
# B1 (rule-based; no training, unaffected by normalisation by construction)
# ---------------------------------------------------------------------------
def run_b1_all_archetypes(seed):
    return {a: evaluate_rule_based(a, seed) for a in ALL_ARCHETYPES}


# ---------------------------------------------------------------------------
# B2 (independent DDPG, trained on NORMALISED reward, evaluated on RAW cost)
# ---------------------------------------------------------------------------
def run_b2_all_archetypes(seed, rounds, steps_per_round, norm_stats):
    out = {}
    for a in ALL_ARCHETYPES:
        torch.manual_seed(seed); np.random.seed(seed)
        env = make_env(a, seed=seed * 11 + 3, norm_stats=norm_stats)
        agent = DDPGAgent(env.state_dim, env.action_dim,
                           noise_sigma=0.4, noise_decay_steps=int(rounds * steps_per_round * 0.6))
        buf = ReplayBuffer()
        for rnd in range(1, rounds + 1):
            warmup = 800 if rnd == 1 else 0
            local_train_round(agent, env, buf, steps_per_round, warmup=warmup)
        out[a] = evaluate_agent(agent, a, seed)
    return out


# ---------------------------------------------------------------------------
# B3 (federated, no personalisation; NORMALISED reward on all 4 together)
# ---------------------------------------------------------------------------
def run_b3_all_archetypes(seed, rounds, steps_per_round, norm_stats):
    torch.manual_seed(seed); np.random.seed(seed)
    envs = {a: make_env(a, seed=seed * 11 + 3 + i, norm_stats=norm_stats) for i, a in enumerate(ALL_ARCHETYPES)}
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
            noised_sd, _ = add_gaussian_dp_noise(agents[a].critic.state_dict(), epsilon=DP_EPSILON, delta=DP_DELTA)
            noised_critics.append(noised_sd)
        new_global = fedavg_aggregate(noised_critics)
        global_critic.load_state_dict(new_global)
        for a in ALL_ARCHETYPES:
            agents[a].critic.load_state_dict(new_global)
            agents[a].critic_target.load_state_dict(new_global)

    return {a: evaluate_agent(agents[a], a, seed) for a in ALL_ARCHETYPES}


# ---------------------------------------------------------------------------
# B4 (centralised oracle; THE key test -- pooled NORMALISED reward, so
# archetypes' contributions to the shared network's gradient are on a
# comparable scale, unlike the original ablation's raw-reward pooling)
# ---------------------------------------------------------------------------
def run_b4_all_archetypes(seed, rounds, steps_per_round, norm_stats):
    torch.manual_seed(seed); np.random.seed(seed)
    envs = {a: make_env(a, seed=seed * 11 + 3 + i, norm_stats=norm_stats) for i, a in enumerate(ALL_ARCHETYPES)}
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
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)   # NORMALISED reward, pooled across archetypes
        pooled_buf.push(s, a, r, s2, done)
        states[a_name] = env.reset() if done else s2
        if len(pooled_buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(pooled_buf.sample(BATCH_SIZE))

    return {a: evaluate_agent(agent, a, seed) for a in ALL_ARCHETYPES}


def print_comparison(original_dir="results/ablation_b1_b4", normalised_dir="results/ablation_b1_b4_normalised"):
    """The key analysis: does normalisation close residential's B4 gap?"""
    print("\n" + "=" * 78)
    print("PHASE 1 RESULT: Does reward normalisation fix residential's B4 disadvantage?")
    print("=" * 78)
    for a in ALL_ARCHETYPES:
        orig_path = os.path.join(original_dir, f"{a}_ablation.json")
        norm_path = os.path.join(normalised_dir, f"{a}_ablation.json")
        if not (os.path.exists(orig_path) and os.path.exists(norm_path)):
            print(f"  {a}: missing data (orig={os.path.exists(orig_path)}, norm={os.path.exists(norm_path)})")
            continue
        with open(orig_path) as f: orig = json.load(f)
        with open(norm_path) as f: norm = json.load(f)

        n = min(len(orig["B1"]), len(orig["B4"]), len(norm["B1"]), len(norm["B4"]))
        orig_b4, orig_b1 = np.array(orig["B4"][:n]), np.array(orig["B1"][:n])
        norm_b4, norm_b1 = np.array(norm["B4"][:n]), np.array(norm["B1"][:n])

        orig_diff = orig_b4 - orig_b1
        norm_diff = norm_b4 - norm_b1
        try:
            _, orig_p = stats.wilcoxon(orig_diff)
        except Exception:
            orig_p = float("nan")
        try:
            _, norm_p = stats.wilcoxon(norm_diff)
        except Exception:
            norm_p = float("nan")

        print(f"\n  [{a}]  (n={n})")
        print(f"    ORIGINAL (raw reward):      B4-B1 mean diff = {orig_diff.mean():+.4f}   p = {orig_p:.6f}")
        print(f"    NORMALISED reward:          B4-B1 mean diff = {norm_diff.mean():+.4f}   p = {norm_p:.6f}")
        if a == "residential":
            if orig_p < 0.05 and norm_p >= 0.05:
                print("    -> HYPOTHESIS CONFIRMED: normalisation eliminated residential's significant B4 disadvantage.")
            elif orig_p < 0.05 and norm_p < 0.05 and abs(norm_diff.mean()) < abs(orig_diff.mean()) * 0.5:
                print("    -> PARTIAL SUPPORT: disadvantage shrank substantially but remains significant.")
            elif orig_p < 0.05 and norm_p < 0.05:
                print("    -> NOT CONFIRMED: disadvantage persists after normalisation; reward scale is not the (sole) cause.")
            else:
                print("    -> Inconclusive given current data; check n and re-examine.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_ROUND)
    parser.add_argument("--out_dir", default="results/ablation_b1_b4_normalised")
    parser.add_argument("--skip_b4", action="store_true")
    parser.add_argument("--compare_only", action="store_true",
                         help="Skip training; just compare existing original vs. normalised results.")
    args = parser.parse_args()

    if args.compare_only:
        print_comparison(normalised_dir=args.out_dir)
        return

    norm_stats = get_or_compute_norm_stats()

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
        print(f"\n=== [NORMALISED] seed {seed+1}/{args.n_seeds} (rounds={args.rounds}, steps/round={args.steps}) ===")

        if seed >= n_done("B1"):
            vals = run_b1_all_archetypes(seed)
            for a in ALL_ARCHETYPES: results[a]["B1"].append(vals[a])
            print("  B1 (rule-based):        " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if seed >= n_done("B2"):
            vals = run_b2_all_archetypes(seed, args.rounds, args.steps, norm_stats)
            for a in ALL_ARCHETYPES: results[a]["B2"].append(vals[a])
            print("  B2 (independent DDPG):  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if seed >= n_done("B3"):
            vals = run_b3_all_archetypes(seed, args.rounds, args.steps, norm_stats)
            for a in ALL_ARCHETYPES: results[a]["B3"].append(vals[a])
            print("  B3 (FedAvg, no pers.):  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

        if not args.skip_b4 and seed >= n_done("B4"):
            vals = run_b4_all_archetypes(seed, args.rounds, args.steps, norm_stats)
            for a in ALL_ARCHETYPES: results[a]["B4"].append(vals[a])
            print("  B4 (centralised oracle):" + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))
            save_all()

    print("\nDone. Per-archetype summaries (normalised-reward run):")
    for a in ALL_ARCHETYPES:
        print(f"\n[{a}]  -> {out_paths[a]}")
        for b, vals in results[a].items():
            if vals:
                arr = np.array(vals)
                print(f"  {b}: mean={arr.mean():.4f}  std={arr.std():.4f}  n={len(arr)}")

    if not args.skip_b4:
        print_comparison(normalised_dir=args.out_dir)


if __name__ == "__main__":
    main()

"""
06_capacity_interference_pilot.py
====================================
PHASE 1b of the Nature-tier roadmap: a SMALL PILOT (N=10, not 30) testing
whether network capacity interference explains the ~0.86 residual B4-B1
gap for residential that survived reward normalisation (Phase 1a).

HYPOTHESIS: B4 pools four archetypes with very different control dynamics
into ONE shared network (hidden=64, same as every other agent in this
codebase). If the network's limited capacity is being "contested" by the
larger/more complex archetypes (office, commercial, mixed-use), the small,
fine-grained residential signal may be systematically under-fit -- a
well-documented multi-task learning failure mode, distinct from (and
additive to) the reward-scale issue already addressed.

DESIGN: re-run B4 with a LARGER shared network (hidden=128) on top of the
already-normalised reward (Phase 1a's fix), and compare the residual
B4-B1 gap for residential against the hidden=64 normalised baseline
already collected (results/ablation_b1_b4_normalised). B1/B2/B3 are not
re-run here (they are unaffected by B4's architecture) -- only B4 needs
retraining at each capacity.

This is a PILOT at N=10 to cheaply establish direction before committing
to a full N=30 campaign. If the residual gap shrinks substantially,
extend to N=30 with --n_seeds 30.

Run:
    python 06_capacity_interference_pilot.py --n_seeds 10
    python 06_capacity_interference_pilot.py --n_seeds 10 --hidden 128 256
        (tests multiple capacities in one run, for a dose-response view)
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from environment import BuildingEnergyEnv
from ddpg_agent import Actor, Critic, ReplayBuffer, OUNoise

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


class VariableCapacityDDPGAgent:
    """Same interface as ddpg_agent.DDPGAgent, but lets you specify the
    Actor/Critic hidden layer size -- DDPGAgent itself hardcodes hidden=64
    via Actor(state_dim, action_dim) / Critic(state_dim, action_dim), so
    this is a minimal reimplementation that reuses the SAME Actor/Critic/
    ReplayBuffer/OUNoise classes from ddpg_agent.py, just passing `hidden`
    through -- no logic is duplicated beyond the constructor.
    """
    def __init__(self, state_dim, action_dim, hidden=64, gamma=0.99, tau=0.005,
                 lr_actor=1e-3, lr_critic=1e-3, noise_sigma=0.3, noise_decay_steps=15000):
        self.actor = Actor(state_dim, action_dim, hidden=hidden)
        self.actor_target = Actor(state_dim, action_dim, hidden=hidden)
        self.critic = Critic(state_dim, action_dim, hidden=hidden)
        self.critic_target = Critic(state_dim, action_dim, hidden=hidden)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau
        self.noise = OUNoise(action_dim, sigma=noise_sigma, decay_steps=noise_decay_steps)
        self.action_dim = action_dim

    def act(self, state, explore=True):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(s).squeeze(0).numpy()
        if explore:
            a = a + self.noise.sample()
        return np.clip(a, -1.0, 1.0)

    def soft_update(self, target, source):
        for tp, p in zip(target.parameters(), source.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self, batch):
        import torch.nn.functional as F
        s, a, r, s2, d = batch
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target_next = self.critic_target(s2, a2)
            y = r + self.gamma * (1 - d) * q_target_next
        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad(); critic_loss.backward(); self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


class RewardNormalisingEnv:
    """Identical to 05_reward_normalisation_ablation.py's wrapper -- reused
    here so this script only needs the pre-computed calibration stats."""
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
        raise FileNotFoundError(
            f"{NORM_STATS_PATH} not found. Run 05_reward_normalisation_ablation.py "
            f"first (even just --n_seeds 1 --skip_b4) to generate calibration stats."
        )
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


def run_b4_with_capacity(seed, rounds, steps_per_round, hidden, norm_stats):
    """B4 (centralised oracle, normalised reward) at a given network capacity."""
    torch.manual_seed(seed); np.random.seed(seed)
    envs = {a: make_env(a, seed=seed * 11 + 3 + i, norm_stats=norm_stats) for i, a in enumerate(ALL_ARCHETYPES)}
    agent = VariableCapacityDDPGAgent(
        envs[ALL_ARCHETYPES[0]].state_dim, envs[ALL_ARCHETYPES[0]].action_dim, hidden=hidden,
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
        s2, r, done, info = env.step(a)
        pooled_buf.push(s, a, r, s2, done)
        states[a_name] = env.reset() if done else s2
        if len(pooled_buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(pooled_buf.sample(BATCH_SIZE))

    return {a: evaluate_agent(agent, a, seed) for a in ALL_ARCHETYPES}


def load_b1_baseline(archetype, n_seeds):
    """Reuse Phase 1a's already-collected B1 (rule-based) results -- B1
    doesn't depend on B4's capacity, so no need to recompute it."""
    path = os.path.join(ORIGINAL_NORMALISED_DIR, f"{archetype}_ablation.json")
    with open(path) as f:
        data = json.load(f)
    return data["B1"][:n_seeds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=10, help="Pilot size; extend to 30 if promising.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_ROUND)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128],
                         help="One or more hidden sizes to test (default network is 64).")
    parser.add_argument("--out_dir", default="results/capacity_pilot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    norm_stats = load_norm_stats()

    for hidden in args.hidden:
        print(f"\n{'='*70}\nTesting hidden={hidden}\n{'='*70}")
        out_path = os.path.join(args.out_dir, f"b4_hidden{hidden}.json")
        results = {a: [] for a in ALL_ARCHETYPES}
        if os.path.exists(out_path):
            with open(out_path) as f:
                results = json.load(f)

        n_done = min(len(results[a]) for a in ALL_ARCHETYPES)
        for seed in range(n_done, args.n_seeds):
            print(f"[hidden={hidden}] seed {seed+1}/{args.n_seeds}")
            vals = run_b4_with_capacity(seed, args.rounds, args.steps, hidden, norm_stats)
            for a in ALL_ARCHETYPES:
                results[a].append(vals[a])
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print("  " + ", ".join(f"{a}={vals[a]:.3f}" for a in ALL_ARCHETYPES))

        # Compare against B1 baseline (reused from Phase 1a) and against
        # the hidden=64 normalised B4 result, for residential specifically.
        print(f"\n--- Capacity={hidden} vs. B1, residual gap analysis ---")
        for a in ALL_ARCHETYPES:
            b1_vals = np.array(load_b1_baseline(a, args.n_seeds))
            b4_vals = np.array(results[a][:args.n_seeds])
            diff = b4_vals - b1_vals
            try:
                _, p = stats.wilcoxon(diff)
            except Exception:
                p = float("nan")
            print(f"  {a:12s} B4(h={hidden})-B1 mean diff = {diff.mean():+.4f}   p = {p:.6f}   n={len(diff)}")

        if hidden != 64:
            baseline_h64_path = os.path.join(ORIGINAL_NORMALISED_DIR, "residential_ablation.json")
            with open(baseline_h64_path) as f:
                h64_data = json.load(f)
            h64_b4 = np.array(h64_data["B4"][:args.n_seeds])
            h64_b1 = np.array(h64_data["B1"][:args.n_seeds])
            h64_diff = (h64_b4 - h64_b1).mean()
            new_diff = (np.array(results["residential"][:args.n_seeds]) - np.array(load_b1_baseline("residential", args.n_seeds))).mean()
            print(f"\n  RESIDENTIAL residual-gap comparison: hidden=64 gap={h64_diff:+.4f}  ->  hidden={hidden} gap={new_diff:+.4f}")
            if abs(new_diff) < abs(h64_diff) * 0.5:
                print("  -> CAPACITY HYPOTHESIS SUPPORTED: larger network substantially reduced the residual gap further.")
            elif abs(new_diff) < abs(h64_diff):
                print("  -> Modest improvement; capacity may be a minor contributing factor.")
            else:
                print("  -> NOT SUPPORTED: capacity increase did not reduce the residual gap. Look elsewhere (e.g. per-archetype learning rate, state feature engineering).")


if __name__ == "__main__":
    main()

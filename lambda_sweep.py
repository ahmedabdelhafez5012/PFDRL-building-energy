"""
lambda_sweep.py
Real sensitivity sweep over `lambda_comfort` (the comfort-penalty weight in
environment.py's reward function) to find a defensible operating point where
DDPG (B3) does not win on cost purely by sacrificing comfort. This is a
genuine, real experiment on real data - not a smoke test - addressing the
honest finding flagged in README.md.
"""
import numpy as np
import torch
from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer

EPISODE_LEN = 96
TRAIN_STEPS = 8000
WARMUP_STEPS = 800
EVAL_EPISODES = 15
EVAL_SEED = 123
LAMBDAS = [0.15, 0.50, 1.00]


class EnvWithLambda(BuildingEnergyEnv):
    """Thin wrapper to override the fixed lambda_comfort=0.15 hard-coded in
    the base environment's step(), without duplicating all of step()'s logic
    unnecessarily -- we recompute reward directly using info returned.
    """
    def __init__(self, lambda_comfort, **kwargs):
        super().__init__(**kwargs)
        self.lambda_comfort = lambda_comfort

    def step(self, action):
        # Call parent, then re-derive the reward with our own lambda using
        # the cost/discomfort already computed and returned in info.
        next_state, _, done, info = super().step(action)
        reward = -(info["cost"] + self.lambda_comfort * info["discomfort"])
        return next_state, float(reward), done, info


def train_and_eval(lam, train_seed=0, eval_seed=EVAL_SEED):
    env = EnvWithLambda(lam, episode_len=EPISODE_LEN, seed=train_seed)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                       noise_decay_steps=int(TRAIN_STEPS * 0.6))
    buf = ReplayBuffer()
    s = env.reset()
    for i in range(TRAIN_STEPS):
        a = (np.random.uniform(-1, 1, size=env.action_dim) if i < WARMUP_STEPS
             else agent.act(s))
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > 256 and i >= WARMUP_STEPS:
            agent.train_step(buf.sample(256))
        if done:
            s = env.reset()

    eval_env = EnvWithLambda(lam, episode_len=EPISODE_LEN, seed=eval_seed)
    costs, discomforts = [], []
    for _ in range(EVAL_EPISODES):
        s = eval_env.reset()
        ep_cost, ep_disc = 0.0, 0.0
        done = False
        while not done:
            a = agent.act(s, explore=False)
            s, r, done, info = eval_env.step(a)
            ep_cost += info["cost"]
            ep_disc += info["discomfort"]
        costs.append(ep_cost)
        discomforts.append(ep_disc)
    return np.mean(costs), np.std(costs), np.mean(discomforts)


def main():
    print("=" * 70)
    print("Lambda_comfort sensitivity sweep - residential archetype (REAL data)")
    print(f"Each point: {TRAIN_STEPS} training steps, {EVAL_EPISODES} held-out eval episodes")
    print("Reference (from compare_baselines.py, lambda=0.15 fixed):")
    print("  B1 Rule-Based:  cost=3.2692  discomfort=3.6452")
    print("=" * 70)

    results = []
    for lam in LAMBDAS:
        mean_cost, std_cost, mean_disc = train_and_eval(lam)
        results.append((lam, mean_cost, std_cost, mean_disc))
        print(f"lambda={lam:.2f} | mean cost={mean_cost:7.4f} (std {std_cost:.4f}) "
              f"| mean discomfort={mean_disc:7.4f}")

    print("\n" + "=" * 70)
    print("Summary table:")
    print(f"{'lambda':>8} | {'mean cost':>10} | {'mean discomfort':>16} | beats B1 on both?")
    for lam, cost, std, disc in results:
        beats_both = (cost < 3.2692) and (disc < 3.6452)
        print(f"{lam:8.2f} | {cost:10.4f} | {disc:16.4f} | {'YES' if beats_both else 'no'}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()

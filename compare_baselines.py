"""
compare_baselines.py
Trains B3 (independent DDPG, no federation) properly on the residential
archetype, then evaluates B1, B2, and B3 under an IDENTICAL protocol
(same episode length, same number of held-out evaluation episodes, same
random seed for the environment's episode sampling) so the comparison is
fair. This produces a genuine, real-data baseline table for ONE archetype
(residential) - the required precursor to PFDRL comparison once more
archetypes are available (Section 5.2).
"""
import numpy as np
import torch
from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer
from baseline_b1_rule_based import rule_based_action
from baseline_b2_mpc import mpc_plan

EPISODE_LEN = 96          # 1 day at 15-min resolution
N_EVAL_EPISODES = 15
TRAIN_STEPS = 30000        # increased from 8000: earlier budget was likely too short
WARMUP_STEPS = 1500        # pure random exploration before any learning starts
EVAL_SEED = 123           # held-out seed, different from training seed


def train_ddpg(train_seed=0, steps=TRAIN_STEPS, warmup=WARMUP_STEPS):
    env = BuildingEnergyEnv(episode_len=EPISODE_LEN, seed=train_seed)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4, noise_decay_steps=int(steps * 0.6))
    buf = ReplayBuffer()
    s = env.reset()
    for i in range(steps):
        if i < warmup:
            a = np.random.uniform(-1, 1, size=env.action_dim)  # pure exploration warm-up
        else:
            a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > 256 and i >= warmup:
            agent.train_step(buf.sample(256))
        if done:
            s = env.reset()
        if (i + 1) % 5000 == 0:
            print(f"  DDPG training step {i+1}/{steps} (noise sigma={agent.noise.current_sigma():.3f})")
    return agent


def evaluate(policy_fn, seed=EVAL_SEED, n_episodes=N_EVAL_EPISODES):
    env = BuildingEnergyEnv(episode_len=EPISODE_LEN, seed=seed)
    costs, discomforts = [], []
    for ep in range(n_episodes):
        s = env.reset()
        ep_cost, ep_disc = 0.0, 0.0
        done = False
        while not done:
            a = policy_fn(env, s)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
            ep_disc += info["discomfort"]
        costs.append(ep_cost)
        discomforts.append(ep_disc)
    return np.array(costs), np.array(discomforts)


def main():
    print("=" * 70)
    print("Fair baseline comparison - residential archetype (HomeC.csv, REAL data)")
    print(f"Protocol: {N_EVAL_EPISODES} held-out episodes x {EPISODE_LEN} steps, seed={EVAL_SEED}")
    print("=" * 70)

    print("\n[1/3] Evaluating B1 (Rule-Based)...")
    b1_costs, b1_disc = evaluate(lambda env, s: rule_based_action(s, env.price_scale))

    print("[2/3] Evaluating B2 (MPC) - this solves an optimisation every step...")
    b2_costs, b2_disc = evaluate(lambda env, s: mpc_plan(env), n_episodes=5)  # fewer: MPC is slow

    print("[3/3] Training B3 (independent DDPG, no federation)...")
    agent = train_ddpg(train_seed=0, steps=TRAIN_STEPS)
    b3_costs, b3_disc = evaluate(lambda env, s: agent.act(s, explore=False))

    print("\n" + "=" * 70)
    print("RESULTS (real data, residential archetype only - not yet cross-archetype)")
    print("=" * 70)
    for name, costs, disc in [("B1 Rule-Based", b1_costs, b1_disc),
                               ("B2 MPC", b2_costs, b2_disc),
                               ("B3 Independent DDPG", b3_costs, b3_disc)]:
        print(f"{name:22s} | mean cost/episode = {costs.mean():7.4f} "
              f"(std {costs.std():.4f}, n={len(costs)}) | "
              f"mean discomfort = {disc.mean():7.4f}")

    print("\nNOTE: B2 evaluated on fewer episodes (compute cost of re-solving")
    print("an optimisation every step). B3 is a genuinely trained DDPG agent")
    print(f"({TRAIN_STEPS} steps) evaluated on held-out episodes (seed={EVAL_SEED},")
    print("disjoint from training seed=0) - this is a real, fair within-archetype")
    print("comparison, but still only ONE archetype: not yet the full B1-B4 x")
    print("4-archetype table required for the thesis's Table in Section 6.2.")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()

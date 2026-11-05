"""
baseline_b1_rule_based.py
B1 — Rule-Based control (Section 5.2). A static heuristic controller:
- HVAC: simple deadband thermostat logic driven by outdoor temperature.
- ESS: charge during off-peak price hours, discharge during peak hours.
No learning involved; this is the classical baseline PFDRL is compared against.
"""
import numpy as np
from environment import BuildingEnergyEnv


def rule_based_action(state, price_scale):
    base, hvac, pv, price, t_out, soc, hour_frac = state
    price_real = price * price_scale

    # HVAC rule: modest, temperature-triggered boost/reduction around nominal
    if t_out < 0.0:   # cold (state is normalised t_out/40)
        hvac_action = 0.3
    elif t_out > 0.6:  # hot
        hvac_action = 0.3
    else:
        hvac_action = -0.2  # mild weather -> reduce load slightly

    # ESS rule: charge in off-peak / cheap price, discharge in peak price
    if price_real <= 0.15:
        ess_action = 0.8 if soc < 0.9 else 0.0
    elif price_real >= 0.35:
        ess_action = -0.8 if soc > 0.1 else 0.0
    else:
        ess_action = 0.0

    return np.array([hvac_action, ess_action], dtype=np.float32)


def evaluate_baseline(n_episodes=20, episode_len=96, seed=0):
    env = BuildingEnergyEnv(episode_len=episode_len, seed=seed)
    costs, discomforts = [], []
    for ep in range(n_episodes):
        s = env.reset()
        ep_cost, ep_disc = 0.0, 0.0
        done = False
        while not done:
            a = rule_based_action(s, env.price_scale)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
            ep_disc += info["discomfort"]
        costs.append(ep_cost)
        discomforts.append(ep_disc)
    return {
        "mean_cost_per_episode": float(np.mean(costs)),
        "std_cost_per_episode": float(np.std(costs)),
        "mean_discomfort_per_episode": float(np.mean(discomforts)),
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    results = evaluate_baseline(n_episodes=30)
    print("B1 Rule-Based baseline (residential archetype, REAL data):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

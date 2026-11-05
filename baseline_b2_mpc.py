"""
baseline_b2_mpc.py
B2 — Model Predictive Control (Section 5.2). Solves a short receding-horizon
optimisation (H=4 steps = 1 hour) at each control step using SciPy, assuming
perfect-foresight forecasts of price/PV/HVAC-nominal-demand over the horizon
(a standard, documented MPC-baseline simplification - see Section 5.2 note
below). This is real optimisation code running against the REAL residential
data; the "perfect foresight" assumption is a modelling choice to keep, since
its purpose is to serve as a strong classical baseline, not a deployable
controller.
"""
import numpy as np
from scipy.optimize import minimize
from environment import BuildingEnergyEnv

HORIZON = 4  # 4 x 15-min = 1 hour lookahead


def mpc_plan(env, horizon=HORIZON):
    """Optimises HVAC and ESS actions over the horizon using the same cost
    model as the environment's reward (perfect-foresight on price/PV/HVAC
    nominal demand for the next `horizon` steps), then returns only the
    FIRST action (standard receding-horizon / MPC practice).
    """
    t0 = env._t
    rows = [env._row(min(t0 + k, env.n - 1)) for k in range(horizon)]
    soc0 = env.soc

    def cost_fn(flat_actions):
        actions = flat_actions.reshape(horizon, 2)
        soc = soc0
        total_cost = 0.0
        for k in range(horizon):
            row = rows[k]
            a = np.clip(actions[k], -1.0, 1.0)
            hvac_nom = row["hvac_kw"]
            hvac_actual = np.clip(hvac_nom * (1.0 + 0.6 * a[0]), 0.0, env.hvac_scale * 1.5)
            discomfort = abs(hvac_actual - hvac_nom)

            ess_rate = a[1] * env.ess_max_rate
            dt_h = 0.25
            if ess_rate >= 0:
                delta_soc = ess_rate * dt_h * env.ess_eff / env.ess_capacity
            else:
                delta_soc = ess_rate * dt_h / env.ess_eff / env.ess_capacity
            soc = np.clip(soc + delta_soc, 0.0, 1.0)
            actual_ess_kw = (soc - (soc - delta_soc)) * env.ess_capacity / dt_h

            net = row["base_kw"] + hvac_actual + actual_ess_kw - row["pv_kw"]
            grid_import = max(net, 0.0)
            grid_export = max(-net, 0.0)
            cost = grid_import * row["price"] * dt_h - grid_export * (row["price"] * 0.5) * dt_h
            total_cost += cost + 0.15 * discomfort
        return total_cost

    x0 = np.zeros(horizon * 2)
    bounds = [(-1.0, 1.0)] * (horizon * 2)
    res = minimize(cost_fn, x0, method="L-BFGS-B", bounds=bounds,
                    options={"maxiter": 50})
    first_action = res.x.reshape(horizon, 2)[0]
    return np.clip(first_action, -1.0, 1.0)


def evaluate_baseline(n_episodes=10, episode_len=96, seed=0):
    """Fewer episodes than B1 by default: MPC solves an optimisation at every
    single step, so it is far more compute-intensive (real trade-off, worth
    reporting in the thesis: MPC's O(n^3)-scaling cost vs. DRL's fast
    inference once trained - see the Design Rationale box on datasets/DDPG).
    """
    env = BuildingEnergyEnv(episode_len=episode_len, seed=seed)
    costs, discomforts = [], []
    for ep in range(n_episodes):
        s = env.reset()
        ep_cost, ep_disc = 0.0, 0.0
        done = False
        while not done:
            a = mpc_plan(env)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
            ep_disc += info["discomfort"]
        costs.append(ep_cost)
        discomforts.append(ep_disc)
        print(f"  episode {ep+1}/{n_episodes} done, cost={ep_cost:.4f}")
    return {
        "mean_cost_per_episode": float(np.mean(costs)),
        "std_cost_per_episode": float(np.std(costs)),
        "mean_discomfort_per_episode": float(np.mean(discomforts)),
        "n_episodes": n_episodes,
    }


if __name__ == "__main__":
    print("Running B2 MPC baseline (this is compute-intensive: an optimisation")
    print("is solved at every 15-minute step)...")
    results = evaluate_baseline(n_episodes=5, episode_len=48)  # shorter for speed
    print("\nB2 MPC baseline (residential archetype, REAL data):")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

"""
multi_seed_h3_latency_test.py
Statistical protocol (Section 5.4) for the latency half of Hypothesis H3:
"SHAP/LIME explanation generation stays within a 1-second control-loop
budget." Unlike H1/H2/H4, this is not a training-seed comparison - it is a
repeated TIMING measurement on a single trained model, since latency does
not depend on training randomness the way cost does.

The acceptance half of H3 (operator trust/acceptance) is Scenario S5 (the
human study) and is NOT covered by this script - see
S5_Study_Materials_Bilingual.docx.

Requires a trained actor (trained_actor_<archetype>.pt) produced by
main_real_federation.py (v11+) - falls back to an untrained actor with a
clear warning if not found, so the script remains runnable for a pipeline
smoke test, but the reported latencies should be re-collected on a genuinely
trained model for the thesis.
"""
import time
import os
import numpy as np
import torch
from scipy import stats

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent
from xai_module import run_shap_global, run_lime_local

N_REPETITIONS = 30
BUDGET_SECONDS = 1.0

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv"},
    {"name": "office", "path": "data/office_archetype_15min.csv"},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv"},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv"},
]


def load_agent_or_warn(archetype_name, state_dim, action_dim):
    agent = DDPGAgent(state_dim, action_dim)
    path = f"trained_actor_{archetype_name}.pt"
    if os.path.exists(path):
        agent.actor.load_state_dict(torch.load(path))
        print(f"  [OK] loaded trained actor: {path}")
    else:
        print(f"  [WARNING] {path} not found - using an UNTRAINED actor. "
              f"Latencies below are a code-verification measurement only, "
              f"NOT valid for the thesis until re-run with a real trained model.")
    return agent


def collect_background_states(env, agent, n=150):
    states = []
    s = env.reset()
    for _ in range(n):
        states.append(s)
        s, r, done, info = env.step(agent.act(s, explore=False))
        if done:
            s = env.reset()
    return np.array(states, dtype=np.float32)


def run():
    print("=" * 78)
    print(f"H3 latency test: {N_REPETITIONS} repetitions per archetype, "
          f"budget={BUDGET_SECONDS}s")
    print("=" * 78)

    all_shap_latencies = {}
    all_lime_latencies = {}

    for archetype in ARCHETYPES:
        print(f"\n--- {archetype['name'].upper()} ---")
        env = BuildingEnergyEnv(data_path=archetype["path"], seed=0)
        agent = load_agent_or_warn(archetype["name"], env.state_dim, env.action_dim)
        states = collect_background_states(env, agent, n=150)

        shap_latencies = []
        lime_latencies = []
        for rep in range(N_REPETITIONS):
            _, shap_ms = run_shap_global(agent.actor, states, n_samples=60)
            shap_latencies.append(shap_ms / 1000.0)  # convert to seconds

            idx = np.random.randint(0, len(states))
            _, lime_ms = run_lime_local(agent.actor, states, states[idx])
            lime_latencies.append(lime_ms / 1000.0)

            if (rep + 1) % 10 == 0:
                print(f"  {rep + 1}/{N_REPETITIONS} repetitions done")

        all_shap_latencies[archetype["name"]] = shap_latencies
        all_lime_latencies[archetype["name"]] = lime_latencies

        shap_arr = np.array(shap_latencies)
        lime_arr = np.array(lime_latencies)
        print(f"  SHAP: mean={shap_arr.mean():.4f}s, std={shap_arr.std():.4f}s, "
              f"95th pct={np.percentile(shap_arr, 95):.4f}s, max={shap_arr.max():.4f}s")
        print(f"  LIME: mean={lime_arr.mean():.4f}s, std={lime_arr.std():.4f}s, "
              f"95th pct={np.percentile(lime_arr, 95):.4f}s, max={lime_arr.max():.4f}s")

    print("\n" + "=" * 78)
    print(f"ONE-SAMPLE T-TEST: mean latency < {BUDGET_SECONDS}s budget")
    print("=" * 78)
    for name in all_shap_latencies:
        combined = np.array(all_shap_latencies[name] + all_lime_latencies[name])
        t_stat, p_two_sided = stats.ttest_1samp(combined, BUDGET_SECONDS)
        # one-sided: is mean significantly LESS than budget?
        p_one_sided = p_two_sided / 2 if t_stat < 0 else 1 - p_two_sided / 2
        sig = "SIGNIFICANTLY within budget (p<0.05)" if p_one_sided < 0.05 else "not significantly within budget"
        within_budget_pct = 100 * np.mean(combined < BUDGET_SECONDS)
        print(f"  {name:12s} | mean={combined.mean():.4f}s | {within_budget_pct:.1f}% of measurements "
              f"within budget | {sig}")

    print("\nRaw latencies (seconds):")
    print("SHAP:", all_shap_latencies)
    print("LIME:", all_lime_latencies)


if __name__ == "__main__":
    np.random.seed(3)
    run()

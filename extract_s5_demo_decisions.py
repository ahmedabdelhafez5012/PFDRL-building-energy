"""
extract_s5_demo_decisions.py
Extracts 2-3 REAL decisions (with real SHAP/LIME explanations) from your
trained models, formatted directly for use in the S5 demonstration script
(Section 5 of S5_Study_Materials_Bilingual.docx) - replacing the
illustrative example there with genuine model output.

Requires trained_actor_<archetype>.pt files (produced by
main_real_federation.py) in the working directory.

Usage: python extract_s5_demo_decisions.py > s5_real_decisions.txt
Then copy the printed blocks into the "Worked Example" section of the S5
materials document (or read directly from this file during the session).
"""
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent
from xai_module import run_shap_global, run_lime_local, STATE_NAMES

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv", "label": "Residential"},
    {"name": "office", "path": "data/office_archetype_15min.csv", "label": "Office"},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv", "label": "Commercial"},
]

FRIENDLY_NAMES = {
    "base_load": "base (non-HVAC) load",
    "hvac_demand": "HVAC demand",
    "pv_gen": "solar generation",
    "price": "electricity price",
    "t_out": "outdoor temperature",
    "soc": "battery state-of-charge",
    "hour_frac": "time of day",
}


def describe_state(state, env):
    """Turn a normalised state vector back into human-readable values."""
    base_kw = state[0] * env.base_scale
    hvac_kw = state[1] * env.hvac_scale
    pv_kw = state[2] * env.pv_scale
    price = state[3] * env.price_scale
    t_out = state[4] * 40.0
    soc = state[5]
    hour = int(state[6] * 24)
    return (f"base load {base_kw:.2f} kW, HVAC demand {hvac_kw:.2f} kW, "
            f"solar {pv_kw:.2f} kW, price ${price:.2f}/kWh, outdoor temp {t_out:.1f}\u00b0C, "
            f"battery {soc*100:.0f}% charged, hour {hour:02d}:00")


def describe_action(action):
    hvac_pct = action[0] * 60  # matches env's 0.6 scaling of hvac modulation
    ess_direction = "charging" if action[1] > 0 else "discharging"
    ess_pct = abs(action[1]) * 100
    return (f"adjust HVAC by {hvac_pct:+.1f}% around nominal demand; "
            f"{ess_direction} battery at {ess_pct:.0f}% of max rate")


def run():
    print("=" * 78)
    print("REAL demonstration decisions extracted from trained models")
    print("For use in S5_Study_Materials_Bilingual.docx, Section 5")
    print("=" * 78)

    for archetype in ARCHETYPES:
        print(f"\n{'#' * 70}")
        print(f"# {archetype['label']} Archetype")
        print(f"{'#' * 70}")

        env = BuildingEnergyEnv(data_path=archetype["path"], seed=7)
        agent = DDPGAgent(env.state_dim, env.action_dim)
        try:
            agent.actor.load_state_dict(torch.load(f"trained_actor_{archetype['name']}.pt"))
            print("[Using genuinely trained model]")
        except FileNotFoundError:
            print("[WARNING: trained_actor file not found - using UNTRAINED model. "
                  "Re-run after training completes for a valid demo decision.]")

        # collect a background pool and pick one representative state (mid-episode)
        states = []
        s = env.reset()
        for _ in range(96):
            states.append(s)
            s, r, done, info = env.step(agent.act(s, explore=False))
            if done:
                s = env.reset()
        states = np.array(states, dtype=np.float32)
        instance = states[40]  # a representative afternoon/evening state

        action = agent.act(instance, explore=False)

        print("\n--- Context (for Sec. 5, Step 1) ---")
        print(describe_state(instance, env))

        print("\n--- Recommended Action (for Sec. 5, Step 2) ---")
        print(describe_action(action))

        print("\n--- SHAP Global Ranking (for Sec. 5, Step 3) ---")
        ranking, shap_ms = run_shap_global(agent.actor, states, n_samples=80)
        for name, val in ranking:
            print(f"  {FRIENDLY_NAMES.get(name, name):22s} mean|SHAP| = {val:.5f}")
        print(f"  (SHAP computed in {shap_ms:.1f} ms)")

        print("\n--- LIME Local Explanation (for Sec. 5, Step 4) ---")
        lime_out, lime_ms = run_lime_local(agent.actor, states, instance)
        total_weight = sum(abs(w) for _, w in lime_out)
        for feat, weight in lime_out:
            pct = 100 * abs(weight) / total_weight if total_weight > 0 else 0
            print(f"  {feat:30s} weight={weight:+.5f} ({pct:.0f}% of total influence)")
        print(f"  (LIME computed in {lime_ms:.1f} ms)")

        print("\n--- Plain-language summary sentence (ready to read aloud) ---")
        top2 = ranking[:2]
        print(f"  \"For {archetype['label'].lower()} buildings, the system generally weighs "
              f"{FRIENDLY_NAMES.get(top2[0][0], top2[0][0])} and "
              f"{FRIENDLY_NAMES.get(top2[1][0], top2[1][0])} most heavily. "
              f"For this specific decision, the top factor was "
              f"{lime_out[0][0]}.\"")


if __name__ == "__main__":
    run()

"""
diagnose_office_actor.py
Quick diagnostic: checks whether the office trained actor's output actually
varies with the input state, or has collapsed to a constant action
(explaining the all-zero SHAP/LIME values observed in
extract_s5_demo_decisions.py's output for the office archetype).
"""
import numpy as np
import torch
from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent

env = BuildingEnergyEnv(data_path="data/office_archetype_15min.csv", seed=7)
agent = DDPGAgent(env.state_dim, env.action_dim)
agent.actor.load_state_dict(torch.load("trained_actor_office.pt"))

print("=" * 70)
print("Checking office actor's output variance across different states")
print("=" * 70)

s = env.reset()
actions = []
states_sample = []
for i in range(50):
    a = agent.act(s, explore=False)
    actions.append(a)
    states_sample.append(s.copy())
    s, r, done, info = env.step(a)
    if done:
        s = env.reset()

actions = np.array(actions)
print(f"\nAction dimension 0 (HVAC): mean={actions[:,0].mean():.4f}, std={actions[:,0].std():.6f}")
print(f"Action dimension 1 (ESS):  mean={actions[:,1].mean():.4f}, std={actions[:,1].std():.6f}")

print("\nFirst 10 raw actions (HVAC, ESS):")
for i in range(10):
    print(f"  {actions[i]}")

if actions[:, 1].std() < 1e-4:
    print("\n[DIAGNOSIS] The ESS (battery) action has collapsed to a near-constant")
    print("value regardless of input state - this IS a policy collapse, confirming")
    print("the all-zero SHAP/LIME values are a genuine finding, not a code bug.")
    print("Likely cause: insufficient/unbalanced training exposure for this action")
    print("dimension, or a local optimum where 'always discharge' was locally optimal")
    print("during this particular training run.")
else:
    print("\n[DIAGNOSIS] The ESS action DOES vary across states - the all-zero SHAP/LIME")
    print("result may instead be specific to the one instance tested, or a numerical")
    print("issue in the explainer's perturbation range for this actor. Try a different")
    print("state instance or increase SHAP's n_samples.")

print("\nAction dimension 0 (HVAC) - does it vary?")
if actions[:, 0].std() < 1e-4:
    print("  HVAC action ALSO appears collapsed/constant.")
else:
    print(f"  HVAC action varies normally (std={actions[:,0].std():.6f}).")

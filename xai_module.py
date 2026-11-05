"""
xai_module.py
Real-time explainability (Section 4.7, Contribution C3): SHAP global
attribution + LIME local explanation applied to the live DDPG actor.
"""
import numpy as np
import torch
import time

STATE_NAMES = ["base_load", "hvac_demand", "pv_gen", "price", "t_out", "soc", "hour_frac"]


def actor_predict_fn(actor):
    """Wraps the PyTorch actor as a numpy-in/numpy-out function for SHAP/LIME,
    returning only the ESS-rate action dimension (index 1) as the scalar
    output to explain (HVAC modulation, index 0, could equally be explained).
    """
    def f(x):
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32)
            out = actor(t).numpy()
        return out[:, 1]
    return f


def run_shap_global(actor, background_states, n_samples=150):
    import shap
    f = actor_predict_fn(actor)
    bg = background_states[np.random.choice(len(background_states),
                                             size=min(n_samples, len(background_states)),
                                             replace=False)]
    t0 = time.time()
    explainer = shap.KernelExplainer(f, bg)
    shap_values = explainer.shap_values(bg[:20], nsamples=100, silent=True)
    elapsed_ms = (time.time() - t0) * 1000 / 20  # per-instance average
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(STATE_NAMES, mean_abs), key=lambda x: -x[1])
    return ranking, elapsed_ms


def run_lime_local(actor, background_states, instance):
    from lime.lime_tabular import LimeTabularExplainer
    f = actor_predict_fn(actor)
    explainer = LimeTabularExplainer(
        background_states, feature_names=STATE_NAMES, mode="regression", verbose=False)
    t0 = time.time()
    exp = explainer.explain_instance(instance, f, num_features=len(STATE_NAMES), num_samples=200)
    elapsed_ms = (time.time() - t0) * 1000
    return exp.as_list(), elapsed_ms


if __name__ == "__main__":
    from environment import BuildingEnergyEnv
    from ddpg_agent import DDPGAgent

    env = BuildingEnergyEnv(seed=1)
    agent = DDPGAgent(env.state_dim, env.action_dim)

    # collect a small pool of real states from the environment to use as
    # SHAP/LIME background + instance to explain
    states = []
    s = env.reset()
    for _ in range(120):
        states.append(s)
        s, r, done, info = env.step(agent.act(s, explore=False))
        if done:
            s = env.reset()
    states = np.array(states, dtype=np.float32)

    print("Running SHAP global attribution (untrained actor, code-verification only)...")
    ranking, shap_ms = run_shap_global(agent.actor, states, n_samples=60)
    print(f"SHAP per-instance latency: {shap_ms:.2f} ms")
    for name, val in ranking:
        print(f"  {name:12s} mean|SHAP| = {val:.5f}")

    print("\nRunning LIME local explanation for one instance...")
    lime_out, lime_ms = run_lime_local(agent.actor, states, states[0])
    print(f"LIME latency: {lime_ms:.2f} ms")
    for feat, weight in lime_out:
        print(f"  {feat}: {weight:.5f}")

    print("\nNOTE: latencies and attributions above are from an UNTRAINED agent")
    print("on ONE archetype - code-verification only, not thesis results.")

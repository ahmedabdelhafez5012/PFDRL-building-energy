"""
multi_seed_h2_test.py
Multi-seed statistical protocol (Section 5.4) for Hypothesis H2: "there
exists a personalisation coefficient mu that meaningfully improves upon
non-personalised (fully local, mu=1.0) fine-tuning."

Reuses each archetype's per-seed Stage-1 model (trained via
main_real_federation.py; a fresh model is trained per seed here directly,
matching multi_seed_h4_test.py's self-contained approach so this script
does not depend on saved .pt files from a specific prior run) and compares
Stage-2 fine-tuning at mu = mu_hat* (the archetype's empirically-best value
from Appendix C.7 / scenario_s3_mu_sweep.py) against mu = 1.0 (fully local,
i.e. no personalisation blending) across N_SEEDS independent seeds.
"""
import numpy as np
import torch
from scipy.stats import wilcoxon

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer

N_SEEDS = 8
ROUNDS_PER_SEED = 15
STEPS_PER_ROUND = 3000
WARMUP_STEPS = 800
FINE_TUNE_STEPS = 2000
BATCH_SIZE = 128
EPISODE_LEN = 96
EVAL_EPISODES = 10

# Empirical mu_hat* per archetype, from the N=1 Scenario S3 result (Appendix C.7).
MU_HAT_STAR = {
    "residential": 0.80,
    "office": 0.00,
    "commercial": 0.01,
    "mixed-use": 0.01,
}

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv", "sqft": 2300},
    {"name": "office", "path": "data/office_archetype_15min.csv", "sqft": 45240},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv", "sqft": 45224},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv", "sqft": 262156},
]


def make_env(path, seed):
    return BuildingEnergyEnv(data_path=path, episode_len=EPISODE_LEN, seed=seed)


def local_train_round(agent, env, buf, steps, warmup=0):
    s = env.reset()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=env.action_dim) if i < warmup else agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE))
        if done:
            s = env.reset()


def evaluate(agent, path, seed):
    env = make_env(path, seed=seed)
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


def train_stage1(archetype, seed):
    """A lightweight, single-client Stage-1 style pretrain (no federation
    partners needed here - only the resulting critic/actor pair matters as
    the Stage-2 starting point, consistent with using each archetype's own
    saved Stage-1 model in Appendix C.4-C.7)."""
    env = make_env(archetype["path"], seed=seed)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                       noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
    buf = ReplayBuffer()
    for rnd in range(ROUNDS_PER_SEED):
        local_train_round(agent, env, buf, STEPS_PER_ROUND, warmup=WARMUP_STEPS if rnd == 0 else 0)
    return agent


def fine_tune(archetype, base_agent, mu, seed):
    import copy
    env = make_env(archetype["path"], seed=seed + 1000)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.1, noise_decay_steps=FINE_TUNE_STEPS)
    agent.actor.load_state_dict(base_agent.actor.state_dict())
    agent.critic.load_state_dict(base_agent.critic.state_dict())
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    global_critic_state = copy.deepcopy(base_agent.critic.state_dict())

    buf = ReplayBuffer()
    s = env.reset()
    for i in range(FINE_TUNE_STEPS):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > 128:
            agent.train_step(buf.sample(128), personalisation_target=global_critic_state, mu=mu)
        if done:
            s = env.reset()
    return agent


def run():
    print("=" * 78)
    print("Multi-seed H2 test: personalised (mu_hat*) vs non-personalised (mu=1.0)")
    print(f"N_SEEDS={N_SEEDS}")
    print("=" * 78)

    personalised_results = {a["name"]: [] for a in ARCHETYPES}
    nonpersonalised_results = {a["name"]: [] for a in ARCHETYPES}

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(100 + seed)
        np.random.seed(100 + seed)
        for archetype in ARCHETYPES:
            base_agent = train_stage1(archetype, seed)
            mu_star = MU_HAT_STAR[archetype["name"]]

            personalised_agent = fine_tune(archetype, base_agent, mu_star, seed)
            nonpersonalised_agent = fine_tune(archetype, base_agent, 1.0, seed)

            p_cost = evaluate(personalised_agent, archetype["path"], seed=9000 + seed)
            np_cost = evaluate(nonpersonalised_agent, archetype["path"], seed=9000 + seed)

            personalised_results[archetype["name"]].append(p_cost)
            nonpersonalised_results[archetype["name"]].append(np_cost)
            print(f"  {archetype['name']:12s} | personalised(mu={mu_star:.2f})={p_cost:10.3f} "
                  f"| non-personalised(mu=1.0)={np_cost:10.3f}")

    print("\n" + "=" * 78)
    print("WILCOXON SIGNED-RANK TEST RESULTS (H2: personalised < non-personalised)")
    print("=" * 78)
    for name in personalised_results:
        p_vals = np.array(personalised_results[name])
        np_vals = np.array(nonpersonalised_results[name])
        try:
            stat, p = wilcoxon(p_vals, np_vals)
        except ValueError as e:
            print(f"  {name:12s} | Wilcoxon failed ({e})")
            continue
        sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        direction = "personalised lower (supports H2)" if p_vals.mean() < np_vals.mean() else "non-personalised lower (contradicts H2)"
        print(f"  {name:12s} | personalised={p_vals.mean():.3f}(+/-{p_vals.std():.3f}) | "
              f"non-pers.={np_vals.mean():.3f}(+/-{np_vals.std():.3f}) | p={p:.4f} | {sig} | {direction}")

    print("\nRaw per-seed values:")
    print("Personalised:", personalised_results)
    print("Non-personalised:", nonpersonalised_results)


if __name__ == "__main__":
    run()

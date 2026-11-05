"""
scenario_s3_mu_sweep.py
Scenario S3 (Section 5.5): sweeps the personalisation coefficient mu across
{0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0} (Appendix A.4) for each of the four
real archetypes, using the Stage-1 trained models saved by
main_real_federation.py (trained_actor_<archetype>.pt / trained_critic_
<archetype>.pt) as the starting point - NOT retraining from scratch.

For each mu, a Stage-2 fine-tune runs: the local critic loss is blended with
the distance to the (frozen) Stage-1 global critic, per Eq. 4.6:
    L' = mu * L_local + (1 - mu) * D(Q_local, Q_global)
mu=1 means "ignore the global critic, fully local" and mu=0 means "fully
match the global critic" - the opposite convention from FedProx's mu, but
consistent with ddpg_agent.py's existing train_step(..., mu=...) parameter
and Eq. 4.6 as implemented (see ddpg_agent.py docstring).

Output: for each archetype, the mu that minimises held-out evaluation cost
is the empirical mu-hat* to compare against the theoretical mu* derived in
Appendix A.
"""
import copy
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer

MUS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
FINE_TUNE_STEPS = 3000     # Stage-2 fine-tune budget per mu value
EVAL_EPISODES = 15
EPISODE_LEN = 96
EVAL_SEED_OFFSET = 9000

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv", "sqft": 2300},
    {"name": "office", "path": "data/office_archetype_15min.csv", "sqft": 45240},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv", "sqft": 45224},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv", "sqft": 262156},
]


def make_env(path, seed):
    return BuildingEnergyEnv(data_path=path, episode_len=EPISODE_LEN, seed=seed)


def load_stage1_agent(archetype_name, state_dim, action_dim):
    agent = DDPGAgent(state_dim, action_dim, noise_sigma=0.1, noise_decay_steps=FINE_TUNE_STEPS)
    agent.actor.load_state_dict(torch.load(f"trained_actor_{archetype_name}.pt"))
    agent.critic.load_state_dict(torch.load(f"trained_critic_{archetype_name}.pt"))
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())
    return agent


def fine_tune_with_mu(archetype, mu, global_critic_state, train_seed=42):
    env = make_env(archetype["path"], seed=train_seed)
    agent = load_stage1_agent(archetype["name"], env.state_dim, env.action_dim)
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
    return float(np.mean(costs)), float(np.std(costs))


def run():
    print("=" * 78)
    print("Scenario S3 - mu-sweep, starting from Stage-1 trained models")
    print(f"mu values: {MUS}")
    print(f"Fine-tune budget per mu: {FINE_TUNE_STEPS} steps")
    print("=" * 78)

    results = {a["name"]: [] for a in ARCHETYPES}

    for archetype in ARCHETYPES:
        print(f"\n--- {archetype['name'].upper()} ---")
        env0 = make_env(archetype["path"], seed=0)
        # Stage-1 global critic = this archetype's own saved critic (proxy for
        # the federated global critic, since only per-client copies were saved)
        global_critic_state = torch.load(f"trained_critic_{archetype['name']}.pt")

        for mu in MUS:
            agent = fine_tune_with_mu(archetype, mu, global_critic_state)
            mean_cost, std_cost = evaluate(agent, archetype["path"], seed=EVAL_SEED_OFFSET)
            norm_cost = mean_cost / (archetype["sqft"] / 1000.0)
            results[archetype["name"]].append((mu, mean_cost, std_cost, norm_cost))
            print(f"  mu={mu:.2f} | mean cost/ep = {mean_cost:10.3f} (std {std_cost:.3f}) "
                  f"| cost/1000sqft = {norm_cost:8.4f}")

    print("\n" + "=" * 78)
    print("SUMMARY: empirical mu-hat* (mu minimising held-out cost) per archetype")
    print("=" * 78)
    for name, rows in results.items():
        best = min(rows, key=lambda r: r[1])
        print(f"  {name:12s} | mu_hat* = {best[0]:.2f} | mean cost/ep = {best[1]:.3f}")

    return results


if __name__ == "__main__":
    torch.manual_seed(1)
    np.random.seed(1)
    run()

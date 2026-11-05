"""
scenario_s4_epsilon_sweep.py
Scenario S4 (Section 5.5): sweeps the differential-privacy budget epsilon
across {None (no DP / infinite epsilon), 8, 4, 1} to trace the privacy-
utility frontier (tighter epsilon = more noise = more privacy, at a cost to
training quality). Unlike S3, this genuinely requires re-running Stage-1
federation from scratch for each epsilon (the DP noise is injected into the
training signal itself, not a post-hoc fine-tune), so this script reuses
main_real_federation.py's federation loop with epsilon as a parameter.

To keep total runtime reasonable, this uses a reduced per-epsilon round
budget by default (adjust EPS_ROUNDS/EPS_STEPS for the full campaign).
"""
import copy
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import fedprox_aggregate, add_gaussian_dp_noise, communication_cost_mb

EPSILONS = [None, 8.0, 4.0, 1.0]   # None = no DP noise added (infinite epsilon)
DP_DELTA = 1e-5
EPS_ROUNDS = 50          # FULL schedule (matches Section 4.4's Stage-1 budget)
EPS_STEPS_PER_ROUND = 5000
WARMUP_STEPS = 2000
BATCH_SIZE = 128
MU_FEDPROX = 0.05
EPISODE_LEN = 96
EVAL_EPISODES = 15
EVAL_SEED_OFFSET = 9500

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
    return float(np.mean(costs)), float(np.std(costs))


def run_federation_with_epsilon(epsilon, seed=0):
    envs = [make_env(a["path"], seed=i * 11 + seed) for i, a in enumerate(ARCHETYPES)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim, noise_sigma=0.4,
                         noise_decay_steps=int(EPS_ROUNDS * EPS_STEPS_PER_ROUND * 0.6))
              for _ in ARCHETYPES]
    bufs = [ReplayBuffer() for _ in ARCHETYPES]
    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)
    total_comm_mb = 0.0

    for rnd in range(1, EPS_ROUNDS + 1):
        client_critics = []
        for i in range(len(ARCHETYPES)):
            warmup = WARMUP_STEPS if rnd == 1 else 0
            local_train_round(agents[i], envs[i], bufs[i], EPS_STEPS_PER_ROUND, warmup=warmup)
            if epsilon is None:
                sd = copy.deepcopy(agents[i].critic.state_dict())
            else:
                sd, _ = add_gaussian_dp_noise(agents[i].critic.state_dict(), epsilon=epsilon, delta=DP_DELTA)
            client_critics.append(sd)
            total_comm_mb += communication_cost_mb(sd)

        new_global = fedprox_aggregate(client_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        global_critic.load_state_dict(new_global)
        for i in range(len(ARCHETYPES)):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

    eval_results = {}
    for i, a in enumerate(ARCHETYPES):
        mean_cost, std_cost = evaluate(agents[i], a["path"], seed=EVAL_SEED_OFFSET + i)
        eval_results[a["name"]] = (mean_cost, std_cost)
    return eval_results, total_comm_mb


def run():
    print("=" * 78)
    print("Scenario S4 - epsilon-sweep (privacy-utility frontier)")
    print(f"Epsilon values: {EPSILONS} (None = no DP / infinite epsilon)")
    print(f"Per-epsilon budget: {EPS_ROUNDS} rounds x {EPS_STEPS_PER_ROUND} steps")
    print("=" * 78)

    frontier = {a["name"]: [] for a in ARCHETYPES}
    for eps in EPSILONS:
        label = "no-DP (inf)" if eps is None else f"eps={eps}"
        print(f"\n--- {label} ---")
        eval_results, comm_mb = run_federation_with_epsilon(eps)
        for name, (mean_cost, std_cost) in eval_results.items():
            frontier[name].append((eps, mean_cost, std_cost))
            print(f"  {name:12s} | mean cost/ep = {mean_cost:10.3f} (std {std_cost:.3f})")
        print(f"  (comm cost this run: {comm_mb:.4f} MB)")

    print("\n" + "=" * 78)
    print("PRIVACY-UTILITY FRONTIER SUMMARY")
    print("=" * 78)
    for name, rows in frontier.items():
        print(f"\n{name}:")
        for eps, cost, std in rows:
            label = "no-DP" if eps is None else f"eps={eps}"
            print(f"  {label:10s} -> mean cost/ep = {cost:.3f} (std {std:.3f})")

    return frontier


if __name__ == "__main__":
    torch.manual_seed(2)
    np.random.seed(2)
    run()

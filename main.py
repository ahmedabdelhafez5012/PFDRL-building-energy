"""
main.py
End-to-end INTEGRATION / CODE-VERIFICATION dry run of the PFDRL Stage-1
pipeline (local DDPG training -> DP noise -> federated aggregation).

*** THIS IS NOT A THESIS EXPERIMENT. ***
Only ONE real archetype dataset (residential, HomeC.csv) is available so far.
To exercise the federation mechanics end-to-end without more data, this script
splits the single real dataset into disjoint time windows and treats each
window as a separate simulated "client" purely to verify the code runs
correctly. Any numbers printed below describe CODE BEHAVIOUR, not empirical
findings, and must never be copied into the thesis as results.
"""
import copy
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import (
    fedprox_aggregate, add_gaussian_dp_noise, communication_cost_mb,
)

N_CLIENTS = 3
N_ROUNDS = 5
LOCAL_STEPS = 300
BATCH_SIZE = 32
MU_FEDPROX = 0.05
DP_EPSILON = 8.0
DP_DELTA = 1e-5


def make_client_env(client_id):
    # Different seeds -> different random start windows within the SAME
    # real dataset, standing in for "different buildings" for code-testing.
    return BuildingEnergyEnv(seed=client_id * 17 + 1)


def local_train(agent, env, buf, steps):
    s = env.reset()
    ep_reward = 0.0
    for _ in range(steps):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        ep_reward += r
        if len(buf) > BATCH_SIZE:
            agent.train_step(buf.sample(BATCH_SIZE), mu=1.0)  # local-only loss during Stage 1
        if done:
            s = env.reset()
    return ep_reward


def run():
    print("=" * 70)
    print("PFDRL Stage-1 CODE-VERIFICATION DRY RUN (not a thesis result)")
    print(f"Clients simulated from ONE real dataset: {N_CLIENTS}")
    print("=" * 70)

    envs = [make_client_env(i) for i in range(N_CLIENTS)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim) for _ in range(N_CLIENTS)]
    bufs = [ReplayBuffer() for _ in range(N_CLIENTS)]

    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)
    total_comm_mb = 0.0

    for rnd in range(1, N_ROUNDS + 1):
        round_rewards = []
        noised_critics = []
        for i in range(N_CLIENTS):
            r_ep = local_train(agents[i], envs[i], bufs[i], LOCAL_STEPS)
            round_rewards.append(r_ep)

            noised_sd, sigma = add_gaussian_dp_noise(
                agents[i].critic.state_dict(), epsilon=DP_EPSILON, delta=DP_DELTA)
            noised_critics.append(noised_sd)
            total_comm_mb += communication_cost_mb(noised_sd)

        new_global = fedprox_aggregate(noised_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        global_critic.load_state_dict(new_global)
        for i in range(N_CLIENTS):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

        print(f"Round {rnd}/{N_ROUNDS} | per-client episodic reward: "
              f"{[f'{r:.2f}' for r in round_rewards]} | "
              f"cumulative comm cost: {total_comm_mb:.4f} MB")

    print("-" * 70)
    print("Dry run complete. Code executed without errors across:")
    print(f"  - {N_CLIENTS} simulated clients x {N_ROUNDS} federated rounds "
          f"x {LOCAL_STEPS} local steps each")
    print(f"  - DP mechanism (eps={DP_EPSILON}, delta={DP_DELTA}), sigma~{sigma:.4f}")
    print("  - FedProx aggregation (mu=0.05)")
    print("Next step for REAL results: replace simulated multi-client split")
    print("with genuine per-archetype datasets (Pecan Street EV/solar homes,")
    print("BDG2/ICPE office data, additional Kaggle commercial data).")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    run()

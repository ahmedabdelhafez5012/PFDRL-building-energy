"""
main_real_federation.py
The FIRST GENUINE Stage-1 federation run across ALL FOUR REAL, DIFFERENT
building archetypes required by the thesis (Contribution C1):
  Client 1: Residential (HomeC.csv, Kaggle Smart Home dataset)
  Client 2: Office (Eagle_office_Katheleen, BDG2)
  Client 3: Commercial/Retail (Fox_retail_Manie, BDG2, with real chilled-water
            sub-metering used to derive a time-varying HVAC share)
  Client 4: Mixed-use/Lodging (Hog_lodging_Shanti, BDG2, same real chilled-water
            HVAC-share methodology)

This is a real step up from main.py's earlier dry run, which only split ONE
dataset into synthetic time-window "clients", and from the 2-archetype run
before this, which lacked Commercial and Mixed-use coverage.

Still honest limitations to keep in mind: training budget is well short of
the thesis's full 50-round schedule (Section 4.4) to fit this session's
compute constraints; office/commercial/mixed-use lack a matched real weather
signal (synthetic seasonal placeholder used instead, see the respective
data_loader_*.py docstrings) and have zero solar generation (no PV data
available for these specific BDG2 buildings); and per-archetype cost
magnitudes are not yet normalised for direct cross-archetype comparison.
"""
import copy
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import (
    fedprox_aggregate, add_gaussian_dp_noise, communication_cost_mb,
)

N_ROUNDS = 6
LOCAL_STEPS_PER_ROUND = 2000
WARMUP_STEPS = 800          # only applied in round 1
BATCH_SIZE = 128
MU_FEDPROX = 0.05
DP_EPSILON = 8.0
DP_DELTA = 1e-5
EPISODE_LEN = 96

CLIENTS = [
    {"name": "Residential (HomeC.csv)", "path": "data/residential_archetype_15min.csv",
     "sqft": 2300, "sqft_source": "US Census average single-family home (HomeC.csv has no metadata; documented estimate)"},
    {"name": "Office (BDG2 Eagle_office_Katheleen)", "path": "data/office_archetype_15min.csv",
     "sqft": 45240, "sqft_source": "BDG2 metadata.csv (real)"},
    {"name": "Commercial (BDG2 Fox_retail_Manie)", "path": "data/commercial_archetype_15min.csv",
     "sqft": 45224, "sqft_source": "BDG2 metadata.csv (real)"},
    {"name": "Mixed-use (BDG2 Hog_lodging_Shanti)", "path": "data/mixeduse_archetype_15min.csv",
     "sqft": 262156, "sqft_source": "BDG2 metadata.csv (real)"},
]


EVAL_SEED_OFFSET = 9000   # held-out seeds, disjoint from training seeds (i*11+3)
EVAL_EPISODES = 15


def evaluate_agent(agent, data_path, seed):
    """Clean held-out evaluation: NO exploration noise (explore=False), a
    seed disjoint from training, averaged over multiple episodes. This is
    the metric that should be reported/compared statistically - NOT the
    noisy training-episode costs printed during the federation loop above,
    which include exploration and single-episode variance."""
    env = make_env(data_path, seed=seed)
    costs, discomforts = [], []
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        ep_cost, ep_disc = 0.0, 0.0
        done = False
        while not done:
            a = agent.act(s, explore=False)
            s, r, done, info = env.step(a)
            ep_cost += info["cost"]
            ep_disc += info["discomfort"]
        costs.append(ep_cost)
        discomforts.append(ep_disc)
    return {
        "mean_cost": float(np.mean(costs)), "std_cost": float(np.std(costs)),
        "mean_discomfort": float(np.mean(discomforts)), "n_episodes": EVAL_EPISODES,
    }


def make_env(path, seed):
    return BuildingEnergyEnv(data_path=path, episode_len=EPISODE_LEN, seed=seed)


def local_train_round(agent, env, buf, steps, warmup=0):
    s = env.reset()
    ep_reward, ep_cost = 0.0, 0.0
    for i in range(steps):
        if i < warmup:
            a = np.random.uniform(-1, 1, size=env.action_dim)
        else:
            a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        ep_reward += r
        ep_cost += info["cost"]
        if len(buf) > BATCH_SIZE and i >= warmup:
            agent.train_step(buf.sample(BATCH_SIZE))
        if done:
            s = env.reset()
    return ep_reward, ep_cost


def run():
    print("=" * 74)
    print(f"PFDRL Stage-1 REAL federation: {len(CLIENTS)} genuine building archetypes")
    for c in CLIENTS:
        print(f"  - {c['name']}")
    print(f"Rounds: {N_ROUNDS} | local steps/round: {LOCAL_STEPS_PER_ROUND} | "
          f"FedProx mu={MU_FEDPROX} | DP (eps={DP_EPSILON}, delta={DP_DELTA})")
    print("=" * 74)

    envs = [make_env(c["path"], seed=i * 11 + 3) for i, c in enumerate(CLIENTS)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim,
                         noise_sigma=0.4, noise_decay_steps=int(N_ROUNDS * LOCAL_STEPS_PER_ROUND * 0.6))
              for _ in CLIENTS]
    bufs = [ReplayBuffer() for _ in CLIENTS]

    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)
    total_comm_mb = 0.0
    history = []

    for rnd in range(1, N_ROUNDS + 1):
        round_costs, round_rewards = [], []
        noised_critics = []
        for i, c in enumerate(CLIENTS):
            warmup = WARMUP_STEPS if rnd == 1 else 0
            ep_reward, ep_cost = local_train_round(
                agents[i], envs[i], bufs[i], LOCAL_STEPS_PER_ROUND, warmup=warmup)
            round_rewards.append(ep_reward)
            round_costs.append(ep_cost)

            noised_sd, sigma = add_gaussian_dp_noise(
                agents[i].critic.state_dict(), epsilon=DP_EPSILON, delta=DP_DELTA)
            noised_critics.append(noised_sd)
            total_comm_mb += communication_cost_mb(noised_sd)

        new_global = fedprox_aggregate(noised_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        global_critic.load_state_dict(new_global)
        for i in range(len(CLIENTS)):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

        history.append({"round": rnd, "costs": list(round_costs), "rewards": list(round_rewards)})
        cost_str = ", ".join(f"{CLIENTS[i]['name'].split(' ')[0]}={round_costs[i]:.2f}"
                              for i in range(len(CLIENTS)))
        print(f"Round {rnd}/{N_ROUNDS} | last-training-episode cost: {cost_str} | "
              f"cumulative comm: {total_comm_mb:.4f} MB")

    print()
    print("Normalised comparison (cost per 1000 sqft, using round-{}'s per-client "
          "cost / building floor area - this is the fair, comparable metric, "
          "NOT the raw costs above):".format(N_ROUNDS))
    for i, c in enumerate(CLIENTS):
        norm_cost = round_costs[i] / (c["sqft"] / 1000.0)
        print(f"  {c['name']:42s} | {c['sqft']:>7,} sqft ({c['sqft_source']}) "
              f"| cost/1000sqft = {norm_cost:8.3f}")

    print()
    print("=" * 74)
    print("CLEAN HELD-OUT EVALUATION (no exploration noise, disjoint seed,")
    print(f"{EVAL_EPISODES} episodes per archetype) - this is the metric to report,")
    print("not the noisy training-episode costs printed above.")
    print("=" * 74)
    eval_results = []
    for i, c in enumerate(CLIENTS):
        res = evaluate_agent(agents[i], c["path"], seed=EVAL_SEED_OFFSET + i)
        eval_results.append(res)
        norm_eval_cost = res["mean_cost"] / (c["sqft"] / 1000.0)
        print(f"  {c['name']:42s} | mean cost/ep = {res['mean_cost']:10.3f} "
              f"(std {res['std_cost']:.3f}) | mean discomfort = {res['mean_discomfort']:8.3f} "
              f"| cost/1000sqft = {norm_eval_cost:8.3f}")
        torch.save(agents[i].actor.state_dict(), f"trained_actor_{c['name'].split(' ')[0].lower()}.pt")
        torch.save(agents[i].critic.state_dict(), f"trained_critic_{c['name'].split(' ')[0].lower()}.pt")
    print()
    print("Trained actor/critic weights saved (trained_actor_<archetype>.pt,")
    print("trained_critic_<archetype>.pt) - reload these for Scenario S3/S4")
    print("fine-tuning or for real (non-random-actor) SHAP/LIME analysis,")
    print("instead of retraining from scratch.")

    print("-" * 74)
    print(f"REAL {len(CLIENTS)}-archetype Stage-1 federation complete.")
    print(f"Total communication cost across {N_ROUNDS} rounds x {len(CLIENTS)} clients: "
          f"{total_comm_mb:.4f} MB ({total_comm_mb / N_ROUNDS:.4f} MB/round average)")
    print(f"DP noise sigma (eps={DP_EPSILON}, delta={DP_DELTA}): {sigma:.4f}")
    print()
    print("Honest scope note: this now covers all 4 archetypes required by the")
    print("thesis (Contribution C1), but with a reduced round/step budget")
    print(f"({N_ROUNDS}x{LOCAL_STEPS_PER_ROUND}) vs. the thesis's full 50-round schedule")
    print("(Section 4.4). All 4 archetypes now use REAL measured weather (BDG2")
    print("weather.csv for office/commercial/mixed-use; HomeC.csv's own weather")
    print("columns for residential). Office/commercial/mixed-use still have zero")
    print("PV generation (no solar data available for these specific BDG2")
    print("buildings). Cost magnitudes ARE now normalised per 1,000 sqft using")
    print("real BDG2 floor areas (residential uses an estimated area). Held-out")
    print("evaluation results above are the low-variance, exploration-free")
    print("metric suitable for the statistical protocol in Section 5.4.")
    return history, global_critic, eval_results


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    run()

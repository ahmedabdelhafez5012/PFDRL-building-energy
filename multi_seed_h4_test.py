"""
multi_seed_h4_test.py
Multi-seed statistical protocol (Section 5.4) for Hypothesis H4: "FedProx
achieves the lowest MAE/cost among the four FL aggregation strategies."
This script tests the core comparison (FedProx vs FedAvg) across N_SEEDS
independent random seeds, then applies the pre-registered Wilcoxon
signed-rank test.

DESIGN NOTE on per-seed budget: a full 50-round schedule per seed (as used
for the single-seed results in Appendix C.4-C.8) would require N_SEEDS x 2
conditions x ~1 hour each - impractical for a first statistical pass. This
script uses a REDUCED per-seed budget (15 rounds x 3,000 steps) - a
documented, deliberate trade-off favouring more independent seeds (needed
for a valid signed-rank test, minimum ~5-6 pairs recommended) over per-seed
training depth. This is noted explicitly in the thesis as a methodological
choice, not hidden. Increase ROUNDS_PER_SEED for a more rigorous (but much
slower) version once initial significance is established.
"""
import copy
import numpy as np
import torch
from scipy.stats import wilcoxon

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import fedprox_aggregate, fedavg_aggregate, add_gaussian_dp_noise, communication_cost_mb

N_SEEDS = 8       # raised from 5: at N=5, Wilcoxon's minimum p-value (0.0625)
                  # cannot reach significance even with a perfect 5/5 direction;
                  # N=8 allows p~=0.0078 in the all-consistent case (see Appendix C.9)
SEED_START = 5    # set to 5 to run ONLY the 3 additional seeds (5,6,7) needed to
                  # extend an already-completed N=5 run, instead of redoing 0-4

# Results from the N=5 run already completed and reported in Appendix C.9.
# When SEED_START=5, these are combined automatically with the newly-run
# seeds 5-7 to give a full N=8 Wilcoxon test without redoing prior work.
PRIOR_FEDPROX = {
    "residential": [4.470080710788204, 3.871849300852247, 2.475729821404399, 2.0578099186761696, 3.4802932921344123],
    "office": [501.54101831345724, 640.1157509643156, 489.1518941374444, 568.3931648555855, 502.62336219135904],
    "commercial": [334.6196493049965, 319.96620382973936, 348.4328371781647, 350.79303136599145, 304.79333923972416],
    "mixed-use": [1323.6220906512035, 1141.4191047838124, 1679.092968714775, 1284.353861054194, 1178.2962847758622],
}
PRIOR_FEDAVG = {
    "residential": [4.693328153291356, 3.929015882761002, 2.086468709879823, 3.110687411831095, 3.714803008111587],
    "office": [525.495547445644, 586.5315666821783, 543.6176336045919, 574.9559670928984, 517.5382091312026],
    "commercial": [351.5514823803077, 325.1544907453946, 342.80265278102, 371.1094486433723, 325.3260776485605],
    "mixed-use": [1278.1545240965265, 1145.350436675535, 1571.9680625084889, 1388.6271194431135, 1246.0285031283152],
}
ROUNDS_PER_SEED = 15
STEPS_PER_ROUND = 3000
WARMUP_STEPS = 800
BATCH_SIZE = 128
MU_FEDPROX = 0.05
DP_EPSILON = 8.0
DP_DELTA = 1e-5
EPISODE_LEN = 96
EVAL_EPISODES = 10

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


def run_one_seed(strategy, seed):
    """strategy: 'fedprox' or 'fedavg'. Returns mean held-out cost per archetype."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = [make_env(a["path"], seed=i * 11 + seed) for i, a in enumerate(ARCHETYPES)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim, noise_sigma=0.4,
                         noise_decay_steps=int(ROUNDS_PER_SEED * STEPS_PER_ROUND * 0.6))
              for _ in ARCHETYPES]
    bufs = [ReplayBuffer() for _ in ARCHETYPES]
    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)

    for rnd in range(1, ROUNDS_PER_SEED + 1):
        client_critics = []
        for i in range(len(ARCHETYPES)):
            warmup = WARMUP_STEPS if rnd == 1 else 0
            local_train_round(agents[i], envs[i], bufs[i], STEPS_PER_ROUND, warmup=warmup)
            sd, _ = add_gaussian_dp_noise(agents[i].critic.state_dict(), epsilon=DP_EPSILON, delta=DP_DELTA)
            client_critics.append(sd)

        if strategy == "fedprox":
            new_global = fedprox_aggregate(client_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        else:
            new_global = fedavg_aggregate(client_critics)
        global_critic.load_state_dict(new_global)
        for i in range(len(ARCHETYPES)):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

    results = {}
    for i, a in enumerate(ARCHETYPES):
        results[a["name"]] = evaluate(agents[i], a["path"], seed=9000 + seed)
    return results


def run():
    print("=" * 78)
    print("Multi-seed H4 test: FedProx vs FedAvg")
    print(f"Seeds {SEED_START} to {N_SEEDS - 1}, budget/seed={ROUNDS_PER_SEED}x{STEPS_PER_ROUND} steps")
    if SEED_START > 0:
        print(f"NOTE: seeds 0-{SEED_START - 1} are assumed already completed in a prior run.")
        print("Their results (hardcoded as PRIOR_FEDPROX/PRIOR_FEDAVG) will be combined")
        print("automatically with this run's new seeds before the Wilcoxon test.")
    print("=" * 78)

    fedprox_results = {a["name"]: [] for a in ARCHETYPES}
    fedavg_results = {a["name"]: [] for a in ARCHETYPES}

    for seed in range(SEED_START, N_SEEDS):
        print(f"\n--- Seed {seed} ---")
        print("  Running FedProx...")
        fp = run_one_seed("fedprox", seed)
        print("  Running FedAvg...")
        fa = run_one_seed("fedavg", seed)
        for name in fedprox_results:
            fedprox_results[name].append(fp[name])
            fedavg_results[name].append(fa[name])
            print(f"    {name:12s} | FedProx={fp[name]:10.3f} | FedAvg={fa[name]:10.3f}")

    if SEED_START > 0:
        print("\nCombining newly-run seeds with the prior N=5 results...")
        for name in fedprox_results:
            fedprox_results[name] = PRIOR_FEDPROX[name] + fedprox_results[name]
            fedavg_results[name] = PRIOR_FEDAVG[name] + fedavg_results[name]

    print("\n" + "=" * 78)
    print("WILCOXON SIGNED-RANK TEST RESULTS (H4: FedProx < FedAvg in cost)")
    print("=" * 78)
    for name in fedprox_results:
        fp_vals = np.array(fedprox_results[name])
        fa_vals = np.array(fedavg_results[name])
        try:
            stat, p = wilcoxon(fp_vals, fa_vals)
        except ValueError as e:
            stat, p = None, None
            print(f"  {name:12s} | Wilcoxon failed ({e}) - likely identical values across seeds")
            continue
        fp_mean, fp_std = fp_vals.mean(), fp_vals.std()
        fa_mean, fa_std = fa_vals.mean(), fa_vals.std()
        sig = "SIGNIFICANT (p<0.05)" if p < 0.05 else "not significant"
        direction = "FedProx lower (supports H4)" if fp_mean < fa_mean else "FedAvg lower (contradicts H4)"
        print(f"  {name:12s} | FedProx={fp_mean:.3f}(+/-{fp_std:.3f}) | FedAvg={fa_mean:.3f}(+/-{fa_std:.3f}) "
              f"| p={p:.4f} | {sig} | {direction}")

    print("\nRaw per-seed values (for the thesis appendix / reproducibility):")
    print("FedProx:", fedprox_results)
    print("FedAvg:", fedavg_results)


if __name__ == "__main__":
    run()

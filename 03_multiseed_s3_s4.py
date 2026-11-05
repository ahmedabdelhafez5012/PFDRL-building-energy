"""
03_multiseed_s3_s4.py  (v2 - REAL implementation, built directly against
scenario_s3_mu_sweep.py and scenario_s4_epsilon_sweep.py)
==============================================================================
Repeats S3 (mu-sweep) and S4 (epsilon-sweep) across multiple independent
seeds, so the headline Figures C.4/C.5 report mean +/- std instead of a
single-seed point estimate.

KEY CHANGES vs the original single-seed scripts:
  S3 - the original script loads a fixed, pre-saved Stage-1 checkpoint
       (trained_actor_<archetype>.pt) and fine-tunes it once per mu. For a
       genuine multi-seed test, THIS version trains a FRESH Stage-1 model
       per seed (same pattern as multi_seed_h2_test.py's train_stage1),
       then sweeps mu from that seed's own checkpoint -- so different seeds
       are genuinely independent, not fine-tunes of the same starting point.
  S4 - inherently trains all four archetypes together per (epsilon, seed)
       call already (federation), so there is no per-archetype redundancy
       to fix here (same efficiency lesson as B3/H4, already satisfied).

BUDGET NOTE: S4's original script defaults to the FULL 50x5000 schedule.
At multiple seeds x 4 epsilon values, that is very expensive (many hours
per seed). This version defaults to the SAME REDUCED budget used
everywhere else in the enhancement campaign (15x3000) for feasibility;
override with --s4_rounds/--s4_steps if you want the full schedule and
have the time.

N_SEEDS default is 5, not 20-30: the goal here is a stable mean +/- CI for
the headline figures, not a Bonferroni-survivable p-value (see
00_power_analysis.py's reasoning).

Run:
    python 03_multiseed_s3_s4.py --scenario s3 --n_seeds 5
    python 03_multiseed_s3_s4.py --scenario s4 --n_seeds 5
    python 03_multiseed_s3_s4.py --scenario both --n_seeds 5
"""
import argparse
import json
import os
import numpy as np
import torch

from environment import BuildingEnergyEnv
from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
from federated_learning import fedprox_aggregate, add_gaussian_dp_noise, communication_cost_mb

# --- S3 settings ---
MUS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
STAGE1_ROUNDS = 15
STAGE1_STEPS_PER_ROUND = 3000
STAGE1_WARMUP = 800
S3_FINE_TUNE_STEPS = 3000

# --- S4 settings ---
EPSILONS = [None, 8.0, 4.0, 1.0]
DP_DELTA = 1e-5
MU_FEDPROX = 0.05
S4_WARMUP = 800

BATCH_SIZE = 128
EPISODE_LEN = 96
EVAL_EPISODES = 15
EVAL_SEED_OFFSET = 9000

ARCHETYPES = [
    {"name": "residential", "path": "data/residential_archetype_15min.csv", "sqft": 2300},
    {"name": "office", "path": "data/office_archetype_15min.csv", "sqft": 45240},
    {"name": "commercial", "path": "data/commercial_archetype_15min.csv", "sqft": 45224},
    {"name": "mixed-use", "path": "data/mixeduse_archetype_15min.csv", "sqft": 262156},
]
ARCH_NAMES = [a["name"] for a in ARCHETYPES]


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


# ---------------------------------------------------------------------------
# S3 - mu-sweep, fresh Stage-1 per seed
# ---------------------------------------------------------------------------
def train_stage1(archetype, seed):
    env = make_env(archetype["path"], seed=seed)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.4,
                       noise_decay_steps=int(STAGE1_ROUNDS * STAGE1_STEPS_PER_ROUND * 0.6))
    buf = ReplayBuffer()
    for rnd in range(STAGE1_ROUNDS):
        local_train_round(agent, env, buf, STAGE1_STEPS_PER_ROUND, warmup=STAGE1_WARMUP if rnd == 0 else 0)
    return agent


def fine_tune_with_mu(archetype, base_agent, mu, global_critic_state, seed):
    env = make_env(archetype["path"], seed=seed + 2000)
    agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.1, noise_decay_steps=S3_FINE_TUNE_STEPS)
    agent.actor.load_state_dict(base_agent.actor.state_dict())
    agent.critic.load_state_dict(base_agent.critic.state_dict())
    agent.actor_target.load_state_dict(agent.actor.state_dict())
    agent.critic_target.load_state_dict(agent.critic.state_dict())

    buf = ReplayBuffer()
    s = env.reset()
    for i in range(S3_FINE_TUNE_STEPS):
        a = agent.act(s)
        s2, r, done, info = env.step(a)
        buf.push(s, a, r, s2, done)
        s = s2
        if len(buf) > 128:
            agent.train_step(buf.sample(128), personalisation_target=global_critic_state, mu=mu)
        if done:
            s = env.reset()
    return agent


def run_s3_one_seed(seed):
    """Returns {archetype: {mu: cost}} for this seed."""
    torch.manual_seed(300 + seed)
    np.random.seed(300 + seed)
    out = {}
    for a in ARCHETYPES:
        base_agent = train_stage1(a, seed)
        global_critic_state = base_agent.critic.state_dict()  # this seed's own Stage-1 critic
        out[a["name"]] = {}
        for mu in MUS:
            agent = fine_tune_with_mu(a, base_agent, mu, global_critic_state, seed)
            cost = evaluate(agent, a["path"], seed=EVAL_SEED_OFFSET + seed)
            out[a["name"]][str(mu)] = cost
    return out


# ---------------------------------------------------------------------------
# S4 - epsilon-sweep, all 4 archetypes trained together per (epsilon, seed)
# ---------------------------------------------------------------------------
def run_s4_one_seed_one_epsilon(epsilon, seed, rounds, steps_per_round):
    torch.manual_seed(400 + seed)
    np.random.seed(400 + seed)
    envs = [make_env(a["path"], seed=i * 11 + seed) for i, a in enumerate(ARCHETYPES)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim, noise_sigma=0.4,
                         noise_decay_steps=int(rounds * steps_per_round * 0.6))
              for _ in ARCHETYPES]
    bufs = [ReplayBuffer() for _ in ARCHETYPES]
    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)

    for rnd in range(1, rounds + 1):
        client_critics = []
        for i in range(len(ARCHETYPES)):
            warmup = S4_WARMUP if rnd == 1 else 0
            local_train_round(agents[i], envs[i], bufs[i], steps_per_round, warmup=warmup)
            if epsilon is None:
                sd = agents[i].critic.state_dict()
            else:
                sd, _ = add_gaussian_dp_noise(agents[i].critic.state_dict(), epsilon=epsilon, delta=DP_DELTA)
            client_critics.append(sd)
        new_global = fedprox_aggregate(client_critics, global_critic.state_dict(), mu=MU_FEDPROX)
        global_critic.load_state_dict(new_global)
        for i in range(len(ARCHETYPES)):
            agents[i].critic.load_state_dict(new_global)
            agents[i].critic_target.load_state_dict(new_global)

    return {a["name"]: evaluate(agents[i], a["path"], seed=9500 + seed) for i, a in enumerate(ARCHETYPES)}


def run_s4_one_seed(seed, rounds, steps_per_round):
    """Returns {archetype: {epsilon_label: cost}} for this seed."""
    out = {a["name"]: {} for a in ARCHETYPES}
    for eps in EPSILONS:
        label = "inf" if eps is None else str(eps)
        res = run_s4_one_seed_one_epsilon(eps, seed, rounds, steps_per_round)
        for name, cost in res.items():
            out[name][label] = cost
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_or_init_s3(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {a: {str(mu): [] for mu in MUS} for a in ARCH_NAMES}


def load_or_init_s4(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    labels = ["inf", "8.0", "4.0", "1.0"]
    return {a: {lab: [] for lab in labels} for a in ARCH_NAMES}


def run_s3(n_seeds, out_dir):
    path = os.path.join(out_dir, "s3_multiseed.json")
    results = load_or_init_s3(path)
    n_done = min(len(results[a][str(MUS[0])]) for a in ARCH_NAMES)
    for seed in range(n_done, n_seeds):
        print(f"[S3] seed {seed+1}/{n_seeds}")
        vals = run_s3_one_seed(seed)
        for a in ARCH_NAMES:
            for mu in MUS:
                results[a][str(mu)].append(vals[a][str(mu)])
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n[S3] Empirical mu_hat* per archetype (mean +/- std across seeds):")
    for a in ARCH_NAMES:
        means = {mu: np.mean(results[a][str(mu)]) for mu in MUS}
        stds = {mu: np.std(results[a][str(mu)]) for mu in MUS}
        best_mu = min(means, key=means.get)
        print(f"  {a:12s} mu_hat*={best_mu}  cost={means[best_mu]:.3f} (+/-{stds[best_mu]:.3f})  n={n_seeds}")


def run_s4(n_seeds, out_dir, rounds, steps_per_round):
    path = os.path.join(out_dir, "s4_multiseed.json")
    results = load_or_init_s4(path)
    n_done = min(len(results[a]["inf"]) for a in ARCH_NAMES)
    for seed in range(n_done, n_seeds):
        print(f"[S4] seed {seed+1}/{n_seeds}  (rounds={rounds}, steps/round={steps_per_round})")
        vals = run_s4_one_seed(seed, rounds, steps_per_round)
        for a in ARCH_NAMES:
            for lab in ["inf", "8.0", "4.0", "1.0"]:
                results[a][lab].append(vals[a][lab])
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    print("\n[S4] Privacy-utility frontier (mean +/- std across seeds):")
    print("Checks whether the non-monotonic finding survives averaging, or was single-seed noise:")
    for a in ARCH_NAMES:
        means = {lab: np.mean(results[a][lab]) for lab in ["inf", "8.0", "4.0", "1.0"]}
        no_dp = means["inf"]
        monotonic = all(means[lab] >= no_dp for lab in ["8.0", "4.0", "1.0"])
        print(f"  {a:12s} no-DP={no_dp:.3f}  eps8={means['8.0']:.3f}  eps4={means['4.0']:.3f}  "
              f"eps1={means['1.0']:.3f}  -> {'monotonic (classical)' if monotonic else 'NON-monotonic'}  n={n_seeds}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["s3", "s4", "both"], default="both")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--s4_rounds", type=int, default=STAGE1_ROUNDS,
                         help="Default matches the reduced budget used elsewhere (15). "
                              "Original single-seed script used 50 (full schedule).")
    parser.add_argument("--s4_steps", type=int, default=STAGE1_STEPS_PER_ROUND,
                         help="Default matches the reduced budget used elsewhere (3000). "
                              "Original single-seed script used 5000 (full schedule).")
    parser.add_argument("--out_dir", default="results/multiseed_s3_s4")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.scenario in ("s3", "both"):
        print("\n" + "=" * 70 + "\nSCENARIO S3 (mu-sweep)\n" + "=" * 70)
        run_s3(args.n_seeds, args.out_dir)
    if args.scenario in ("s4", "both"):
        print("\n" + "=" * 70 + "\nSCENARIO S4 (epsilon-sweep)\n" + "=" * 70)
        run_s4(args.n_seeds, args.out_dir, args.s4_rounds, args.s4_steps)


if __name__ == "__main__":
    main()

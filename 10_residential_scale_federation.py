"""
10_residential_scale_federation.py
=====================================
Two stages:
  STAGE 1 (--finalize): adds the exact same time-of-use price schedule used
    in residential_archetype_15min.csv (0.11 off-peak / 0.22 mid-peak / 0.42
    peak, by hour-of-day) to the 15 new Ausgrid-derived files, and reformats
    them to the EXACT column layout BuildingEnergyEnv already expects
    (index=timestamp, hvac_kw, base_kw, pv_kw, t_out_c, price) -- confirmed
    identical to residential_archetype_15min.csv, so NO changes to
    environment.py are needed; these files can be loaded by the exact same
    BuildingEnergyEnv class used everywhere else in this codebase.

  STAGE 2 (--run_federation): runs a FedProx federation across ALL 16
    residential clients (the original HomeC-derived one + the 15 new
    Ausgrid-derived ones) and reports the empirical mu_hat* -- directly
    testing Appendix A's bias-variance prediction that mu* should DECREASE
    (more benefit from the federated/global model) as the number of clients
    N increases. This is the core Phase 2 scale-effect test.

Run:
    python 10_residential_scale_federation.py --finalize
    python 10_residential_scale_federation.py --run_federation --n_clients 16
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

RESIDENTIAL_EXPANSION_DIR = "data/residential_expansion"
ORIGINAL_RESIDENTIAL = "data/residential_archetype_15min.csv"
FINALIZED_DIR = "data/residential_scale_clients"

ROUNDS = 15
STEPS_PER_ROUND = 3000
WARMUP = 800
BATCH_SIZE = 128
EPISODE_LEN = 96
EVAL_EPISODES = 15
MUS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]


def price_for_hour(hour):
    if hour in (17, 18, 19, 20):
        return 0.42
    if hour in (7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22):
        return 0.22
    return 0.11  # 0-6, 23


def finalize_files():
    os.makedirs(FINALIZED_DIR, exist_ok=True)

    # Client 0: the original file, copied as-is (already correct format).
    orig = pd.read_csv(ORIGINAL_RESIDENTIAL, index_col=0)
    orig.to_csv(os.path.join(FINALIZED_DIR, "residential_client0_15min.csv"))
    print(f"Client 0 (original): {len(orig)} rows -> {FINALIZED_DIR}/residential_client0_15min.csv")

    # Clients 1-15: the new Ausgrid-derived files, reformatted.
    raw_files = sorted(glob.glob(os.path.join(RESIDENTIAL_EXPANSION_DIR, "residential_ausgrid_customer*_15min.csv")))
    for i, path in enumerate(raw_files, start=1):
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df["price"] = df["timestamp"].dt.hour.map(price_for_hour)
        df = df.set_index("timestamp")[["hvac_kw", "base_kw", "pv_kw", "t_out_c", "price"]]
        out_path = os.path.join(FINALIZED_DIR, f"residential_client{i}_15min.csv")
        df.to_csv(out_path)
        print(f"Client {i}: {len(df)} rows -> {out_path}")

    print(f"\nDone. {len(raw_files) + 1} total residential clients ready in {FINALIZED_DIR}/")


# ---------------------------------------------------------------------------
# Stage 2: scale federation test
# ---------------------------------------------------------------------------
def _client_sort_key(path):
    """Numeric sort by client index (client0, client1, ..., client15), not
    lexicographic string sort -- lexicographic order would incorrectly place
    client10-client15 before client2-client9 (e.g. for --n_clients 8, this
    previously selected clients {0,1,10,11,12,13,14,15} instead of the
    intended {0,1,2,3,4,5,6,7}), silently testing a biased, non-representative
    subset. This was diagnosed from an anomalous N=8 result (mu_hat*=1.0,
    opposite direction from theory) that did not replicate the expected
    monotonic pattern seen at N=4 and N=16."""
    import re
    m = re.search(r"client(\d+)_", path)
    return int(m.group(1)) if m else float("inf")


def run_scale_federation(n_clients, seed=0):
    from environment import BuildingEnergyEnv
    from ddpg_agent import DDPGAgent, ReplayBuffer, Critic
    from federated_learning import fedprox_aggregate, add_gaussian_dp_noise

    client_files = sorted(glob.glob(os.path.join(FINALIZED_DIR, "residential_client*_15min.csv")),
                           key=_client_sort_key)[:n_clients]
    if len(client_files) < n_clients:
        raise ValueError(f"Only {len(client_files)} client files found; run --finalize first, "
                          f"or reduce --n_clients.")
    print(f"Running federation across N={len(client_files)} residential clients: "
          f"{[os.path.basename(f) for f in client_files]}  (seed={seed})")

    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = [BuildingEnergyEnv(data_path=f, episode_len=EPISODE_LEN, seed=seed + i)
            for i, f in enumerate(client_files)]
    agents = [DDPGAgent(envs[0].state_dim, envs[0].action_dim, noise_sigma=0.4,
                         noise_decay_steps=int(ROUNDS * STEPS_PER_ROUND * 0.6))
              for _ in client_files]
    bufs = [ReplayBuffer() for _ in client_files]
    global_critic = Critic(envs[0].state_dim, envs[0].action_dim)

    for rnd in range(1, ROUNDS + 1):
        print(f"  [N={n_clients}, seed={seed}] Round {rnd}/{ROUNDS}...")
        noised_critics = []
        for i, (agent, env, buf) in enumerate(zip(agents, envs, bufs)):
            warmup = WARMUP if rnd == 1 else 0
            s = env.reset()
            for step in range(STEPS_PER_ROUND):
                a = np.random.uniform(-1, 1, size=env.action_dim) if step < warmup else agent.act(s)
                s2, r, done, info = env.step(a)
                buf.push(s, a, r, s2, done)
                s = s2
                if len(buf) > BATCH_SIZE and step >= warmup:
                    agent.train_step(buf.sample(BATCH_SIZE), mu=1.0)
                if done:
                    s = env.reset()
            noised_sd, _ = add_gaussian_dp_noise(agent.critic.state_dict(), epsilon=8.0, delta=1e-5)
            noised_critics.append(noised_sd)
        new_global = fedprox_aggregate(noised_critics, global_critic.state_dict(), mu=0.05)
        global_critic.load_state_dict(new_global)
        for agent in agents:
            agent.critic.load_state_dict(new_global)
            agent.critic_target.load_state_dict(new_global)

    # mu-sweep for client 0 only (as a representative probe), fine-tuning
    # from this N-client federation's global critic.
    global_critic_state = global_critic.state_dict()
    print(f"\n[N={n_clients}, seed={seed}] Running mu-sweep for client 0...")
    results = {}
    for mu in MUS:
        env = BuildingEnergyEnv(data_path=client_files[0], episode_len=EPISODE_LEN, seed=seed + 2000)
        agent = DDPGAgent(env.state_dim, env.action_dim, noise_sigma=0.1, noise_decay_steps=3000)
        agent.actor.load_state_dict(agents[0].actor.state_dict())
        agent.critic.load_state_dict(agents[0].critic.state_dict())
        agent.actor_target.load_state_dict(agent.actor.state_dict())
        agent.critic_target.load_state_dict(agent.critic.state_dict())
        buf = ReplayBuffer()
        s = env.reset()
        for step in range(3000):
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            buf.push(s, a, r, s2, done)
            s = s2
            if len(buf) > 128:
                agent.train_step(buf.sample(128), personalisation_target=global_critic_state, mu=mu)
            if done:
                s = env.reset()

        eval_env = BuildingEnergyEnv(data_path=client_files[0], episode_len=EPISODE_LEN, seed=seed + 9000)
        costs = []
        for _ in range(EVAL_EPISODES):
            s = eval_env.reset()
            ep_cost = 0.0
            done = False
            while not done:
                a = agent.act(s, explore=False)
                s, r, done, info = eval_env.step(a)
                ep_cost += info["cost"]
            costs.append(ep_cost)
        results[str(mu)] = float(np.mean(costs))
        print(f"  mu={mu}: cost={results[str(mu)]:.4f}")

    best_mu = min(results, key=results.get)
    print(f"[N={n_clients}, seed={seed}] empirical mu_hat* = {best_mu} (cost={results[best_mu]:.4f})")
    return {"mu_sweep": results, "mu_hat_star": best_mu}


def run_scale_federation_multiseed(n_clients, n_seeds, seed_start=0):
    """Resume-capable: runs N_SEEDS independent federations at this client
    count, saving incrementally, then reports mean/std/mode of mu_hat*
    across seeds -- applying the SAME seed-robustness lesson learned in
    Phase 1 (single-seed pilots at N=8 were misleading; only multi-seed
    results should be trusted for concluding anything about the true
    N-vs-mu* relationship)."""
    out_path = f"results/residential_scale_test_N{n_clients}_multiseed.json"
    os.makedirs("results", exist_ok=True)
    all_results = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)

    for seed in range(seed_start, seed_start + n_seeds):
        key = str(seed)
        if key in all_results:
            print(f"[N={n_clients}, seed={seed}] already done, skipping.")
            continue
        result = run_scale_federation(n_clients, seed)
        all_results[key] = result
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)

    mu_hats = [float(v["mu_hat_star"]) for v in all_results.values()]
    print(f"\n{'='*70}\nN={n_clients} SUMMARY across {len(mu_hats)} seed(s)\n{'='*70}")
    print(f"  mu_hat* values: {mu_hats}")
    print(f"  mean={np.mean(mu_hats):.4f}  median={np.median(mu_hats):.4f}  std={np.std(mu_hats):.4f}")
    return np.mean(mu_hats), np.std(mu_hats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run_federation", action="store_true")
    parser.add_argument("--n_clients", type=int, default=16)
    parser.add_argument("--n_seeds", type=int, default=1,
                         help="Number of independent seeds to run at this --n_clients "
                              "(resume-capable; recommend 3-5 for a trustworthy mu* estimate).")
    parser.add_argument("--seed_start", type=int, default=0)
    args = parser.parse_args()

    if args.finalize:
        finalize_files()
    if args.run_federation:
        run_scale_federation_multiseed(args.n_clients, args.n_seeds, args.seed_start)
    if not args.finalize and not args.run_federation:
        print("Specify --finalize and/or --run_federation.")


if __name__ == "__main__":
    main()

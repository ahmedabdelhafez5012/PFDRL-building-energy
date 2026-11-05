"""
environment.py
Gym-style environment implementing the building-energy MDP of Section 4.3
(State/Action/Reward, Equations 4.1-4.3) for one building archetype.

This instance is driven by REAL exogenous data (residential_archetype_15min.csv,
derived from the uploaded HomeC.csv). The battery (ESS) is a SIMULATED
controllable asset layered on top of the real load/PV/price signals, since the
source home has no battery circuit in its raw data - this is standard practice
in building-energy RL research (real exogenous signals + simulated controllable
assets) and is documented explicitly here for reproducibility (Section 5.1).

Controllable actions for this archetype: HVAC modulation, ESS charge/discharge.
(EV charging is omitted for this archetype: HomeC.csv has no EV circuit. Other
archetypes drawn from datasets with EV circuits, e.g. Pecan Street, would add a
third action dimension here.)
"""
import numpy as np
import pandas as pd


class BuildingEnergyEnv:
    """A minimal Gym-like interface (reset/step) without a hard gymnasium
    dependency, so it runs in constrained environments.
    """

    metadata = {"archetype": "residential", "source": "HomeC.csv (Kaggle)"}

    def __init__(self, data_path="data/residential_archetype_15min.csv",
                 ess_capacity_kwh=None, ess_max_rate_kw=None, ess_eff=0.95,
                 comfort_band_c=2.0, episode_len=96, seed=None):
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.n = len(self.df)

        # Auto-scale the simulated battery to the archetype's own load
        # magnitude if not explicitly given, so a residential-sized battery
        # (13.5 kWh / 5 kW) isn't silently reused (and rendered meaningless)
        # for a much larger commercial/office load. Ratio chosen to roughly
        # match the residential default's coverage (~a few hours of mean load).
        mean_total_kw = (self.df["base_kw"] + self.df["hvac_kw"]).mean()
        if ess_capacity_kwh is None:
            ess_capacity_kwh = max(13.5, mean_total_kw * 2.0)
        if ess_max_rate_kw is None:
            ess_max_rate_kw = max(5.0, mean_total_kw * 0.5)

        self.ess_capacity = ess_capacity_kwh
        self.ess_max_rate = ess_max_rate_kw
        self.ess_eff = ess_eff
        self.comfort_band = comfort_band_c
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

        self.hvac_scale = self.df["hvac_kw"].quantile(0.99) + 1e-6
        self.base_scale = self.df["base_kw"].quantile(0.99) + 1e-6
        self.pv_scale = self.df["pv_kw"].quantile(0.99) + 1e-6
        self.price_scale = self.df["price"].max()

        self.state_dim = 7
        self.action_dim = 2

        self._t0 = 0
        self._t = 0
        self.soc = 0.5

    def reset(self):
        max_start = self.n - self.episode_len - 1
        self._t0 = int(self.rng.integers(0, max_start))
        self._t = self._t0
        self.soc = 0.5
        return self._get_state()

    def _row(self, t):
        return self.df.iloc[t]

    def _get_state(self):
        row = self._row(self._t)
        hour_frac = (self._t % 96) / 96.0
        state = np.array([
            row["base_kw"] / self.base_scale,
            row["hvac_kw"] / self.hvac_scale,
            row["pv_kw"] / self.pv_scale,
            row["price"] / self.price_scale,
            row["t_out_c"] / 40.0,
            self.soc,
            hour_frac,
        ], dtype=np.float32)
        return state

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        row = self._row(self._t)

        hvac_nominal = row["hvac_kw"]
        hvac_actual = np.clip(hvac_nominal * (1.0 + 0.6 * action[0]), 0.0, self.hvac_scale * 1.5)
        discomfort = abs(hvac_actual - hvac_nominal)

        ess_rate = action[1] * self.ess_max_rate
        dt_h = 0.25
        if ess_rate >= 0:
            delta_soc = ess_rate * dt_h * self.ess_eff / self.ess_capacity
        else:
            delta_soc = ess_rate * dt_h / self.ess_eff / self.ess_capacity
        new_soc = np.clip(self.soc + delta_soc, 0.0, 1.0)
        actual_delta_soc = new_soc - self.soc
        actual_ess_kw = (actual_delta_soc * self.ess_capacity / dt_h)
        self.soc = new_soc

        net_load_kw = row["base_kw"] + hvac_actual + actual_ess_kw - row["pv_kw"]
        grid_import_kw = max(net_load_kw, 0.0)
        grid_export_kw = max(-net_load_kw, 0.0)
        cost = grid_import_kw * row["price"] * dt_h - grid_export_kw * (row["price"] * 0.5) * dt_h

        lambda_comfort = 0.15
        reward = -(cost + lambda_comfort * discomfort)

        self._t += 1
        done = (self._t - self._t0) >= self.episode_len or self._t >= self.n - 1
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)

        info = {
            "cost": cost, "discomfort": discomfort, "soc": self.soc,
            "grid_import_kw": grid_import_kw, "grid_export_kw": grid_export_kw,
        }
        return next_state, float(reward), done, info


if __name__ == "__main__":
    env = BuildingEnergyEnv()
    s = env.reset()
    print("Initial state:", s, " (dim =", env.state_dim, ")")
    total_r = 0.0
    for step in range(10):
        a = np.random.uniform(-1, 1, size=env.action_dim)
        s, r, done, info = env.step(a)
        total_r += r
        print(f"step {step}: reward={r:.4f} soc={info['soc']:.3f} cost={info['cost']:.4f}")
        if done:
            break
    print("Smoke-test total reward (10 random steps):", total_r)

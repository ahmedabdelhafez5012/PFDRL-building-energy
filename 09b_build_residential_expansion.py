"""
09b_build_residential_expansion.py
=====================================
STAGE B: builds additional residential archetype clients by combining:
  - REAL base load (GC) and REAL solar generation (GG) from Ausgrid
    (300 real Sydney homes, half-hourly, 2010-2013)
  - a TEMPERATURE-CALIBRATED HVAC model: fit on Pecan Street's REAL
    HVAC-vs-outdoor-temperature relationship (10 real Austin homes,
    1-minute, Aug 2018), then applied to Sydney's REAL historical
    weather for the Ausgrid period.

TRANSPARENCY NOTE (please carry this into any thesis/paper write-up):
  base_kw and pv_kw are REAL measurements. hvac_kw is NOT a direct
  measurement of the same homes -- it is a physically-grounded estimate
  (a standard degree-day-style temperature-response model, a common and
  accepted building-energy-modelling technique) calibrated on real HVAC
  data from a different real population and applied to real weather at
  the target site. This should be reported as semi-synthetic / model-
  derived, not as an independent real measurement, to preserve this
  thesis's standard of transparency.

Prerequisites:
  - Ausgrid CSVs already downloaded by 09a into data/residential_expansion_raw/
  - PecanStreet_10_Homes_1Min_Data.csv present in data/ (already confirmed)
  - Internet access to Open-Meteo's free historical weather API (no key needed)

Run:
    python 09b_build_residential_expansion.py --n_customers 5
"""
import argparse
import os
import glob
import json
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

AUSGRID_DIR = "data/residential_expansion_raw"
PECAN_PATH = "data/PecanStreet_10_Homes_1Min_Data.csv"
OUT_DIR = "data/residential_expansion"

# Austin, TX (Pecan Street) and Sydney, AU (Ausgrid) coordinates
AUSTIN_LAT, AUSTIN_LON = 30.27, -97.74
SYDNEY_LAT, SYDNEY_LON = -33.87, 151.21

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Step 1: real Ausgrid base load + solar generation
# ---------------------------------------------------------------------------
def load_ausgrid_customer(customer_id, n_years_files=None):
    """Loads GC (general consumption) and GG (gross generation/solar) for one
    customer across all available Ausgrid year-files, returns a combined
    half-hourly DataFrame with columns [timestamp, base_kw, pv_kw]."""
    files = n_years_files or sorted(glob.glob(os.path.join(AUSGRID_DIR, "Solar home *.csv")))
    all_rows = []
    for path in files:
        df = pd.read_csv(path, skiprows=1, low_memory=False)
        df = df[df["Customer"] == customer_id]
        if df.empty:
            continue
        time_cols = [c for c in df.columns if ":" in str(c)]
        for cat, out_col in [("GC", "base_kw"), ("GG", "pv_kw")]:
            sub = df[df["Consumption Category"] == cat]
            if sub.empty:
                continue
            melted = sub.melt(id_vars=["date"], value_vars=time_cols,
                               var_name="time", value_name=out_col)
            melted["timestamp"] = pd.to_datetime(
                melted["date"] + " " + melted["time"], dayfirst=True, errors="coerce")
            melted = melted.dropna(subset=["timestamp"])[["timestamp", out_col]]
            all_rows.append((out_col, melted))

    base = None
    for out_col, melted in all_rows:
        melted = melted.groupby("timestamp", as_index=False).mean()
        if base is None:
            base = melted if out_col == "base_kw" else melted.rename(columns={out_col: out_col})
        base = base.merge(melted, on="timestamp", how="outer") if out_col not in (base.columns if base is not None else []) else base

    # Simpler, robust merge: build separately then join
    gc_frames = [m for c, m in all_rows if c == "base_kw"]
    gg_frames = [m for c, m in all_rows if c == "pv_kw"]
    if not gc_frames:
        return None
    gc = pd.concat(gc_frames).groupby("timestamp", as_index=False).mean()
    if gg_frames:
        gg = pd.concat(gg_frames).groupby("timestamp", as_index=False).mean()
        merged = gc.merge(gg, on="timestamp", how="left")
    else:
        merged = gc
        merged["pv_kw"] = 0.0
    merged["pv_kw"] = merged["pv_kw"].fillna(0.0)
    return merged.sort_values("timestamp").reset_index(drop=True)


def find_customers_with_solar(sample_file, min_customers=10):
    """Ausgrid customer IDs are 1..300; scan the first year-file to find
    which ones have BOTH GC and GG (not all 300 have solar)."""
    df = pd.read_csv(sample_file, skiprows=1, usecols=["Customer", "Consumption Category"], low_memory=False)
    has_gc = set(df.loc[df["Consumption Category"] == "GC", "Customer"].unique())
    has_gg = set(df.loc[df["Consumption Category"] == "GG", "Customer"].unique())
    both = sorted(has_gc & has_gg)
    print(f"Found {len(both)} customers with both GC and GG in {sample_file}")
    return both[:min_customers]


# ---------------------------------------------------------------------------
# Step 2: fetch weather (Open-Meteo, free, no API key)
# ---------------------------------------------------------------------------
def fetch_weather(lat, lon, start_date, end_date, cache_path):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["timestamp"])

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m", "timezone": "auto",
    }
    url = OPEN_METEO_ARCHIVE + "?" + urllib.parse.urlencode(params)
    print(f"Fetching weather: {url}")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["hourly"]["time"]),
        "t_out_c": data["hourly"]["temperature_2m"],
    })
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Step 3: calibrate HVAC-vs-temperature model on REAL Pecan Street data
# ---------------------------------------------------------------------------
def calibrate_hvac_model():
    print("Loading Pecan Street real HVAC data...")
    df = pd.read_csv(PECAN_PATH)
    df = df[df["Measure"] == "Real Power"]
    df["Datetime (UTC)"] = pd.to_datetime(df["Datetime (UTC)"])

    hvac = df[df["Circuit"].isin(["HVAC Condenser", "HVAC Fan"])]
    # Sum Condenser+Fan per home per MINUTE (keep minute resolution -- do not
    # collapse to hourly here). Using raw minute-level data gives ~43,000
    # data points (10 homes x 3 days x 1440 min) instead of 73-730, and lets
    # us fit a diurnal (hour-of-day) pattern in addition to the temperature
    # response, since 3 days of already-hot Austin summer weather has very
    # little day-to-day temperature VARIANCE to explain HVAC use from
    # temperature alone (confirmed: R^2=0.001 regardless of aggregation
    # level when temperature is the only predictor -- this is a genuine
    # data limitation, not a bug, since outdoor temp barely changes across
    # these 3 hot days while AC cycles on a daily occupancy/thermostat
    # schedule that correlates much more with time-of-day).
    per_minute = hvac.groupby([hvac["Home ID"], "Datetime (UTC)"])["Value"].sum()
    per_minute = per_minute.reset_index()
    per_minute.columns = ["home_id", "timestamp", "hvac_kw"]

    weather = fetch_weather(AUSTIN_LAT, AUSTIN_LON, "2018-08-23", "2018-08-26",
                             os.path.join(OUT_DIR, "austin_weather_cache.csv"))
    weather["timestamp"] = weather["timestamp"].dt.tz_localize("UTC") if weather["timestamp"].dt.tz is None else weather["timestamp"]
    per_minute["timestamp"] = per_minute["timestamp"].dt.tz_localize(None).dt.tz_localize("UTC") \
        if per_minute["timestamp"].dt.tz is None else per_minute["timestamp"]

    merged = pd.merge_asof(per_minute.sort_values("timestamp"),
                            weather.sort_values("timestamp"),
                            on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h"))
    merged = merged.dropna()
    print(f"  Regression dataset: {len(merged)} (home, minute) data points "
          f"across {merged['home_id'].nunique()} homes.")

    # Multi-variate model: HVAC_kW = a*cooling_degree + b*sin(hour) + d*cos(hour) + c
    # The sin/cos pair captures the diurnal (time-of-day) usage cycle, which
    # is expected to matter more than temperature alone over just 3 days.
    T_base = 21.0
    merged["cooling_degree"] = np.clip(merged["t_out_c"] - T_base, 0, None)
    hour_frac = merged["timestamp"].dt.hour + merged["timestamp"].dt.minute / 60.0
    merged["hour_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    merged["hour_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    A = np.vstack([merged["cooling_degree"], merged["hour_sin"], merged["hour_cos"],
                   np.ones(len(merged))]).T
    coef, resid, rank, sv = np.linalg.lstsq(A, merged["hvac_kw"], rcond=None)
    a, b, d, c = coef
    pred = a * merged["cooling_degree"] + b * merged["hour_sin"] + d * merged["hour_cos"] + c
    r2 = 1 - np.sum((merged["hvac_kw"] - pred)**2) / np.sum((merged["hvac_kw"] - merged["hvac_kw"].mean())**2)

    print(f"Calibrated model: HVAC_kW = {a:.4f}*max(0,T_out-{T_base}) + {b:.4f}*sin(hour) "
          f"+ {d:.4f}*cos(hour) + {c:.4f}   (R^2={r2:.3f}, N={len(merged)})")
    return {"a": float(a), "b": float(b), "d": float(d), "c": float(c),
            "T_base": T_base, "r2": float(r2), "n": len(merged)}


def apply_hvac_model(model, weather_df):
    cooling_degree = np.clip(weather_df["t_out_c"] - model["T_base"], 0, None)
    hour_frac = weather_df["timestamp"].dt.hour + weather_df["timestamp"].dt.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour_frac / 24)
    hour_cos = np.cos(2 * np.pi * hour_frac / 24)
    hvac_kw = model["a"] * cooling_degree + model["b"] * hour_sin + model["d"] * hour_cos + model["c"]
    return np.clip(hvac_kw, 0, None)  # HVAC power can't be negative


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_customers", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70 + "\nSTEP 1: Calibrating HVAC model on real Pecan Street data\n" + "=" * 70)
    hvac_model = calibrate_hvac_model()
    with open(os.path.join(OUT_DIR, "hvac_model.json"), "w") as f:
        json.dump(hvac_model, f, indent=2)

    print("\n" + "=" * 70 + "\nSTEP 2: Fetching real Sydney weather (2010-2013)\n" + "=" * 70)
    sydney_weather = fetch_weather(SYDNEY_LAT, SYDNEY_LON, "2010-07-01", "2013-06-30",
                                    os.path.join(OUT_DIR, "sydney_weather_cache.csv"))
    print(f"  {len(sydney_weather)} hourly readings fetched/cached.")

    print("\n" + "=" * 70 + "\nSTEP 3: Loading real Ausgrid customers (base load + solar)\n" + "=" * 70)
    sample_file = sorted(glob.glob(os.path.join(AUSGRID_DIR, "Solar home *.csv")))[0]
    customer_ids = find_customers_with_solar(sample_file, args.n_customers)

    for cid in customer_ids:
        print(f"\n--- Customer {cid} ---")
        ausgrid_data = load_ausgrid_customer(cid)
        if ausgrid_data is None or len(ausgrid_data) < 100:
            print("  Insufficient data, skipping.")
            continue

        # Merge with weather (nearest hour) to compute modelled HVAC
        ausgrid_data = ausgrid_data.sort_values("timestamp")
        sw = sydney_weather.copy()
        sw["timestamp"] = sw["timestamp"].dt.tz_localize(None)
        ausgrid_data["timestamp"] = pd.to_datetime(ausgrid_data["timestamp"])
        merged = pd.merge_asof(ausgrid_data, sw, on="timestamp", direction="nearest", tolerance=pd.Timedelta("1h"))
        merged["hvac_kw"] = apply_hvac_model(hvac_model, merged)
        merged["hour_frac"] = merged["timestamp"].dt.hour / 24.0 + merged["timestamp"].dt.minute / 1440.0

        # Resample 30-min -> 15-min via linear interpolation, matching existing archetype format
        merged = merged.set_index("timestamp").resample("15min").interpolate(method="linear").reset_index()

        out_path = os.path.join(OUT_DIR, f"residential_ausgrid_customer{cid}_15min.csv")
        merged[["timestamp", "base_kw", "pv_kw", "hvac_kw", "t_out_c"]].to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(merged)} rows, "
              f"base_kw mean={merged['base_kw'].mean():.3f}, "
              f"pv_kw mean={merged['pv_kw'].mean():.3f}, "
              f"hvac_kw mean={merged['hvac_kw'].mean():.3f})")

    print(f"\nDone. Files saved in {OUT_DIR}/.")
    print("Next step: adapt data_loader.py's remaining logic (price signal, state "
          "normalisation) to these new files, then add them as additional residential "
          "clients in 01_full_ablation_b1_b4.py / main_real_federation.py.")


if __name__ == "__main__":
    main()

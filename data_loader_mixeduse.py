"""
data_loader_mixeduse.py
Fourth building archetype: Mixed-use — "Hog_lodging_Shanti" (BDG2, Kaggle
Building Data Genome Project 2). Lodging (hotel) buildings are commonly
treated as "mixed-use" in the building-energy literature because they
combine residential-like occupancy (guest rooms) with commercial services
(restaurants, common areas, laundry, conference facilities).

Real electricity (100% complete), real chilled-water cooling sub-metering
(100% complete), AND real steam heating sub-metering from the CLEANED BDG2
release (steam_cleaned.csv, 99.23% complete, ~0.77% anomalous readings
masked as NaN and linearly interpolated - e.g. the raw release's implausible
13,850 max reading is corrected to 10,091 in the cleaned version) for this
building. HVAC electrical share is derived from a COMBINED real
heating+cooling thermal-activity signal, same methodology and caveats as
data_loader_commercial.py (see that module's docstring for the full
tons/Btu->kW / COP assumption note).

UPDATE: outdoor temperature is now REAL measured weather (weather.csv, site
"Hog", 99.99% complete) - previously a location-grounded synthetic
Minneapolis-climate placeholder. This removes that earlier data-quality
caveat entirely, and reveals the real record is notably colder in winter
(-28.9C observed min) than the earlier placeholder assumed (-9C).
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/mixeduse_raw_hog_lodging_shanti.csv"
WEATHER_PATH = "data/weather_hog.csv"
ELEC_COL = "Hog_lodging_Shanti_elec"
CHW_COL = "Hog_lodging_Shanti_chw"
STEAM_COL = "Hog_lodging_Shanti_steam"

MIN_HVAC_FRAC = 0.15
MAX_HVAC_FRAC = 0.65


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df[ELEC_COL] = df[ELEC_COL].interpolate(limit_direction="both")
    df[CHW_COL] = df[CHW_COL].interpolate(limit_direction="both").clip(lower=0)
    df[STEAM_COL] = df[STEAM_COL].interpolate(limit_direction="both").clip(lower=0)
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    df = df.merge(weather, on="timestamp", how="left")
    df["airTemperature"] = df["airTemperature"].interpolate(limit_direction="both")
    return df


def build_archetype_frame(df, resample_to="15min"):
    df = df.set_index("timestamp").sort_index()
    hourly = pd.DataFrame(index=df.index)

    total_kw = df[ELEC_COL].astype(float)
    cw = df[CHW_COL].astype(float)
    steam = df[STEAM_COL].astype(float)
    cooling_idx = (cw / (cw.max() + 1e-9)).clip(0, 1)
    heating_idx = (steam / (steam.max() + 1e-9)).clip(0, 1)
    thermal_idx = np.maximum(cooling_idx, heating_idx)

    hvac_frac = MIN_HVAC_FRAC + (MAX_HVAC_FRAC - MIN_HVAC_FRAC) * thermal_idx
    hourly["hvac_kw"] = total_kw * hvac_frac
    hourly["base_kw"] = total_kw * (1 - hvac_frac)
    hourly["pv_kw"] = 0.0  # no solar data available for this building
    hourly["t_out_c"] = df["airTemperature"].values  # REAL measured weather (site "Hog")

    res = hourly.resample(resample_to).interpolate(method="linear")
    res[["hvac_kw", "base_kw"]] = hourly[["hvac_kw", "base_kw"]].resample(resample_to).ffill()
    res = res.dropna()
    return res


def time_of_use_price(index):
    hour = index.hour
    price = np.where(
        (hour >= 17) & (hour < 21), 0.42,
        np.where((hour >= 7) & (hour < 17) | (hour >= 21) & (hour < 23), 0.22, 0.11),
    )
    return pd.Series(price, index=index)


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw rows: {len(raw)}")
    frame = build_archetype_frame(raw)
    frame["price"] = time_of_use_price(frame.index)
    print(f"Rows after resampling: {len(frame)}  span: {frame.index[0]} -> {frame.index[-1]}")
    print(frame.describe())
    frame.to_csv("data/mixeduse_archetype_15min.csv")
    print("Saved: data/mixeduse_archetype_15min.csv")

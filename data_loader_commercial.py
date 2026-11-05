"""
data_loader_commercial.py
Third building archetype: Commercial/Retail — "Fox_retail_Manie" (BDG2,
Kaggle Building Data Genome Project 2). Real electricity (99.9% complete),
real chilled-water cooling sub-metering (99.99% complete), AND real
hot-water heating sub-metering (98.0% complete) for this building — the
HVAC electrical share is now derived from a COMBINED real heating+cooling
thermal-activity signal (previously cooling-only), a genuine fidelity
improvement addressing winter heating loads that were incorrectly folded
into "base load" in the earlier version.

METHODOLOGY NOTE (Section 5.1 reproducibility): BDG2's chilled-water/hot-water
meter units and this building's exact plant COP are not available without
further documentation, so an exact tons/Btu->kW conversion cannot be
certified. Instead, both series are separately normalised to 0-1 activity
indices (relative to this building's own 2-year record) and combined via
max(cooling_index, heating_index) - physically motivated since heating and
cooling plants are rarely both at full load simultaneously - to modulate the
HVAC electrical share between a documented minimum (MIN_HVAC_FRAC) and
maximum (MAX_HVAC_FRAC) of total metered electricity.

UPDATE: outdoor temperature is now REAL measured weather (weather.csv, site
"Fox", 99.99% complete) - previously a location-grounded synthetic Phoenix-
climate placeholder. This removes that earlier data-quality caveat entirely.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/commercial_raw_fox_retail_manie.csv"
WEATHER_PATH = "data/weather_fox.csv"
ELEC_COL = "Fox_retail_Manie_elec"
CHW_COL = "Fox_retail_Manie_chw"
HW_COL = "Fox_retail_Manie_hw"

MIN_HVAC_FRAC = 0.15
MAX_HVAC_FRAC = 0.65


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df[ELEC_COL] = df[ELEC_COL].interpolate(limit_direction="both")
    df[CHW_COL] = df[CHW_COL].interpolate(limit_direction="both").clip(lower=0)
    df[HW_COL] = df[HW_COL].interpolate(limit_direction="both").clip(lower=0)
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    df = df.merge(weather, on="timestamp", how="left")
    df["airTemperature"] = df["airTemperature"].interpolate(limit_direction="both")
    return df


def build_archetype_frame(df, resample_to="15min"):
    df = df.set_index("timestamp").sort_index()
    hourly = pd.DataFrame(index=df.index)

    total_kw = df[ELEC_COL].astype(float)
    cw = df[CHW_COL].astype(float)
    hw = df[HW_COL].astype(float)
    cooling_idx = (cw / (cw.max() + 1e-9)).clip(0, 1)
    heating_idx = (hw / (hw.max() + 1e-9)).clip(0, 1)
    thermal_idx = np.maximum(cooling_idx, heating_idx)  # rarely both at once

    hvac_frac = MIN_HVAC_FRAC + (MAX_HVAC_FRAC - MIN_HVAC_FRAC) * thermal_idx
    hourly["hvac_kw"] = total_kw * hvac_frac
    hourly["base_kw"] = total_kw * (1 - hvac_frac)
    hourly["pv_kw"] = 0.0  # no solar data available for this building
    hourly["t_out_c"] = df["airTemperature"].values  # REAL measured weather (site "Fox")

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
    frame.to_csv("data/commercial_archetype_15min.csv")
    print("Saved: data/commercial_archetype_15min.csv")

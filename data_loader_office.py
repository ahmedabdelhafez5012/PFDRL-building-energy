"""
data_loader_office.py
Loads and re-profiles the SECOND (office) building archetype from real BDG2
data (Building Data Genome Project 2, Kaggle) - building "Eagle_office_Katheleen",
hourly electricity meter, 2016-01-01 to 2017-12-31 (2 full years, 100% complete,
no missing values).

UPDATE: outdoor temperature is now REAL measured weather (weather.csv, site
"Eagle", 99.98% complete, linearly interpolated for the small remainder) -
previously a synthetic seasonal placeholder. This removes that earlier
data-quality caveat entirely for this archetype.

There is still no PV/solar generation data available for this building
(BDG2's solar.csv had no non-null column for the office buildings checked),
so pv_kw is set to 0 throughout - a real, not fabricated, absence.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/office_raw_eagle_katheleen_cleaned.csv"
WEATHER_PATH = "data/weather_eagle.csv"
BUILDING_COL = "Eagle_office_Katheleen"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df[BUILDING_COL] = df[BUILDING_COL].interpolate(limit_direction="both")
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["timestamp"])
    df = df.merge(weather, on="timestamp", how="left")
    df["airTemperature"] = df["airTemperature"].interpolate(limit_direction="both")
    return df


def build_archetype_frame(df, resample_to="15min"):
    df = df.set_index("timestamp").sort_index()
    hourly = pd.DataFrame(index=df.index)

    total_kw = df[BUILDING_COL].astype(float)
    # Office HVAC-equivalent share: assume ~45% of an office's load is HVAC
    # (a standard rule-of-thumb split for commercial office buildings; see
    # e.g. DOE commercial building end-use breakdowns), remainder is base
    # (plug loads, lighting, IT).
    hourly["hvac_kw"] = total_kw * 0.45
    hourly["base_kw"] = total_kw * 0.55
    hourly["pv_kw"] = 0.0  # no on-site generation for this building (see docstring)
    hourly["t_out_c"] = df["airTemperature"].values  # REAL measured weather (site "Eagle")

    # Upsample hourly -> 15-min via forward-fill + linear interpolation on
    # temperature only (load stays hourly-stepwise, consistent with the
    # meter's real reporting resolution; documented explicitly, unlike the
    # residential archetype which is natively fine-grained).
    res = hourly.resample(resample_to).interpolate(method="linear")
    res[["hvac_kw", "base_kw"]] = hourly[["hvac_kw", "base_kw"]].resample(resample_to).ffill()
    res = res.dropna()
    return res


def time_of_use_price(index):
    """Same 3-tier ToU tariff structure as the residential archetype, for a
    consistent, comparable price signal across archetypes (Section 4.3)."""
    hour = index.hour
    price = np.where(
        (hour >= 17) & (hour < 21), 0.42,
        np.where((hour >= 7) & (hour < 17) | (hour >= 21) & (hour < 23), 0.22, 0.11),
    )
    return pd.Series(price, index=index)


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw rows: {len(raw)}, missing values: {raw[BUILDING_COL].isna().sum()}")
    frame = build_archetype_frame(raw)
    frame["price"] = time_of_use_price(frame.index)
    print(f"Rows after resampling to 15-min: {len(frame)}")
    print(f"Span: {frame.index[0]} -> {frame.index[-1]}")
    print(frame.describe())
    frame.to_csv("data/office_archetype_15min.csv")
    print("Saved: data/office_archetype_15min.csv")

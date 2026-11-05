"""
data_loader.py
Loads and cleans the residential smart-home dataset (HomeC.csv, Kaggle "Smart Home
Dataset with weather Information") and re-profiles it into 15-minute control
intervals suitable for the PFDRL environment (Section 5.1 of the thesis).

IMPORTANT DATA NOTE (documented for reproducibility / Section 5.1):
The 'time' column in the raw file increments by 1 per row but the dataset is
widely documented (Kaggle) as representing 1-MINUTE resolution readings over
~350 days. We follow the standard community convention: treat each row as one
minute, generating a synthetic minute-level DatetimeIndex from the first
timestamp rather than trusting the raw 'time' column literally. This is noted
as a data-quality caveat in Section 5.1 rather than silently corrected.
"""
import pandas as pd
import numpy as np

RAW_PATH = "data/HomeC.csv"


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, low_memory=False)
    # drop the malformed trailing row (blank time field)
    df = df[df["time"].apply(lambda x: str(x).strip().isdigit())]
    # drop duplicated-header corruption rows (cloudCover literal string artifact)
    df = df[df["cloudCover"] != "cloudCover"].reset_index(drop=True)
    return df


def build_archetype_frame(df, freq="1min", resample_to="15min"):
    """Builds a clean, resampled dataframe with the exogenous signals the
    PFDRL environment needs: base load, HVAC-equivalent load, PV generation,
    outdoor temperature. Returns a dataframe indexed by 15-minute timestamps.
    """
    n = len(df)
    idx = pd.date_range("2016-01-01", periods=n, freq=freq)
    out = pd.DataFrame(index=idx)

    # HVAC-equivalent controllable thermal load = Furnace 1 + Furnace 2
    out["hvac_kw"] = pd.to_numeric(df["Furnace 1 [kW]"], errors="coerce").values \
        + pd.to_numeric(df["Furnace 2 [kW]"], errors="coerce").values
    # Fixed (uncontrollable) base load = house overall - hvac - solar credit
    out["base_kw"] = pd.to_numeric(df["use [kW]"], errors="coerce").values - out["hvac_kw"].values
    out["base_kw"] = out["base_kw"].clip(lower=0)
    # PV generation (this home has rooftop solar)
    out["pv_kw"] = pd.to_numeric(df["gen [kW]"], errors="coerce").values
    # outdoor temperature (Fahrenheit in source; convert to Celsius)
    temp_f = pd.to_numeric(df["temperature"], errors="coerce").values
    out["t_out_c"] = (temp_f - 32.0) * 5.0 / 9.0

    out = out.interpolate(limit_direction="both")

    # Resample to the 15-minute control interval used in the thesis
    # (mean power over each 15-min window; energy-preserving for kW readings)
    res = out.resample(resample_to).mean()
    res = res.dropna()
    return res


def time_of_use_price(index):
    """Simple 3-tier time-of-use tariff ($/kWh), matching the ToU structure
    referenced in Section 4.3 (Reward Function). Peak: 17:00-21:00,
    Shoulder: 07:00-17:00 and 21:00-23:00, Off-peak: 23:00-07:00.
    """
    hour = index.hour
    price = np.where(
        (hour >= 17) & (hour < 21), 0.42,
        np.where((hour >= 7) & (hour < 17) | (hour >= 21) & (hour < 23), 0.22, 0.11),
    )
    return pd.Series(price, index=index)


def summarize(res):
    print(f"Rows after resampling: {len(res)}  |  span: {res.index[0]} -> {res.index[-1]}")
    print(res.describe())


if __name__ == "__main__":
    raw = load_raw()
    print(f"Raw rows (cleaned): {len(raw)}")
    frame = build_archetype_frame(raw)
    frame["price"] = time_of_use_price(frame.index)
    summarize(frame)
    frame.to_csv("data/residential_archetype_15min.csv")
    print("Saved: data/residential_archetype_15min.csv")

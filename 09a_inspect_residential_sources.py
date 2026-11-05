"""
09a_inspect_residential_sources.py
=====================================
STAGE A (diagnostic, run this FIRST): downloads Ausgrid's real solar-home
data and reports its exact structure, and checks for your local Pecan
Street Kaggle sample. This lets us confirm real column names/formats
before writing the full combination pipeline (09b), avoiding wasted work
on guessed structures.

Ausgrid: downloaded automatically (public mirror, CC BY 4.0).
Pecan Street: you must download it yourself first (Kaggle account
required, but free, no institutional registration):
    1. https://www.kaggle.com/datasets/pecanstreetinc/ps-data-10-homes-1min-frequency-aug-2018
    2. Click "Download" (requires a free Kaggle login)
    3. Unzip into: data/pecan_street_sample/

Run:
    python 09a_inspect_residential_sources.py
"""
import os
import zipfile
import urllib.request

import pandas as pd

AUSGRID_URL = "https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip"
OUT_DIR = "data/residential_expansion_raw"
PECAN_DIR = "data/pecan_street_sample"


def download_ausgrid():
    os.makedirs(OUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUT_DIR, "ausgrid_raw.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading Ausgrid data from {AUSGRID_URL} ...")
        urllib.request.urlretrieve(AUSGRID_URL, zip_path)
        print(f"  Saved: {zip_path}")
    else:
        print(f"  Already downloaded: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        print(f"\nAusgrid zip contains {len(names)} file(s):")
        for n in names[:20]:
            print(f"  {n}")
        # Extract CSV(s)
        zf.extractall(OUT_DIR)
    return [os.path.join(OUT_DIR, n) for n in names if n.lower().endswith(".csv")]


def inspect_ausgrid(csv_paths):
    for path in csv_paths:
        print(f"\n{'='*70}\nInspecting: {path}\n{'='*70}")
        try:
            df = pd.read_csv(path, nrows=10, skiprows=0)
        except Exception as e:
            print(f"  Standard read failed ({e}); trying skiprows=1 (common for this dataset -- "
                  f"first row is often a title, second row is the real header)...")
            df = pd.read_csv(path, nrows=10, skiprows=1)
        print("Columns:", df.columns.tolist())
        print("\nFirst rows:")
        print(df.head())
        if "Consumption Category" in df.columns or "Consumption Category" in str(df.columns):
            print("\n'Consumption Category' column found -- unique values in a larger sample:")
            full = pd.read_csv(path, skiprows=1, usecols=["Consumption Category"]) \
                if "Consumption Category" not in df.columns else pd.read_csv(path, usecols=["Consumption Category"])
            print(full["Consumption Category"].unique())


def inspect_pecan_street():
    print(f"\n{'='*70}\nChecking for local Pecan Street sample in {PECAN_DIR}/\n{'='*70}")
    if not os.path.isdir(PECAN_DIR):
        print(f"  NOT FOUND. Please download the Kaggle sample first:")
        print(f"  https://www.kaggle.com/datasets/pecanstreetinc/ps-data-10-homes-1min-frequency-aug-2018")
        print(f"  and unzip its contents into: {PECAN_DIR}/")
        return

    csvs = [f for f in os.listdir(PECAN_DIR) if f.lower().endswith(".csv")]
    print(f"Found {len(csvs)} CSV file(s): {csvs}")
    for f in csvs[:3]:
        path = os.path.join(PECAN_DIR, f)
        print(f"\n--- {f} ---")
        df = pd.read_csv(path, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head())


def main():
    print("STAGE A: Inspecting residential data sources\n")
    csv_paths = download_ausgrid()
    inspect_ausgrid(csv_paths)
    inspect_pecan_street()
    print("\n\nDone. Share this output so we can confirm exact column names "
          "before writing the full combination pipeline (09b).")


if __name__ == "__main__":
    main()

import argparse, glob, json, os
import numpy as np
import pandas as pd
D = "data/bdg2"; OUT = "data/multi"
USAGE = {"office": "Office", "commercial": "Retail", "mixed_use": "Lodging/residential"}
def price(idx):
    h = idx.hour
    return pd.Series(np.where((h >= 17) & (h < 21), 0.42, np.where(((h >= 7) & (h < 17)) | ((h >= 21) & (h < 23)), 0.22, 0.11)), index=idx)
def load(*names):
    for n in names:
        p = os.path.join(D, n + ".csv")
        if os.path.exists(p): return pd.read_csv(p, parse_dates=["timestamp"], index_col="timestamp")
    return None
def build(b, site, elec, chw, hw, steam, weather, arch):
    s = elec[b].interpolate(limit_direction="both").astype(float)
    basis = "fixed 0.45" if arch == "office" else "fixed 0.40"
    frac = pd.Series(0.45 if arch == "office" else 0.40, index=s.index)
    if arch != "office":
        idx = []
        if chw is not None and b in chw.columns and chw[b].notna().mean() > 0.8:
            c = chw[b].reindex(s.index).interpolate(limit_direction="both").clip(lower=0); idx.append((c / (c.max() + 1e-9)).clip(0, 1))
        for heat in (hw, steam):
            if heat is not None and b in heat.columns and heat[b].notna().mean() > 0.8:
                h = heat[b].reindex(s.index).interpolate(limit_direction="both").clip(lower=0); idx.append((h / (h.max() + 1e-9)).clip(0, 1))
        if idx:
            thermal = pd.concat(idx, axis=1).max(axis=1); frac = 0.15 + 0.50 * thermal; basis = "chilled-water/heating activity, 0.15-0.65 of metered electricity"
    w = weather[weather["site_id"] == site].set_index("timestamp")["airTemperature"]
    w = w[~w.index.duplicated()].reindex(s.index).interpolate(limit_direction="both")
    hourly = pd.DataFrame({"hvac_kw": s * frac, "base_kw": s * (1 - frac), "pv_kw": 0.0, "t_out_c": w.values}, index=s.index)
    res = hourly.resample("15min").interpolate(method="linear")
    loads = hourly[["hvac_kw", "base_kw"]].resample("15min").ffill(); res[["hvac_kw", "base_kw"]] = loads
    res = res.dropna(); res["price"] = price(res.index)
    return res[["hvac_kw", "base_kw", "pv_kw", "t_out_c", "price"]], basis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--min_complete", type=float, default=0.95)
    ap.add_argument("--max_per_site", type=int, default=2)
    ap.add_argument("--residential_dir", default="data/residential_scale_clients")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    meta = pd.read_csv(os.path.join(D, "metadata.csv")); elec = load("electricity_cleaned", "electricity")
    chw = load("chilledwater_cleaned", "chilledwater"); hw = load("hotwater_cleaned", "hotwater"); steam = load("steam_cleaned", "steam")
    weather = pd.read_csv(os.path.join(D, "weather.csv"), parse_dates=["timestamp"])
    manifest = []
    for arch, usage in USAGE.items():
        cand = meta[(meta["primaryspaceusage"] == usage) & (meta["building_id"].isin(elec.columns))].copy()
        cand["complete"] = [elec[b].notna().mean() for b in cand["building_id"]]
        cand = cand[cand["complete"] >= a.min_complete].sort_values(["complete", "building_id"], ascending=[False, True])
        chosen, per_site = [], {}
        for _, r in cand.iterrows():
            if per_site.get(r["site_id"], 0) >= a.max_per_site: continue
            chosen.append(r); per_site[r["site_id"]] = per_site.get(r["site_id"], 0) + 1
            if len(chosen) == a.k: break
        print(arch, "candidates:", len(cand), "chosen:", [c["building_id"] for c in chosen])
        for r in chosen:
            df, basis = build(r["building_id"], r["site_id"], elec, chw, hw, steam, weather, arch)
            path = os.path.join(OUT, "%s_%s_15min.csv" % (arch, r["building_id"])); df.to_csv(path)
            manifest.append({"arch": arch, "id": str(r["building_id"]), "path": path, "site": str(r["site_id"]), "complete": float(r["complete"]), "hvac_basis": basis, "rows": int(len(df))})
    files = sorted(glob.glob(os.path.join(a.residential_dir, "residential_client*_15min.csv")), key=lambda p: int(p.split("client")[-1].split("_")[0]))
    for f in files[:a.k]:
        cid = os.path.basename(f).split("_")[1]
        manifest.append({"arch": "residential", "id": cid, "path": f, "site": "n/a", "complete": 1.0, "hvac_basis": "client0 = HomeC measured furnace circuits; others Ausgrid base/PV with a modelled HVAC series shared by all Ausgrid clients", "rows": -1})
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("manifest:", len(manifest), "clients ->", os.path.join(OUT, "manifest.json"))
main()
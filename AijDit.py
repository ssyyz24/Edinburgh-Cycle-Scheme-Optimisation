import pandas as pd
import numpy as np
from pathlib import Path
import os


script_dir = Path(__file__).parent
os.chdir(script_dir)


STATION_FILE = Path("station_data")   
COUNTS_DIR   = Path("counts-data")    
OUTPUT_DIR   = Path("outputs")

RADIUS_M = 500  
AM_HOURS = set(range(7, 10))    
PM_HOURS = set(range(16, 20))  

def hour_to_period(h: int) -> str:
    if h in AM_HOURS: return "AM"
    if h in PM_HOURS: return "PM"
    return "OP"


def read_any_table(path_wo_suffix: Path) -> pd.DataFrame:
    
    for suf, reader in [(".csv", pd.read_csv), (".xlsx", pd.read_excel), (".xls", pd.read_excel)]:
        p = path_wo_suffix.with_suffix(suf)
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"Cannot find {path_wo_suffix} with .csv/.xls/.xlsx")

def read_all_counts(counts_dir: Path) -> pd.DataFrame:
    files = []
    files += list(counts_dir.glob("*.csv"))
    files += list(counts_dir.glob("*.xlsx"))
    files += list(counts_dir.glob("*.xls"))
    if not files:
        raise FileNotFoundError(f"No CSV/XLS files found in {counts_dir.resolve()}")

    dfs = []
    for f in files:
        if f.suffix.lower()==".csv":
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        
        lower = {c.lower(): c for c in df.columns}

        origin_candidates = ["origin_id","start_station_id","start_id","origin","from","o","startstationid"]
        hour_candidates   = ["hour","start_hour","hour_of_day","starthour"]
        cnt_candidates    = ["total_count","trips","count","n","num_trips"]

        def pick(cands):
            for key in cands:
                if key in lower: return lower[key]
            return None

        c_origin = pick(origin_candidates)
        c_hour   = pick(hour_candidates)
        c_cnt    = pick(cnt_candidates)

        if c_origin is None: raise ValueError(f"{f.name}: cannot find origin column")
        if c_hour   is None: raise ValueError(f"{f.name}: cannot find hour column (0-23)")
        if c_cnt    is None:
            df["__ones__"] = 1
            c_cnt = "__ones__"

        df = df[[c_origin, c_hour, c_cnt]].rename(
            columns={c_origin:"origin_id", c_hour:"hour", c_cnt:"total_count"}
        )
        df = df[(df["hour"]>=0) & (df["hour"]<=23)]
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out["origin_id"]   = out["origin_id"].astype(str)
    out["total_count"] = pd.to_numeric(out["total_count"], errors="coerce").fillna(0)
    return out

def load_stations() -> pd.DataFrame:
    s = read_any_table(STATION_FILE)
    lower = {c.lower(): c for c in s.columns}
    def pick(*names):
        for n in names:
            if n in lower: return lower[n]
        return None
    c_id  = pick("station_id","id","station","code")
    c_lat = pick("lat","latitude")
    c_lon = pick("lon","lng","long","longitude")
    if not all([c_id, c_lat, c_lon]):
        raise ValueError("station_data.* must contain station_id, lat, lon (or variants)")
    s = s[[c_id, c_lat, c_lon]].rename(columns={c_id:"station_id", c_lat:"lat", c_lon:"lon"})
    s["station_id"] = s["station_id"].astype(str)
    s = s.dropna(subset=["lat","lon"])
    return s


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088  # km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = load_stations()
    counts   = read_all_counts(COUNTS_DIR)

    
    counts["origin_id"] = counts["origin_id"].astype(str)
    valid_ids = set(stations["station_id"])
    counts = counts[counts["origin_id"].isin(valid_ids)]

   
    counts["period"] = counts["hour"].astype(int).apply(hour_to_period)

    
    d_it = (
        counts.groupby(["origin_id","period"], as_index=False)["total_count"]
              .sum()
              .rename(columns={"origin_id":"i","period":"t","total_count":"d_it"})
    )
   
    all_i = stations["station_id"].unique()
    all_t = ["AM","OP","PM"]
    full = pd.MultiIndex.from_product([all_i, all_t], names=["i","t"]).to_frame(index=False)
    d_it = full.merge(d_it, on=["i","t"], how="left").fillna({"d_it":0})
    d_it["d_it"] = d_it["d_it"].astype(int)
    d_it.to_csv(OUTPUT_DIR/"d_it.csv", index=False)

    
    I = stations.rename(columns={"station_id":"i","lat":"ilat","lon":"ilon"}).copy()
    J = stations.rename(columns={"station_id":"j","lat":"jlat","lon":"jlon"}).copy()
    I["_k"]=1; J["_k"]=1
    IJ = I.merge(J, on="_k", how="outer").drop(columns="_k")

    dist_km = haversine_km(IJ["ilat"].to_numpy(), IJ["ilon"].to_numpy(),
                           IJ["jlat"].to_numpy(), IJ["jlon"].to_numpy())
    IJ["distance_m"] = dist_km * 1000
    IJ["A_ij"] = (IJ["distance_m"] <= RADIUS_M).astype(int)

    A = IJ.pivot(index="i", columns="j", values="A_ij").sort_index(axis=0).sort_index(axis=1)
    D = IJ.pivot(index="i", columns="j", values="distance_m").sort_index(axis=0).sort_index(axis=1)


    A.to_csv(OUTPUT_DIR/"A_ij.csv")
    D.to_csv(OUTPUT_DIR/"distance_matrix_m.csv")

    print("✅ finish：")
    print(" - outputs/d_it.csv")
    print(" - outputs/A_ij.csv")
    print(" - outputs/distance_matrix_m.csv")

if __name__ == "__main__":
    main()

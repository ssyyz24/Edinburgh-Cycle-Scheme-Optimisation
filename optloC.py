import pandas as pd
import numpy as np
from pathlib import Path
import pulp as pl


script_dir = Path(__file__).parent.resolve()

STATIONS = script_dir / "station_data.csv"
A_FILE   = script_dir / "A_ij.csv"
D_FILE   = script_dir / "distance_matrix_m.csv"
DIT_FILE = script_dir / "d_it.csv"
OUT_CSV  = script_dir / "outputs" / "optimal_location_baseline.csv"


C_d = 500.0                 
DEFAULT_F_SITE = 10000.0    
DEFAULT_F_INST =  6000.0    


c_op = 0.10                
lam  = 1.0                  


kappa = 5.0                 


BUDGET = 900000.0           

P0 = 0.80                   
alpha = {                   
    "AM": 9.6,
    "OP": 7.2,
    "PM": 9.6,
}
v = {                       
    "AM": 12.0,
    "OP": 15.0,
    "PM": 12.0,
}

all_T = ["AM", "OP", "PM"]


MIN_DOCKS_PER_OPEN_STATION = 15


MIN_OPEN_STATIONS = 35


def main():
    print(">>> Loading data ...")

    
    stations = pd.read_csv(STATIONS)
    stations.columns = [c.strip() for c in stations.columns]
    stations["station_id"] = stations["station_id"].astype(str)

    
    A = pd.read_csv(A_FILE, index_col=0)
    A.index = A.index.astype(str)
    A.columns = A.columns.astype(str)

    
    D = pd.read_csv(D_FILE, index_col=0)
    D.index = D.index.astype(str)
    D.columns = D.columns.astype(str)
    D = D / 1000.0  # km

    
    d_it_df = pd.read_csv(DIT_FILE)
    d_it_df["i"] = d_it_df["i"].astype(str)

    
    I = sorted(set(A.index) & set(D.index) & set(d_it_df["i"]))
    J = sorted(set(A.columns) & set(D.columns) & set(stations["station_id"]))

    print(f"--- Index Check ---")
    print(f"Demand nodes I: {len(I)}")
    print(f"Station sites J: {len(J)}")
    if len(I) == 0 or len(J) == 0:
        raise ValueError("If I or J is empty, please verify that the IDs for A_ij / D_ij / d_it / station_data are consistent.")
    print("---------------------\n")

    
    A = A.loc[I, J].astype(int)
    D = D.loc[I, J]

    
    grid = pd.MultiIndex.from_product([I, all_T], names=["i", "t"]).to_frame(index=False)
    d_it_df = grid.merge(d_it_df, on=["i", "t"], how="left").fillna({"d_it": 0.0})
    d_it_df["d_it"] = d_it_df["d_it"].astype(float)

    
    d_it = d_it_df.pivot(index="i", columns="t", values="d_it").reindex(index=I, columns=all_T, fill_value=0.0)

    
    def pick(col, default):
        if col in stations.columns:
            return stations.set_index("station_id")[col].astype(float)
        else:
            return pd.Series(default, index=stations["station_id"].astype(str))

    F_site_series = pick("F_site", DEFAULT_F_SITE)
    F_inst_series = pick("F_inst", DEFAULT_F_INST)

    F_site = F_site_series.reindex(J, fill_value=DEFAULT_F_SITE)
    F_inst = F_inst_series.reindex(J, fill_value=DEFAULT_F_INST)

    
    print(">>> Pre-computing P_ijt ...")
    P_ijt = {}
    for i in I:
        for j in J:
            if A.loc[i, j] == 1:        
                dist_km = float(D.loc[i, j])  
                for t in all_T:
                    tau = dist_km / v[t]      
                    P_ijt[(i, j, t)] = P0 + alpha[t] * tau - c_op

    
    print(">>> Building MILP model ...")
    m = pl.LpProblem("Baseline_Station_Location_Model", pl.LpMaximize)

    
    y = pl.LpVariable.dicts("y", J, lowBound=0, upBound=1, cat=pl.LpBinary)

    
    m_j_var_type = pl.LpInteger  
    m_j = pl.LpVariable.dicts("m", J, lowBound=0, cat=m_j_var_type)

    
    X_index = []
    for i in I:
        for j in J:
            if A.loc[i, j] == 1:
                for t in all_T:
                    if d_it.loc[i, t] > 0:
                        X_index.append((i, j, t))
    x = pl.LpVariable.dicts("x", X_index, lowBound=0, cat=pl.LpContinuous)

    
    U_index = [(i, t) for i in I for t in all_T]
    u = pl.LpVariable.dicts("u", U_index, lowBound=0, cat=pl.LpContinuous)

    
    print(">>> Setting objective ...")
    revenue_term = pl.lpSum(
        P_ijt[(i, j, t)] * x[(i, j, t)]
        for (i, j, t) in X_index
    )

    capex_site_term = pl.lpSum((F_site[j] + F_inst[j]) * y[j] for j in J)
    capex_dock_term = pl.lpSum(C_d * m_j[j] for j in J)
    unmet_term = pl.lpSum(lam * u[(i, t)] for (i, t) in U_index)

    m += revenue_term - capex_site_term - capex_dock_term - unmet_term

    
    print(">>> Adding demand balance constraints ...")
    for i in I:
        for t in all_T:
            m += (
                pl.lpSum(x[(i, j, t)] for j in J if (i, j, t) in X_index)
                + u[(i, t)]
                == d_it.loc[i, t]
            ), f"demand_balance_{i}_{t}"

    
    print(">>> Adding coverage constraints ...")
    for (i, j, t) in X_index:
        m += x[(i, j, t)] <= d_it.loc[i, t] * y[j], f"coverage_{i}_{j}_{t}"

    
    print(">>> Adding throughput constraints ...")
    for j in J:
        for t in all_T:
            m += (
                pl.lpSum(
                    x[(i, j2, t2)]
                    for (i, j2, t2) in X_index
                    if j2 == j and t2 == t
                )
                <= kappa * m_j[j]
            ), f"throughput_{j}_{t}"

    
    print(">>> Adding budget constraint ...")
    m += pl.lpSum(
        (F_site[j] + F_inst[j]) * y[j] + C_d * m_j[j]
        for j in J
    ) <= BUDGET, "capital_budget"

    
    print(">>> Adding linking & station-count constraints ...")
    BIG_M_DOCKS = 1000
    for j in J:
        m += m_j[j] <= BIG_M_DOCKS * y[j], f"dock_link_upper_{j}"
        m += m_j[j] >= MIN_DOCKS_PER_OPEN_STATION * y[j], f"dock_link_lower_{j}"

   
    effective_min_stations = min(MIN_OPEN_STATIONS, len(J))
    m += pl.lpSum(y[j] for j in J) >= effective_min_stations, "min_open_stations"

    
    print(">>> Solving MILP ...")
    m.solve(pl.PULP_CBC_CMD(msg=1))
    status = pl.LpStatus[m.status]
    print(f"Solver Status: {status}")

    if status != "Optimal":
        print("⚠ No optimal solution found or solution failed.")
        return

    
    print("\n>>> Extracting solution ...")

    sel_stations = pd.DataFrame({
        "station_id": J,
        "selected": [int(pl.value(y[j]) > 0.5) for j in J],
        "docks": [int(round(pl.value(m_j[j]) if pl.value(m_j[j]) is not None else 0)) for j in J]
    })

    out = stations.merge(sel_stations, on="station_id", how="right")
    script_dir.joinpath("outputs").mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

   
    total_demand = float(d_it_df["d_it"].sum())
    total_served = float(sum(pl.value(x[idx]) for idx in X_index))
    total_unmet = float(sum(pl.value(u[idx]) for idx in U_index))
    coverage_rate = 0.0 if total_demand == 0 else total_served / total_demand

    total_capex = float(sum(
        (F_site[j] + F_inst[j]) * pl.value(y[j]) + C_d * pl.value(m_j[j])
        for j in J
    ))
    obj_value = pl.value(m.objective)
    num_stations_open = sum(int(pl.value(y[j]) > 0.5) for j in J)

    print(f"✅ Saved baseline solution to: {OUT_CSV}")
    print("\n--- Baseline Model Results (one-day, ≥35 stations, ≥15 docks per station) ---")
    print(f"Objective value:   {obj_value:,.2f}")
    print(f"Total demand:      {total_demand:,.0f}")
    print(f"Served demand:     {total_served:,.0f}")
    print(f"Unmet demand:      {total_unmet:,.0f}")
    print(f"Service coverage:  {coverage_rate*100:5.1f}%")
    print(f"Capital used:      £{total_capex:,.0f} / £{BUDGET:,.0f}")
    print(f"Stations opened:   {num_stations_open}")

    picked_cols = ["station_id", "docks"]
    if "lat" in out.columns and "lon" in out.columns:
        picked_cols += ["lat", "lon"]
    picked = out.query("selected == 1")[picked_cols]
    print("\nSelected stations (first few rows):")
    print(picked.head())


if __name__ == "__main__":
    main()

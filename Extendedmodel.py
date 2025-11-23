import pandas as pd
import numpy as np
from pathlib import Path
import math
import pulp as pl



script_dir = Path(__file__).parent.resolve()
outputs_dir = script_dir / "outputs"


BASELINE_CSV = outputs_dir / "optimal_location_baseline.csv"
IMBAL_CSV    = outputs_dir / "station_imbalance.csv"


OUT_FLOW_CSV  = outputs_dir / "extended_rebalancing_flows.csv"
OUT_KPI_CSV   = outputs_dir / "extended_kpis.csv"
OUT_TRACE_CSV = outputs_dir / "extended_inventory_trace.csv"


ALL_T = ["AM", "OP", "PM"]
T_SEQ = {0: "AM", 1: "OP", 2: "PM"}

C_MOVE = 0.50    
LAMBDA = 5.00    
W_MAX  = 200.0   
FLEET_RATIO = 0.60 

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def main():
    print(">>> [Step 1] Loading Baseline Infrastructure...")
    
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Missing file: {BASELINE_CSV}")
    if not IMBAL_CSV.exists():
        raise FileNotFoundError(f"Missing file: {IMBAL_CSV}")

    base = pd.read_csv(BASELINE_CSV)
    base["station_id"] = base["station_id"].astype(str)
    
    sel = base[base["selected"] == 1].copy()
    if sel.empty: raise ValueError("No selected stations found in baseline!")
    
    J = sorted(sel["station_id"].unique())
    
    
    if "docks" in sel.columns:
        print("    Using optimized 'docks' column.")
        capacity = sel.set_index("station_id")["docks"].to_dict()
    else:
        print("    Using raw 'capacity' column.")
        capacity = sel.set_index("station_id")["capacity"].to_dict()

    lat = sel.set_index("station_id")["lat"].to_dict()
    lon = sel.set_index("station_id")["lon"].to_dict()
    name = sel.set_index("station_id")["name"].to_dict() if "name" in sel.columns else {j:j for j in J}

    total_docks = sum(capacity.values())
    B_FLEET = int(total_docks * FLEET_RATIO)
    print(f"    Total Docks: {total_docks}")
    print(f"    Fixed Fleet Size (B_fleet): {B_FLEET} (60% Ratio)")

    
    print(">>> [Step 2] Loading Demand Patterns...")
    try:
        
        imb_df = pd.read_csv(IMBAL_CSV, encoding='utf-16')
    except:
        
        print("    UTF-16 read failed, switching to default encoding...")
        imb_df = pd.read_csv(IMBAL_CSV)
    

    imb_df["station_id"] = imb_df["station_id"].astype(str)
    
    grid = pd.MultiIndex.from_product([J, ALL_T], names=["station_id", "period"]).to_frame(index=False)
    data = grid.merge(imb_df, on=["station_id", "period"], how="left").fillna(0)

    demand = {}
    arrivals = {}
    for _, row in data.iterrows():
        net = row["imbalance"]
        if net < 0:
            demand[(row.station_id, row.period)] = abs(net)
            arrivals[(row.station_id, row.period)] = 0
        else:
            demand[(row.station_id, row.period)] = 0
            arrivals[(row.station_id, row.period)] = abs(net)

    print(">>> [Step 3] Computing Distances...")
    dist_matrix = {}
    for j1 in J:
        for j2 in J:
            if j1 != j2:
                dist_matrix[(j1, j2)] = haversine_km(lat[j1], lon[j1], lat[j2], lon[j2])

    print(">>> [Step 4] Building Optimization Model...")
    prob = pl.LpProblem("Extended_Model", pl.LpMinimize)

    # Variables
    # t ranges: 0=AM, 1=OP, 2=PM, 3=End of day
    b = pl.LpVariable.dicts("Inventory", ((j, t) for j in J for t in [0,1,2,3]), lowBound=0, cat=pl.LpInteger)
    z = pl.LpVariable.dicts("Rebalance", ((j1, j2, t) for j1 in J for j2 in J if j1!=j2 for t in [0,1,2]), lowBound=0, cat=pl.LpInteger)
    u = pl.LpVariable.dicts("Unmet", ((j, t) for j in J for t in [0,1,2]), lowBound=0, cat=pl.LpContinuous)

    # Objective: Minimize (Cost + Penalty)
    prob += pl.lpSum(C_MOVE * dist_matrix[(j1,j2)] * z[(j1,j2,t)] for j1 in J for j2 in J if j1!=j2 for t in [0,1,2]) + \
            pl.lpSum(LAMBDA * u[(j,t)] for j in J for t in [0,1,2])

    # Constraints
    # 1. Initial Distribution (Proportional to capacity)
    current_assigned = 0
    for i, j in enumerate(J):
        if i == len(J) - 1:
            target_b = B_FLEET - current_assigned 
        else:
            target_b = int(capacity[j] * FLEET_RATIO)
            current_assigned += target_b
        
        prob += b[(j, 0)] == target_b, f"Init_Stock_{j}"

    # 2. Daily Constraints
    for t in [0, 1, 2]:
        p_name = T_SEQ[t]
        
        # Work Budget (Eq 16)
        prob += pl.lpSum(dist_matrix[(j1,j2)] * z[(j1,j2,t)] for j1 in J for j2 in J if j1!=j2) <= W_MAX, f"Budget_{t}"

        for j in J:
            # Capacity Bounds
            prob += b[(j, t)] <= capacity[j], f"Cap_{j}_{t}"
            
            # Demand Fulfillment Logic
            # Inventory used = Demand - Unmet
            prob += b[(j, t)] >= demand[(j, p_name)] - u[(j, t)], f"Stock_Serve_{j}_{t}"
            prob += u[(j, t)] <= demand[(j, p_name)], f"Unmet_Limit_{j}_{t}"

            # Flow Balance (Eq 13)
            rebal_in  = pl.lpSum(z[(k, j, t)] for k in J if k!=j)
            rebal_out = pl.lpSum(z[(j, k, t)] for k in J if k!=j)
            
            served_demand = demand[(j, p_name)] - u[(j, t)]
            
            prob += b[(j, t+1)] == b[(j, t)] - served_demand + arrivals[(j, p_name)] + rebal_in - rebal_out

    print(">>> [Step 5] Solving with CBC Solver...")
    prob.solve(pl.PULP_CBC_CMD(msg=1))
    status = pl.LpStatus[prob.status]
    print(f"    Status: {status}")

    if status == "Optimal":
        # Export Flows
        flows = []
        for t in [0,1,2]:
            for j1 in J:
                for j2 in J:
                    if j1!=j2:
                        val = pl.value(z[(j1,j2,t)])
                        if val and val > 0.1:
                            flows.append({
                                "period": T_SEQ[t], "from_station": j1, "to_station": j2, 
                                "amount": int(val), "from_lat": lat[j1], "from_lon": lon[j1],
                                "to_lat": lat[j2], "to_lon": lon[j2]
                            })
        pd.DataFrame(flows).to_csv(OUT_FLOW_CSV, index=False)

        # Export Trace
        trace = []
        for t in [0,1,2]:
            for j in J:
                trace.append({
                    "station_id": j, "station_name": name[j], "capacity": capacity[j],
                    "period": T_SEQ[t], "period_idx": t,
                    "inventory_start": pl.value(b[(j, t)]),
                    "unmet_demand": pl.value(u[(j, t)]),
                    "demand": demand[(j, T_SEQ[t])]
                })
        # End State
        for j in J:
             trace.append({
                    "station_id": j, "station_name": name[j], "capacity": capacity[j],
                    "period": "END", "period_idx": 3, 
                    "inventory_start": pl.value(b[(j, 3)]), "unmet_demand": 0, "demand": 0
             })
        
        pd.DataFrame(trace).to_csv(OUT_TRACE_CSV, index=False)
        print(f"✅ SUCCESS! Data saved to:\n  {OUT_FLOW_CSV}\n  {OUT_TRACE_CSV}")
        print("\nYou can now run the visualisation script!")
    else:
        print("❌ Optimization failed or infeasible.")

if __name__ == "__main__":
    main()
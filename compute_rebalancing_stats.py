import pandas as pd
from pathlib import Path


BASELINE_DAILY_OBJ = -127_313.89  
CAPEX = 822_500.0                  
YEARS = 5
DAYS_PER_YEAR = 365

script_dir = Path(__file__).parent.resolve()
REBAL_SUMMARY_CSV = script_dir / "outputs" / "rebalancing_summary_stats.csv"
OUT_5Y_CSV        = script_dir / "outputs" / "five_year_objective_summary.csv"


def main():
    
    stats = pd.read_csv(REBAL_SUMMARY_CSV)
    rebal_daily_cost = float(stats["total_cost"].iloc[0])   

    days = YEARS * DAYS_PER_YEAR

    
    baseline_daily = BASELINE_DAILY_OBJ
    extended_daily = baseline_daily - rebal_daily_cost

    
    baseline_5y_oper = baseline_daily * days
    extended_5y_oper = extended_daily * days

    
    rebal_5y_cost = rebal_daily_cost * days

    
    baseline_5y_total = baseline_5y_oper - CAPEX
    extended_5y_total = extended_5y_oper - CAPEX

    print("\n=========== 5-year Objective Summary ===========")
    print(f"Days considered:                   {days} ( {YEARS} years )")
    print("-----------------------------------------------")
    print(f"Baseline daily objective:          £{baseline_daily:,.2f}")
    print(f"Extended daily objective:          £{extended_daily:,.2f}")
    print("-----------------------------------------------")
    print(f"Baseline 5-year oper. objective:   £{baseline_5y_oper:,.2f}")
    print(f"Extended 5-year oper. objective:   £{extended_5y_oper:,.2f}")
    print(f"5-year rebalancing cost total:     £{rebal_5y_cost:,.2f}")
    print("-----------------------------------------------")
    print(f"Baseline 5-year total objective:   £{baseline_5y_total:,.2f}")
    print(f"Extended 5-year total objective:   £{extended_5y_total:,.2f}")
    print(f"Difference (Extended - Baseline):  £{extended_5y_total - baseline_5y_total:,.2f}")
    print("===============================================\n")

    
    out = pd.DataFrame({
        "metric": [
            "baseline_daily_objective",
            "extended_daily_objective",
            "baseline_5y_oper_objective",
            "extended_5y_oper_objective",
            "rebalancing_5y_cost",
            "baseline_5y_total_objective",
            "extended_5y_total_objective",
            "extended_minus_baseline_5y_total"
        ],
        "value": [
            baseline_daily,
            extended_daily,
            baseline_5y_oper,
            extended_5y_oper,
            rebal_5y_cost,
            baseline_5y_total,
            extended_5y_total,
            extended_5y_total - baseline_5y_total
        ]
    })

    OUT_5Y_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_5Y_CSV, index=False)
    print(f"5-year objective summary saved to: {OUT_5Y_CSV}")


if __name__ == "__main__":
    main()

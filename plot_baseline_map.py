import pandas as pd
from pathlib import Path
import folium


script_dir = Path(__file__).parent.resolve()
RESULT_CSV = script_dir / "outputs" / "optimal_location_baseline.csv"
OUT_HTML   = script_dir / "outputs" / "baseline_selected_stations_map.html"

def main():
    
    df = pd.read_csv(RESULT_CSV)

    sel = df[df["selected"] == 1].copy()
    if sel.empty:
        raise ValueError("No selected stations found in the result file.")

    
    center_lat = sel["lat"].mean()
    center_lon = sel["lon"].mean()

    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    for _, row in sel.iterrows():
        popup_html = (
            f"Station: {row['name']}<br>"
            f"ID: {row['station_id']}<br>"
            f"Docks: {row['docks']}"
        )
        popup = folium.Popup(popup_html, max_width=250)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5,
            fill=True,
            tooltip=row["name"],      
            popup=popup               
        ).add_to(m)

   
    OUT_HTML.parent.mkdir(exist_ok=True)
    m.save(str(OUT_HTML))
    print(f"Saved interactive map to: {OUT_HTML}")

if __name__ == "__main__":
    main()

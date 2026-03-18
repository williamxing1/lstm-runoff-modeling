from pathlib import Path
import pandas as pd

coord_path = Path("/Volumes/ml_ssd/lstm-runoff-modeling/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt")

rows = []

with open(coord_path) as f:
    line1 = True
    for line in f:
        if line1:
            line1 = False
            continue
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        huc = parts[0]
        gauge_id = parts[1]
        lat = float(parts[-3])
        lon = float(parts[-2])
        area = float(parts[-1])
        name = " ".join(parts[2:-3])

        rows.append([huc, gauge_id, name, lat, lon, area])

df = pd.DataFrame(rows, columns=[
    "HUC", "gauge_id", "name", "lat", "lon", "drainage_area"
])

df.to_csv("basin_metadata.csv", index=False)
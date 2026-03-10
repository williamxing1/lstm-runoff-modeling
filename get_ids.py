from pathlib import Path

path = Path("basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2/basin_mean_forcing/daymet")

files = list(path.glob("**/*.txt"))
basin_ids = sorted([file.name.split("_")[0] for file in files])
print(len(basin_ids))
print(basin_ids[:10])
print(len(basin_ids) == len(set(basin_ids)))
with open("basin_ids.txt", "w") as f:
    for basin_id in basin_ids:
        f.write(basin_id + "\n")
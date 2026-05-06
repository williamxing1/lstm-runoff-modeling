from pathlib import Path
import pandas as pd

data_path = Path("../data/raw")

for file in data_path.glob("*"):
    if file.stem == "gauge_information":
        continue
    df = pd.read_csv(file, sep=";", dtype={"gauge_id": str})
    df.to_csv(f"../data/cleaned/{file.stem}.csv", index=False)
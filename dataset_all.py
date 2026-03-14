from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

# /data for modal only
root = Path("/data/basin_timeseries_v1p2_metForcing_obsFlow/basin_dataset_public_v1p2")

def load_discharge(basin_id: str, area):
    path = root / "usgs_streamflow"
    files = list(path.glob(f"**/{basin_id}_*.txt"))
    if len(files) > 1:
        print(f"More than one file found for basin {basin_id}")
    file_path = files[0]
    col_names = ["basin_id", "Year", "Mnth", "Day", "QObs", "Flag"]
    with open(file_path, "r") as f:
        df = pd.read_csv(f, sep="\s+", names=col_names)
    dates = df["Year"].astype(str) + "/" + df["Mnth"].astype(str) + "/" + df["Day"].astype(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    df = df.drop(columns=["Year", "Mnth", "Day"])
    df["QObs"] = df["QObs"].astype(float) * 28316846.592 * 86400 / (area * (10 ** 6)) # Cubic mm
    return df["QObs"]

def load_forcing(basin_id: str):
    path = root / "basin_mean_forcing" / "daymet"
    files = list(path.glob(f"**/{basin_id}_*.txt"))
    if len(files) > 1:
        print(f"More than one file found for basin {basin_id}")
    file_path = files[0]

    with open(file_path, "r") as f:
        content = f.readlines()
        area = int(content[2])
    
    df = pd.read_csv(file_path, sep='\s+', header=3)
    dates = df["Year"].astype(str) + "/" + df["Mnth"].astype(str) + "/" + df["Day"].astype(str)
    df.index = pd.to_datetime(dates, format="%Y/%m/%d")
    df = df.drop(columns=["Year", "Mnth", "Day", "Hr"])

    return df, area

def reshape_data(x, y, seq_length, pred_length):
    num_samples, num_features = x.shape
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(x_new.shape[0]):
        x_new[i, :, :] = x[i:i+seq_length, :]
        y_new[i, :] = y[i+seq_length-1]

    return x_new, y_new

class CamelsTXT(Dataset):
    def __init__(self, split, seq_length, dates, x_means=None, x_stds=None, y_means=None, y_stds=None, basin_count=100):
        self.dates = dates
        self.seq_length = seq_length
        self.split = split
        self.x_means = x_means
        self.x_stds = x_stds
        self.y_means = y_means
        self.y_stds = y_stds
        self.basin_count = basin_count
        self.x, self.y = self._load_data()

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def _load_data(self):
        x_list = []
        y_list = []
        with open("/data/basin_ids.txt", "r") as f:
            basins = f.readlines()
        index = 0
        for basin_id in basins:
            index += 1
            if index % 25 == 0:
                print(f"Index: {index}")
            if index > self.basin_count:
                break
            basin_id = basin_id.strip()
            try:
                df, area = load_forcing(basin_id)
            except:
                continue
            df["QObs"] = load_discharge(basin_id, area)
            if self.dates is not None:
                if self.dates[0] - pd.DateOffset(days=self.seq_length) > df.index[0]:
                    start_date = self.dates[0] - pd.DateOffset(days=self.seq_length)
                else:
                    start_date = self.dates[0]
                df = df[start_date:self.dates[1]]
            
            x = df.drop(columns=["QObs"]).to_numpy()
            y = df["QObs"].to_numpy()

            x, y = reshape_data(x, y, self.seq_length, 1)

            if self.split == "train":
                mask = (~np.isnan(y).flatten()) & (y.flatten() >= 0)
                x = x[mask]
                y = y[mask]

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            x_list.append(x)
            y_list.append(y)

        x_final = torch.cat(x_list)
        y_final = torch.cat(y_list)

        if self.split == "train":
            x_mean = x_final.mean(dim=(0,1), keepdim=True)
            x_std = x_final.std(dim=(0,1), keepdim=True)
            x_std = x_std.clamp(min=1e-6)
            y_mean = y_final.mean()
            y_std = y_final.std()
            y_std = y_std.clamp(min=1e-6)

            self.x_means = x_mean
            self.x_stds = x_std
            self.y_means = y_mean
            self.y_stds = y_std

        x_final = self._local_normalization(x_final, "input", True)
        y_final = self._local_normalization(y_final, "output", True)
        print(f"X shape: {x_final.shape}")
        print(f"Y shape: {y_final.shape}")
        # x_final: dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
        # y_final: QObs
        
        return x_final, y_final

    def _local_normalization(self, df, variable, norm):
        if variable == "input":
            means = self.x_means
            stds = self.x_stds
        elif variable == "output":
            means = self.y_means
            stds = self.y_stds
        if norm:
            feature = (df - means) / stds
        else:
            feature = (df * stds) + means
        return feature

    def get_means(self):
        return self.x_means, self.y_means
    def get_stds(self):
        return self.x_stds, self.y_stds
"""
1. Load data
2. Use dates to see if there's past info
3. Norm if input
4. Reshape data
5. Delete all bad values if train
6. Write an unnorm function
"""
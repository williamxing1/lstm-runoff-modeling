# Removed id 03448942
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import json

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

def reshape_data(x, y, seq_length):
    num_samples, num_features = x.shape
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(x_new.shape[0]):
        x_new[i, :, :] = x[i:i+seq_length, :]
        y_new[i, :] = y[i+seq_length-1]

    return x_new, y_new

def reshape_data_intermediate(x, y, seq_length):
    num_samples, num_features = x.shape
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, seq_length, 1))
    
    for i in range(x_new.shape[0]):
        x_new[i, :, :] = x[i:i+seq_length, :]
        y_new[i, :, :] = y[i:i+seq_length, :]
    
    return x_new, y_new

class CamelsTXT(Dataset):
    def __init__(self, split, seq_length, huc, dates, x_means=None, x_stds=None, y_means=None, y_stds=None, add_static_features=False, use_intermediate=False):
        self.split = split
        self.seq_length = seq_length
        self.huc = huc
        self.dates = dates
        self.x_means = x_means
        self.x_stds = x_stds
        self.y_means = y_means or {}
        self.y_stds = y_stds or {}
        self.add_static_features = add_static_features
        self.use_intermediate = use_intermediate
        self.x, self.y, self.basin_ids = self._load_data()

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.basin_ids[idx]
    def _load_data(self):
        print(f"Loading data for HUC {self.huc}, {self.split} split")
        x_list = []
        y_list = []
        basin_list = []

        metadata = pd.read_csv("/data/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id", drop=False)
        if self.huc != 0:
            metadata = metadata[metadata["HUC"] == self.huc]
        basins = metadata["gauge_id"].tolist()
        static_features = pd.read_csv("/data/merged_static_features.csv", dtype={"gauge_id": str}).set_index("gauge_id", drop=True)
        index = 0
        for basin_id in basins:
            basin_id = basin_id.strip()
            try: # Some basins don't have metadata for some reason
                row = metadata.loc[basin_id]
            except:
                continue
            index += 1
            if index % 10 == 0:
                print(f"Finished basin {index}")
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

            if self.use_intermediate:
                x, y = reshape_data_intermediate(x, y, self.seq_length) # x: (num_samples, seq_length, num_features); y: (num_samples, seq_length, 1)
            else:
                x, y = reshape_data(x, y, self.seq_length) # x: (num_samples, seq_length, num_features); y: (num_samples, 1)
            if self.add_static_features:
                static_vals = static_features.loc[basin_id].to_numpy(dtype=np.float32)
                static_vals = static_vals.reshape(1, 1, -1)
                static_expanded = np.tile(static_vals, (x.shape[0], self.seq_length, 1))
                x = np.concatenate([x, static_expanded], axis=2)
            
            if self.use_intermediate:
                nan_count = (np.isnan(y)).all(axis=(1,2)).sum()
                negative_count = (y < 0).all(axis=(1,2)).sum()
                print(f"NaN: {nan_count}, Negative: {negative_count}, Total: {nan_count + negative_count}")

                mask = (~np.isnan(y)).all(axis=(1,2)) & (y >= 0).all(axis=(1,2))
            else:
                nan_count = np.isnan(y).sum()
                negative_count = (y < 0).sum()
                print(f"NaN: {nan_count}, Negative: {negative_count}, Total: {nan_count + negative_count}")

                mask = (~np.isnan(y).flatten()) & (y.flatten() >= 0)
            x = x[mask]
            y = y[mask]

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            if self.split == "train":
                y_mean = y.mean()
                y_std = y.std().clamp(min=1e-6)
                self.y_means[basin_id] = y_mean
                self.y_stds[basin_id] = y_std
            else:
                y_mean = self.y_means[basin_id]
                y_std = self.y_stds[basin_id]

            y = (y - y_mean) / y_std
            print(y.shape)
            
            x_list.append(x)
            y_list.append(y)
            num_samples = x.shape[0]
            basin_list.extend([basin_id] * num_samples)

        x_final = torch.cat(x_list)
        y_final = torch.cat(y_list)

        if self.split == "train":
            x_means = x_final.mean(dim=(0,1), keepdim=True)
            x_stds = x_final.std(dim=(0,1), keepdim=True).clamp(min=1e-6)
            self.x_means = x_means
            self.x_stds = x_stds
        else:
            x_means = self.x_means
            x_stds = self.x_stds
        
        x_final = (x_final - x_means) / x_stds

        print(f"X shape: {x_final.shape}")
        print(f"Y shape: {y_final.shape}")

        # x_final: dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)
        # y_final: QObs
        
        basin_array = np.array(basin_list)
        return x_final, y_final, basin_array

    def get_means(self):
        return self.x_means, self.y_means
    def get_stds(self):
        return self.x_stds, self.y_stds
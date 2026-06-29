from data_code.dataset import CamelsTXT
from models.lstm import LSTM, LSTM_NSE
from models.transformer import Transformer
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import os

class Config:
    # Transformer
    d_model: int = 64 # Used in RR-Former
    n_layers: int = 4 # Used in RR-Former
    n_heads: int = 4 # Used in RR-Former
    transformer_dropout: float = 0.1 # Used in RR-Former
    # LSTM
    lstm_hidden_size: int = 256 # 20 in 2018 paper, 256 in 2019 paper
    layers: int = 1 # 2 in 2018 paper, 1 in 2019 paper
    dropout: float = 0.4 # 0.1 in 2018 paper, 0.4 in 2019 paper
    # Data
    input_dim: int = 34 # 7 in 2018 paper (no static features), 34 in 2019 paper (with static features)
    seq_length: int = 270 # 365 in 2018 paper, 270 in 2019 paper, 22 for transformer
    add_static_features: bool = True # False in 2018 paper, True in 2019 paper
    if add_static_features:
        input_dim: int = 34
    else:
        input_dim: int = 7
    use_intermediate: bool = True # Must be False if using transformer
    # Training
    model: str = "lstm" # "transformer" or "lstm"
    if model == "transformer":
        use_intermediate = False
    loss_function: str = "nse" # MSE ("mse") in 2018 paper, NSE* ("nse") in 2019 paper
    batch_size: int = 256 # 512 in 2018 paper, 256 in 2019 paper
    learning_rate: float = 1e-3 # Used in 2018 paper and 2019 paper
    hucs: list[int] = [1, 3, 11, 17]
    # [1, 3, 11, 17] ([New England, South Atlantic-Gulf, Arkansas-White-Red, Pacific Northwest]) in 2018 paper,
    # [0] (Use all HUCs) in 2019 paper, although this is super slow
    epochs: int = 20 # 20 epochs per HUC in 2018 paper, 30 in total in 2019 paper
    train_start: str = "1980-10-01" # 1980-10-01 in 2018 paper, 1999-10-01 in 2019 paper
    train_end: str = "1995-09-30" # 1995-09-30 in 2018 paper, 2008-09-30 in 2019 paper
    val_start: str = "1995-10-01" # 1995-10-01 in 2018 paper, 1989-10-01 in 2019 paper
    val_end: str = "2014-12-31" # 2014-12-31 (Last date in dataset) in 2018 paper, 1999-09-30 in 2019 paper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
config = Config()

def calc_nse(obs, pred):
    mask = (~np.isnan(pred).flatten())
    obs = obs[mask]
    pred = pred[mask]
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    return 1 - (numerator / denominator)

# No epsilon is used when dividing by standard deviation since in the dataset_test.py, std is already clamped
class NSELoss(nn.Module):
    def __init__(self, y_stds):
        super().__init__()
        self.y_stds = y_stds
    def forward(self, y_pred, y_true, basin_array): # y_pred, y_true: (B, seq_length, 1); basin_array: # (B,); y_std contains a dictionary of {basin_id: standard_deviation}
        y_pred = y_pred.squeeze(-1) # y_pred: (B, seq_length)
        y_true = y_true.squeeze(-1) # y_true: (B, seq_length)
        squared_errors = (y_pred - y_true) ** 2 # (B, seq_length)
        basin_losses = []
        basin_array = np.array(basin_array)
        for basin in np.unique(basin_array):
            mask = (basin_array == basin)
            mask = torch.tensor(mask, device=y_pred.device)

            se_basin = squared_errors[mask] # (num_samples_b, seq_length)
            total_se = se_basin.sum()
            basin_y_std = self.y_stds[basin]
            denom = basin_y_std ** 2
            basin_loss = total_se / denom
            basin_losses.append(basin_loss)
        
        basin_losses = torch.stack(basin_losses)
        return basin_losses.mean()

json_path = "/outputs/index.json"
with open(json_path, "r") as f:
    data = json.load(f)
folder_path = "model" + str(data["index"])
config_str=f"""
add_static_features={config.add_static_features}
use_intermediate={config.use_intermediate}
loss_function={config.loss_function}
model={config.model}
"""
print(f"Folder: {folder_path}")
data["index"] += 1
with open(json_path, "w") as f:
    json.dump(data, f)
os.makedirs(f"/outputs/{folder_path}")
with open(f"/outputs/{folder_path}/config.txt", "w") as f:
    f.write(config_str)

for huc in config.hucs:
    gpu_settings = {"num_workers": 8, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4} if device == "cuda" else {}
    train_start = pd.to_datetime(config.train_start, format="%Y-%m-%d")
    train_end = pd.to_datetime(config.train_end, format="%Y-%m-%d")
    train_dataset = CamelsTXT(split="train", seq_length=config.seq_length, huc=huc, dates=[train_start, train_end], add_static_features=config.add_static_features, use_intermediate=config.use_intermediate)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **gpu_settings)

    val_start = pd.to_datetime(config.val_start, format="%Y-%m-%d")
    val_end = pd.to_datetime(config.val_end, format="%Y-%m-%d")
    x_mean, y_mean = train_dataset.get_means()
    x_std, y_std = train_dataset.get_stds()
    val_dataset = CamelsTXT(split="eval", seq_length=config.seq_length, huc=huc, dates=[val_start, val_end], x_means=x_mean, x_stds=x_std, y_means=y_mean, y_stds=y_std, add_static_features=config.add_static_features, use_intermediate=config.use_intermediate)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **gpu_settings)
    print(f"Train Length: {len(train_loader.dataset)}, Val Length: {len(val_loader.dataset)}")

    if config.model == "lstm":
        if config.use_intermediate:
            model = LSTM_NSE(config).to(device)
        else:
            model = LSTM(config).to(device)
    elif config.model == "transformer":
        model = Transformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if config.loss_function == "mse":
        criterion = nn.MSELoss()
    elif config.loss_function == "nse":
        criterion = NSELoss(y_std)
    scaler = torch.amp.GradScaler("cuda")
    best_val = 0
    train_losses = []
    val_losses = []

    for epoch in range(config.epochs):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for xb, yb, basin_array in train_loader:
            optimizer.zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                preds = model(xb)
                if config.loss_function == "mse":
                    loss = criterion(preds, yb)
                elif config.loss_function == "nse":
                    loss = criterion(preds, yb, basin_array)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        for xb, yb, basin_array in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                preds = model(xb)
                if config.loss_function == "mse":
                    loss = criterion(preds, yb)
                elif config.loss_function == "nse":
                    loss = criterion(preds, yb, basin_array)
            val_loss += loss.item() * xb.size(0)

            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        t1 = time.time()
        time_passed = t1 - t0
        val_nse = calc_nse(all_targets.flatten(), all_preds.flatten())
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val NSE: {val_nse}, Time: {time_passed:.2f} seconds")
        if val_nse > best_val:
            best_val = val_nse
            torch.save(model.state_dict(), f"/outputs/{folder_path}/best_model_huc_{huc}.pth")
            print("Weights saved")

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Train-Val Curve")
    plt.savefig(f"/outputs/{folder_path}/train_val_curve.png")
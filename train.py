from dataset_all import CamelsTXT
from models.lstm import LSTM
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
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    # LSTM
    lstm_hidden_size: int = 32
    layers: int = 2
    dropout: float = 0.2
    # Data
    input_dim: int = 7
    max_basins: int = 300
    seq_length: int = 365
    use_all: bool = True
    basin_limits: list[float] = [-81.0, 36.0, -75.0, 40.0] # west, south, east, north; Run basin_map.py to see the basins in this area
    # Training
    model: str = "transformer" # "transformer" or "lstm"
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 30

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
config = Config()

def calc_nse(obs, pred):
    mask = (pred.flatten() > 0) & (~np.isnan(pred).flatten())
    obs = obs[mask]
    pred = pred[mask]
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    return 1 - (numerator / denominator)

json_path = "/outputs/index.json"
with open(json_path, "r") as f:
    data = json.load(f)
folder_path = "model" + str(data["index"])
print(f"Folder: {folder_path}")
data["index"] += 1
with open(json_path, "w") as f:
    json.dump(data, f)
os.makedirs(f"/outputs/{folder_path}")

gpu_settings = {"num_workers": 8, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4} if device == "cuda" else {}
train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
train_dataset = CamelsTXT("train", config.seq_length, dates=[train_start, train_end], max_basins=config.max_basins, use_all=config.use_all, basin_limits=config.basin_limits)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **gpu_settings)

val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
x_mean, y_mean = train_dataset.get_means()
x_std, y_std = train_dataset.get_stds()
val_dataset = CamelsTXT("eval", config.seq_length, dates=[val_start, val_end], x_means=x_mean, x_stds=x_std, y_means=y_mean, y_stds=y_std, max_basins=config.max_basins, use_all=False, basin_limits=config.basin_limits)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **gpu_settings)
print(f"Train Length: {len(train_loader.dataset)}, Val Length: {len(val_loader.dataset)}")

if config.model == "lstm":
    model = LSTM(config).to(device)
elif config.model == "transformer":
    model = Transformer(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler("cuda")
best_val = float("inf")
train_losses = []
val_losses = []

for epoch in range(config.epochs):
    t0 = time.time()
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(xb)
            loss = criterion(preds, yb)
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

    for xb, yb in val_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(xb)
            loss = criterion(preds, yb)
        val_loss += loss.item() * xb.size(0)
        
        unnormed_preds = val_dataset._local_normalization(preds, "output", False)
        unnormed_yb = val_dataset._local_normalization(yb, "output", False)
        all_preds.append(unnormed_preds.detach().cpu().numpy())
        all_targets.append(unnormed_yb.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    t1 = time.time()
    time_passed = t1 - t0
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val NSE: {calc_nse(all_targets.flatten(), all_preds.flatten())}, Time: {time_passed:.2f} seconds")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), f"/outputs/{folder_path}/best_model.pth")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train-Val Curve")
plt.savefig(f"/outputs/{folder_path}/train_val_curve.png")
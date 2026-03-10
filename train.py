# Removed datasets:
# Remove basin 01150900
# Remove basin 02081113 - Area breaks




from dataset_all import CamelsTXT
from model import Model
import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import os
from pathlib import Path

class Config:
    basin_id: str = "01022500"
    hidden_size: int = 256
    dropout_rate: int = 0.0
    learning_rate: float = 1e-3
    sequence_length: int = 365
    input_dim: int = 7
    epochs: int = 30
    dropout: float = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu"
config = Config()

def calc_nse(obs, pred):
    mask = (pred.flatten() > 0) & (~np.isnan(pred).flatten())
    obs = obs[mask]
    pred = pred[mask]
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    return 1 - (numerator / denominator)

with open("outputs/index.json", "r") as f:
    data = json.load(f)
folder_path = "model" + str(data["index"])
data["index"] += 1
with open("outputs/index.json", "w") as f:
    json.dump(data, f)
os.makedirs(f"outputs/{folder_path}")

gpu_settings = {"num_workers": 8, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4} if device == "cuda" else {}
train_start = pd.to_datetime("1980-10-01", format="%Y-%m-%d")
train_end = pd.to_datetime("1995-09-30", format="%Y-%m-%d")
train_dataset = CamelsTXT("train", config.sequence_length, dates=[train_start, train_end])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, **gpu_settings)

val_start = pd.to_datetime("1995-10-01", format="%Y-%m-%d")
val_end = pd.to_datetime("2000-09-30", format="%Y-%m-%d")
x_mean, y_mean = train_dataset.get_means()
x_std, y_std = train_dataset.get_stds()
val_dataset = CamelsTXT("eval", config.sequence_length, dates=[val_start, val_end], x_means=x_mean, x_stds=x_std, y_means=y_mean, y_stds=y_std)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, **gpu_settings)

model = Model(config.input_dim, config.hidden_size, config.dropout).to(device)
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
            unnormed_preds = val_dataset._local_normalization(preds.detach().cpu().numpy(), "output", False)
            loss = criterion(preds, yb)
        val_loss += loss.item() * xb.size(0)

        all_preds.append(unnormed_preds)
        all_targets.append(yb.detach().cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    t1 = time.time()
    time_passed = t1 - t0
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val NSE: {calc_nse(all_targets.flatten(), all_preds.flatten())}, Time: {time_passed:.2f} seconds")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), f"outputs/{folder_path}/best_model.pth")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train-Val Curve")
plt.savefig(f"outputs/{folder_path}/train_val_curve.png")
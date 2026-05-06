import pandas as pd
import numpy as np
import torch
from torch import tensor
from torch.utils.data import DataLoader
from dataset import CamelsTXT
from lstm import LSTM, LSTM_NSE
from transformer import Transformer
import matplotlib.pyplot as plt

class Config:
    hucs: list[int] = [1, 3, 11, 17]
    train_start: str = "1980-10-01"
    train_end: str = "1995-09-30"
    val_start: str = "1995-10-01"
    val_end: str = "2014-12-31"
    batch_size: int = 256
    # LSTM
    lstm_hidden_size: int = 256
    layers: int = 1
    dropout: float = 0.4
    model: str = "lstm"
    seq_length: int = 270

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

basin_metadata = pd.read_csv("/data/basin_metadata.csv", dtype={"gauge_id": str})

folders_to_load_from = {
    1: {
        "baseline_folder": "model1",
        "intermediate_static_nse_folder": "model16",
        "static_nse_folder": "model15"
    },
    3: {
        "baseline_folder": "model1",
        "intermediate_static_nse_folder": "model6",
        "static_nse_folder": "model7"
    },
    11: {
        "baseline_folder": "model1",
        "intermediate_static_nse_folder": "model6",
        "static_nse_folder": "model7"
    },
    17: {
        "baseline_folder": "model1",
        "intermediate_static_nse_folder": "model16",
        "static_nse_folder": "model15"
    } 
}

nses = {
    "baseline_model": [],
    "intermediate_static_nse_model": [],
    "static_nse_model": []
}
for huc in config.hucs:
    for model_type_folder in ["baseline_folder", "intermediate_static_nse_folder", "static_nse_folder"]:
        print(f"Inference on HUC {huc} for folder {model_type_folder}")
        if model_type_folder == "baseline_folder":
            add_static_features = False
            use_intermediate = False
        elif model_type_folder == "intermediate_static_nse_folder":
            add_static_features = True
            use_intermediate = True
        elif model_type_folder == "static_nse_folder":
            add_static_features = True
            use_intermediate = False
        if add_static_features:
            config.input_dim = 34
        else:
            config.input_dim = 7
        train_start = pd.to_datetime(config.train_start, format="%Y-%m-%d")
        train_end = pd.to_datetime(config.train_end, format="%Y-%m-%d")
        train_dataset = CamelsTXT(split="train", seq_length=config.seq_length, huc=huc, dates=[train_start, train_end], add_static_features=add_static_features, use_intermediate=use_intermediate)
        
        val_start = pd.to_datetime(config.val_start, format="%Y-%m-%d")
        val_end = pd.to_datetime(config.val_end, format="%Y-%m-%d")
        x_mean, y_mean = train_dataset.get_means()
        x_std, y_std = train_dataset.get_stds()
        val_dataset = CamelsTXT(split="eval", seq_length=config.seq_length, huc=huc, dates=[val_start, val_end], x_means=x_mean, x_stds=x_std, y_means=y_mean, y_stds=y_std, add_static_features=add_static_features, use_intermediate=use_intermediate)
        
        gpu_settings = {"num_workers": 8, "pin_memory": True, "persistent_workers": True, "prefetch_factor": 4} if device == "cuda" else {}
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, **gpu_settings)
    
        if config.model == "lstm":
            if use_intermediate:
                model = LSTM_NSE(config).to(device)
            else:
                model = LSTM(config).to(device)
        elif config.model == "transformer":
            model = Transformer(config).to(device)
        folder = folders_to_load_from[huc][model_type_folder]
        weights = torch.load(f"/weights/{folder}/best_model_huc_{huc}.pth")
        model.load_state_dict(weights)
        model.eval()

        all_preds = []
        all_targets = []
        all_basins = []
        for xb, yb, basin_id in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16):
                preds = model(xb)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(yb.cpu().numpy())
            all_basins.extend(basin_id)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_basins = np.array(all_basins)

        huc_nses = []
        for basin in np.unique(all_basins):
            mask = (all_basins == basin)

            pred_b = all_preds[mask].flatten()
            obs_b = all_targets[mask].flatten()

            nse = calc_nse(obs_b, pred_b)
            huc_nses.append(nse)

        name = model_type_folder.removesuffix("_folder") + "_model"
        nses[name].extend(huc_nses)

plt.figure()
for model_type in ["baseline_model", "intermediate_static_nse_model", "static_nse_model"]:
    if model_type == "baseline_model":
        label = "Baseline LSTM (MSE Loss, No Int., No static)"
    elif model_type == "intermediate_static_nse_model":
        label = "LSTM + NSE Loss + Int. + Static"
    elif model_type == "static_nse_model":
        label = "LSTM + NSE Loss + Static"
         
    vals = np.array(nses[model_type])
    vals = np.sort(vals)
    cdf = np.arange(1, len(vals) + 1) / len(vals)

    plt.plot(vals, cdf, label=label)

plt.xlabel("NSE")
plt.ylabel("CDF")
plt.title("CDF of NSE across basins using LSTM")
plt.xlim(-0.25, 1.0)
plt.grid()
plt.legend()
plt.savefig(f"/outputs/nse_cdf.png")
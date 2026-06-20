"""
Fine-tune GNN for band-gap regression.
Compares Contrastive SSL, Masked SSL, and a random-init baseline.
"""

import os
import copy
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

from data_utils import fetch_materials, structure_to_pyg_data, augment_graph
from modelsjk import GNNEncoder, ProjectionHead, MaskedAtomDecoder, RegHead, nt_xent_loss, MASK_TOKEN_ID

warnings.filterwarnings("ignore")

# config
API_KEY = "insert your api key there"
TARGET_N = 3600
MAX_N_SITES = 80
SSL_EPOCHS = 50
FT_EPOCHS = 100
VAL_EVERY = 5
BATCH_SIZE = 32
SSL_BATCH_SIZE = 64
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# fetch data
entries = fetch_materials(API_KEY, target_n=TARGET_N, max_n_sites=MAX_N_SITES)
pyg_list = []
for entry in tqdm(entries, desc="converting to graphs"):
    if entry.band_gap is None:
        continue
    try:
        data = structure_to_pyg_data(entry.structure)
        data.band_gap = torch.tensor([float(entry.band_gap)], dtype=torch.float)
        pyg_list.append(data)
    except Exception:
        continue

# remove outliers
y_raw = np.array([g.band_gap.item() for g in pyg_list])
sane_max = max(15.0, np.percentile(y_raw, 99.5))
clean_graphs = [g for g, y in zip(pyg_list, y_raw) if y <= sane_max]
clean_y = np.array([g.band_gap.item() for g in clean_graphs])

# splits and scaling
indices = np.arange(len(clean_graphs))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=SEED)
train_idx, val_idx = train_test_split(train_idx, test_size=0.15, random_state=SEED)

scaler = RobustScaler().fit(clean_y[train_idx].reshape(-1, 1))
y_scaled = scaler.transform(clean_y.reshape(-1, 1)).flatten()

def make_dataset(idx_list):
    out = []
    for i in idx_list:
        g = clean_graphs[i].clone()
        g.y = torch.tensor([y_scaled[i]], dtype=torch.float)
        out.append(g)
    return out

train_dataset = make_dataset(train_idx)
val_dataset = make_dataset(val_idx)
test_dataset = make_dataset(test_idx)

ssl_loader = DataLoader(clean_graphs, batch_size=SSL_BATCH_SIZE, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_rmse(encoder, head, loader):
    encoder.eval()
    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = encoder(batch.x, batch.edge_index, batch.edge_attr)
            emb = global_mean_pool(h, batch.batch)
            preds.extend(head(emb).squeeze(-1).cpu().numpy())
            trues.extend(batch.y.cpu().numpy())
    p_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    t_inv = scaler.inverse_transform(np.array(trues).reshape(-1, 1))
    return np.sqrt(mean_squared_error(t_inv, p_inv))

def fine_tune(encoder, loader_train, loader_val, epochs=FT_EPOCHS, lr=5e-4, val_every=VAL_EVERY):
    head = RegHead().to(device)
    opt = optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    best_val_rmse = float("inf")
    best_epoch = 0
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    best_head_state = copy.deepcopy(head.state_dict())

    for epoch in range(epochs):
        encoder.train()
        head.train()
        ep_loss = 0.0
        for batch in loader_train:
            batch = batch.to(device)
            opt.zero_grad()
            h = encoder(batch.x, batch.edge_index, batch.edge_attr)
            preds = head(global_mean_pool(h, batch.batch)).squeeze(-1)
            loss = loss_fn(preds, batch.y)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        losses.append(ep_loss / len(loader_train))

        if (epoch + 1) % val_every == 0 or (epoch + 1) == epochs:
            val_rmse = evaluate_rmse(encoder, head, loader_val)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_epoch = epoch + 1
                best_encoder_state = copy.deepcopy(encoder.state_dict())
                best_head_state = copy.deepcopy(head.state_dict())

    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    return head, losses

# contrastive pretraining
print("starting contrastive pretraining...")
contrastive_encoder = GNNEncoder().to(device)
projection_head = ProjectionHead().to(device)
opt_cl = optim.Adam(list(contrastive_encoder.parameters()) + list(projection_head.parameters()), lr=1e-3)

contrastive_encoder.train()
projection_head.train()

for epoch in range(SSL_EPOCHS):
    total_loss = 0.0
    for batch in ssl_loader:
        opt_cl.zero_grad()
        graphs = batch.to_data_list()
        v1 = [augment_graph(g) for g in graphs]
        v2 = [augment_graph(g) for g in graphs]
        b1 = next(iter(DataLoader(v1, batch_size=len(v1)))).to(device)
        b2 = next(iter(DataLoader(v2, batch_size=len(v2)))).to(device)

        h1 = contrastive_encoder(b1.x, b1.edge_index, b1.edge_attr)
        h2 = contrastive_encoder(b2.x, b2.edge_index, b2.edge_attr)
        z1 = projection_head(global_mean_pool(h1, b1.batch))
        z2 = projection_head(global_mean_pool(h2, b2.batch))

        loss = nt_xent_loss(z1, z2)
        loss.backward()
        opt_cl.step()
        total_loss += loss.item()

# masked pretraining
print("starting masked pretraining...")
masked_encoder = GNNEncoder().to(device)
masked_decoder = MaskedAtomDecoder().to(device)
opt_mae = optim.Adam(list(masked_encoder.parameters()) + list(masked_decoder.parameters()), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()

masked_encoder.train()
masked_decoder.train()

for epoch in range(SSL_EPOCHS):
    total_loss = 0.0
    for batch in ssl_loader:
        batch = batch.to(device)
        opt_mae.zero_grad()

        x_orig = torch.clamp(batch.x.squeeze(-1).long(), 0, 100)
        num_nodes = x_orig.size(0)
        num_to_mask = max(1, int(0.15 * num_nodes))
        mask_idx = torch.randperm(num_nodes, device=device)[:num_to_mask]

        x_masked = batch.x.clone()
        x_masked[mask_idx] = float(MASK_TOKEN_ID)

        h = masked_encoder(x_masked, batch.edge_index, batch.edge_attr)
        preds = masked_decoder(h)

        loss = ce_loss(preds[mask_idx], x_orig[mask_idx])
        loss.backward()
        opt_mae.step()
        total_loss += loss.item()

# fine-tuning
print("fine-tuning contrastive model...")
ssl_cl_head, ssl_cl_losses = fine_tune(contrastive_encoder, train_loader, val_loader)

print("fine-tuning masked model...")
ssl_mae_head, ssl_mae_losses = fine_tune(masked_encoder, train_loader, val_loader)

print("training baseline...")
baseline_encoder = GNNEncoder().to(device)
baseline_head, baseline_losses = fine_tune(baseline_encoder, train_loader, val_loader)

# evaluation
print("evaluating...")
rmse_cl = evaluate_rmse(contrastive_encoder, ssl_cl_head, test_loader)
rmse_mae = evaluate_rmse(masked_encoder, ssl_mae_head, test_loader)
rmse_base = evaluate_rmse(baseline_encoder, baseline_head, test_loader)

results_df = pd.DataFrame([
    {"Model": "Contrastive", "Test RMSE (eV)": rmse_cl, "Improvement (%)": (rmse_base - rmse_cl) / rmse_base * 100},
    {"Model": "Masked", "Test RMSE (eV)": rmse_mae, "Improvement (%)": (rmse_base - rmse_mae) / rmse_base * 100},
    {"Model": "Baseline", "Test RMSE (eV)": rmse_base, "Improvement (%)": 0.0},
])

print("\n" + results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/regression_results.csv", index=False)

# plot
plt.figure(figsize=(8, 5))
plt.plot(ssl_cl_losses, label="Contrastive SSL")
plt.plot(ssl_mae_losses, label="Masked SSL")
plt.plot(baseline_losses, label="Random-Init Baseline", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Train MSE (scaled)")
plt.legend()
plt.savefig("outputs/regression_convergence.png", dpi=150, bbox_inches="tight")
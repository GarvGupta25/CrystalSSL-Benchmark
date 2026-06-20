"""
Compare Contrastive vs Masked SSL pretraining on a GNN backbone.

Outputs:
  - outputs/tsne_comparison.png (t-SNE colored by metal/non-metal)
  - outputs/benchmark_results.csv (downstream linear-probe metrics)
"""

import os
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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.manifold import TSNE

from data_utils import fetch_materials, structure_to_pyg_data, augment_graph
from modelsjk import GNNEncoder, ProjectionHead, MaskedAtomDecoder, nt_xent_loss, MASK_TOKEN_ID

warnings.filterwarnings("ignore")

# config
API_KEY = "insert your api key there"
TARGET_N = 3600
MAX_N_SITES = 80
SSL_EPOCHS = 50
BATCH_SIZE = 32
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

loader = DataLoader(pyg_list, batch_size=BATCH_SIZE, shuffle=True)

# contrastive pretraining
print("starting contrastive pretraining...")
contrastive_encoder = GNNEncoder().to(device)
projection_head = ProjectionHead().to(device)
opt_cl = optim.Adam(list(contrastive_encoder.parameters()) + list(projection_head.parameters()), lr=1e-3)

contrastive_encoder.train()
projection_head.train()

for epoch in range(SSL_EPOCHS):
    total_loss = 0.0
    for batch in loader:
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

    print(f"epoch {epoch+1}/{SSL_EPOCHS} | loss: {total_loss/len(loader):.4f}")

# masked pretraining
print("\nstarting masked pretraining...")
masked_encoder = GNNEncoder().to(device)
masked_decoder = MaskedAtomDecoder().to(device)
opt_mae = optim.Adam(list(masked_encoder.parameters()) + list(masked_decoder.parameters()), lr=1e-3)
ce_loss = nn.CrossEntropyLoss()

masked_encoder.train()
masked_decoder.train()

for epoch in range(SSL_EPOCHS):
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt_mae.zero_grad()

        x_orig = torch.clamp(batch.x.squeeze(-1).long(), 0, 100)
        num_nodes = x_orig.size(0)
        num_mask = max(1, int(0.15 * num_nodes))

        mask_idx = torch.randperm(num_nodes, device=device)[:num_mask]
        x_masked = batch.x.clone()
        x_masked[mask_idx] = float(MASK_TOKEN_ID)

        h = masked_encoder(x_masked, batch.edge_index, batch.edge_attr)
        preds = masked_decoder(h)

        loss = ce_loss(preds[mask_idx], x_orig[mask_idx])
        loss.backward()
        opt_mae.step()

        total_loss += loss.item()

    print(f"epoch {epoch+1}/{SSL_EPOCHS} | loss: {total_loss/len(loader):.4f}")

# downstream evaluation
print("\nevaluating embeddings...")
contrastive_encoder.eval()
masked_encoder.eval()

eval_loader = DataLoader(pyg_list, batch_size=BATCH_SIZE, shuffle=False)
cl_embs, mae_embs, band_gaps = [], [], []

with torch.no_grad():
    for batch in eval_loader:
        batch = batch.to(device)

        h_cl = contrastive_encoder(batch.x, batch.edge_index, batch.edge_attr)
        cl_embs.append(global_mean_pool(h_cl, batch.batch).cpu())

        h_mae = masked_encoder(batch.x, batch.edge_index, batch.edge_attr)
        mae_embs.append(global_mean_pool(h_mae, batch.batch).cpu())

        band_gaps.extend([g.band_gap.item() for g in batch.to_data_list()])

X_cl = torch.cat(cl_embs, dim=0).numpy()
X_mae = torch.cat(mae_embs, dim=0).numpy()
y = (np.array(band_gaps) > 0).astype(int) 

def evaluate_embeddings(X, y, name, seed=SEED):
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y)
    
    scaler = StandardScaler().fit(X[idx_train])
    X_scaled = scaler.transform(X)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_scaled[idx_train], y[idx_train])
    acc = accuracy_score(y[idx_test], clf.predict(X_scaled[idx_test]))

    km = KMeans(n_clusters=2, random_state=seed, n_init=10).fit(X_scaled)
    
    return {
        "method": name,
        "acc": acc,
        "ari": adjusted_rand_score(y, km.labels_),
        "nmi": normalized_mutual_info_score(y, km.labels_),
        "silhouette": silhouette_score(X_scaled, km.labels_),
    }

results_df = pd.DataFrame([
    evaluate_embeddings(X_cl, y, "contrastive"),
    evaluate_embeddings(X_mae, y, "masked"),
])

print(results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/benchmark_results.csv", index=False)

# tsne viz
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30)
cl_2d = tsne.fit_transform(X_cl)
mae_2d = tsne.fit_transform(X_mae)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc1 = axes[0].scatter(cl_2d[:, 0], cl_2d[:, 1], c=y, cmap="coolwarm", s=15, alpha=0.7)
axes[0].set_title("Contrastive")
axes[1].scatter(mae_2d[:, 0], mae_2d[:, 1], c=y, cmap="coolwarm", s=15, alpha=0.7)
axes[1].set_title("Masked")

handles, _ = sc1.legend_elements(prop="colors", alpha=0.7)
fig.legend(handles, ["Metal", "Non-Metal"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig("outputs/tsne_comparison.png", dpi=150, bbox_inches="tight")
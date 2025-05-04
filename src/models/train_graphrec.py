# src/models/train_graphrec.py
"""
Treina o GraphRec adaptado offline.
Necessita:
    data/processed/combined_features.npy
    data/logs/interactions.csv
Suporta CPU ou GPU automaticamente.
"""
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.datasets import GraphRecDataset
from src.models.graphrec import GraphRec

# ----- paths -----
ROOT  = Path(__file__).resolve().parents[2]
FEATS = ROOT / "data" / "processed" / "combined_features.npy"
LOGS  = ROOT / "data" / "logs" / "interactions.csv"
CKPT  = ROOT / "data" / "models" / "graphrec.pth"
CKPT.parent.mkdir(parents=True, exist_ok=True)

# ----- device -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"▶ usando dispositivo: {device}")

# ----- dataset & loader -----
ds = GraphRecDataset(FEATS, LOGS, k_social=10)
loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=True)

# ----- modelo -----
D_ITEM = ds.features.shape[1]
model  = GraphRec(d_item=D_ITEM).to(device)
optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

EPOCHS = 5
for epoch in range(1, EPOCHS+1):
    total_loss = 0.0
    model.train()
    for batch in loader:
        item_idx = batch["item_idx"].to(device)         # (B,)
        rating   = batch["rating"].to(device)           # (B,)
        neigh    = batch["neigh_idxs"].to(device)       # (B, K)

        # monta tensores de features
        all_feats = torch.from_numpy(ds.features).float().to(device)
        target_feats = all_feats[item_idx]               # (B, D)
        neigh_feats  = all_feats[neigh]                  # (B, K, D)

        # histórico de 1 interação (B,1,D) e opinião (B,1,1)
        item_hist = target_feats.unsqueeze(1)            # (B, 1, D)
        opinions  = rating.view(-1, 1, 1)                # (B, 1, 1)

        optim.zero_grad()
        logits = model(item_hist, opinions, target_feats, neigh_feats)
        loss = loss_fn(logits, rating)
        loss.backward()
        optim.step()
        total_loss += loss.item() * rating.size(0)

    avg_loss = total_loss / len(ds)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

# salva checkpoint
torch.save(model.state_dict(), CKPT)
print("✅ modelo salvo em", CKPT)

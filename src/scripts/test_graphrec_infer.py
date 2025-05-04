# src/scripts/test_graphrec_infer.py

import sys
from pathlib import Path
import torch
import numpy as np

# Ajusta path do projeto para imports absolutos
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.datasets import GraphRecDataset
from src.models.graphrec import GraphRec

def main():
    # Paths
    FEATS = ROOT / "data" / "processed" / "combined_features.npy"
    LOGS  = ROOT / "data" / "logs" / "interactions.csv"
    CKPT  = ROOT / "data" / "models" / "graphrec.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Usando dispositivo: {device}")

    # Carrega dataset e pega uma amostra
    ds = GraphRecDataset(FEATS, LOGS, k_social=10)
    sample = ds[0]
    item_idx = sample["item_idx"].unsqueeze(0)      # shape (1,)
    rating   = sample["rating"].unsqueeze(0)        # shape (1,)
    neigh    = sample["neigh_idxs"].unsqueeze(0)    # shape (1, K)

    # Constrói tensor de features completo em device
    all_feats = torch.from_numpy(ds.features).float().to(device)  # (N, D)
    target_feats = all_feats[item_idx]                            # (1, D)
    neigh_feats  = all_feats[neigh]                               # (1, K, D)
    item_hist    = target_feats.unsqueeze(1)                      # (1, 1, D)
    opinions     = rating.view(-1, 1, 1).to(device)               # (1, 1, 1)

    # Inicializa modelo e carrega checkpoint
    model = GraphRec(d_item=ds.features.shape[1]).to(device)
    state = torch.load(CKPT, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Inferência
    with torch.no_grad():
        logit = model(item_hist, opinions, target_feats, neigh_feats)
        prob  = torch.sigmoid(logit).item()

    print(f"Item idx: {item_idx.item()}   Rating real: {rating.item()}")
    print(f"Logit de saída   : {logit.item():.4f}")
    print(f"Probabilidade de like: {prob:.4f}")

if __name__ == "__main__":
    main()

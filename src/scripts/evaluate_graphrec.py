# src/scripts/evaluate_graphrec.py

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# Ajusta sys.path para importar src.*
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

    # Carrega dataset e modelo
    ds = GraphRecDataset(FEATS, LOGS, k_social=10)
    model = GraphRec(d_item=ds.features.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # Containers
    y_true = []
    y_prob = []

    # Itera todas as interações
    for sample in ds:
        idx = sample["item_idx"].unsqueeze(0).to(device)    # (1,)
        r   = sample["rating"].item()                      # float 0.0/1.0
        neigh = sample["neigh_idxs"].unsqueeze(0).to(device)  # (1,K)

        # Monta tensores
        all_feats    = torch.from_numpy(ds.features).float().to(device)
        tgt_feats    = all_feats[idx]                         # (1,D)
        neigh_feats  = all_feats[neigh]                       # (1,K,D)
        item_hist    = tgt_feats.unsqueeze(1)                 # (1,1,D)
        opinions     = torch.tensor([[ [r] ]], device=device) # (1,1,1)

        with torch.no_grad():
            logit = model(item_hist, opinions, tgt_feats, neigh_feats)
            prob  = torch.sigmoid(logit).item()

        y_true.append(r)
        y_prob.append(prob)

    # Converte em numpy
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # Métricas
    auc   = roc_auc_score(y_true, y_prob)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    mean_like    = y_prob[y_true == 1].mean() if (y_true==1).any() else float('nan')
    mean_dislike = y_prob[y_true == 0].mean() if (y_true==0).any() else float('nan')

    print("=== Avaliação do GraphRec ===")
    print(f"AUC      : {auc:.3f}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"Média prob (likes)   : {mean_like:.3f}")
    print(f"Média prob (dislikes): {mean_dislike:.3f}")

if __name__ == '__main__':
    main()

# src/scripts/evaluate_graphrec.py

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

# Ajusta sys.path para importar src.*
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.datasets import GraphRecDataset
from src.models.graphrec import GraphRec

def main():
    # Paths
    FEATS = ROOT / "data" / "processed" / "combined_features.npy"
    LOGS_TEST  = ROOT / "data" / "logs" / "interactions_test.csv" # Apontando para o teste
    CKPT  = ROOT / "data" / "models" / "graphrec.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    # Carrega dataset de TESTE e modelo
    ds_test = GraphRecDataset(FEATS, LOGS_TEST, k_social=10)
    test_loader = DataLoader(ds_test, batch_size=32, shuffle=False)

    if not CKPT.exists():
        print(f"[ERRO] Checkpoint não encontrado em {CKPT}. Execute o treino primeiro.")
        return
    
    if len(ds_test) == 0:
        print(f"[ERRO] Dataset de teste ({LOGS_TEST}) está vazio. Verifique o arquivo.")
        return

    model = GraphRec(d_item=ds_test.features.shape[1]).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    # Containers
    y_true_list = []
    y_prob_list = []

    print(f"[INFO] Iniciando avaliação com {len(ds_test)} amostras de teste...")
    with torch.no_grad():
        for sample in ds_test: # Iterar sobre o dataset diretamente para simplicidade, batches também funcionam
            idx = sample["item_idx"].unsqueeze(0).to(device)
            r_true = sample["rating"].item() # Rótulo real, apenas para comparação final
            neigh = sample["neigh_idxs"].unsqueeze(0).to(device)

            # Monta tensores de features
            all_feats_tensor = torch.from_numpy(ds_test.features).float().to(device)
            target_feats = all_feats_tensor[idx]
            neigh_feats  = all_feats_tensor[neigh]

            # Histórico de 1 interação para o item alvo
            item_hist = target_feats.unsqueeze(1)

            # --- CORREÇÃO DE DATA LEAKAGE ---
            # Durante a avaliação, não podemos passar a opinião real (r_true) para o modelo.
            # Usamos um valor neutro (ex: 0.5) como placeholder.
            # Isso força o modelo a prever sem conhecer a resposta.
            neutral_opinion = torch.tensor([[[0.5]]], device=device, dtype=torch.float)

            logit = model(item_hist, neutral_opinion, target_feats, neigh_feats)
            prob  = torch.sigmoid(logit).item()

            y_true_list.append(r_true)
            y_prob_list.append(prob)

    if not y_true_list:
        print("[ERRO] Nenhuma predição foi feita. Verifique o dataset de teste e o loop de avaliação.")
        return

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    # Métricas
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    mean_like_prob = y_prob[y_true == 1].mean() if (y_true == 1).any() else float('nan')
    mean_dislike_prob = y_prob[y_true == 0].mean() if (y_true == 0).any() else float('nan')

    print("\n=== Avaliação do GraphRec (Conjunto de Teste Corrigido) ===")
    print(f"Total de amostras de teste: {len(y_true)}")
    print(f"AUC      : {auc:.3f}")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"Média prob (likes)   : {mean_like_prob:.3f}")
    print(f"Média prob (dislikes): {mean_dislike_prob:.3f}")

if __name__ == '__main__':
    main()
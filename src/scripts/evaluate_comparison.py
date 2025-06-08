# src/scripts/evaluate_comparison.py

import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm

# --- CONFIGURAÇÃO E PATHS ---
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.datasets import GraphRecDataset
from src.models.graphrec import GraphRec

# Paths dos artefatos
FEATS_PATH = ROOT / "data/processed/combined_features.npy"
MODEL_CKPT_PATH = ROOT / "data/models/graphrec.pth"
TEST_LOGS_PATH = ROOT / "data/logs/interactions_test.csv"

# Paths das personas
PATH_PERSONA_CONSISTENT = ROOT / "data/processed/persona_consistent_history.json"
PATH_PERSONA_INCONSISTENT = ROOT / "data/processed/persona_inconsistent_history.json"

# --- FUNÇÃO PARA PREPARAR HISTÓRICO DA PERSONA ---
def prepare_persona_history(persona_path: Path, features: np.ndarray, device: torch.device):
    """Carrega o histórico de uma persona e o converte em tensores do PyTorch."""
    with open(persona_path, "r") as f:
        history_indices = json.load(f)
    
    # Pega as features dos itens do histórico
    history_features = torch.from_numpy(features[history_indices]).float().to(device)
    # Simula que todos os itens do histórico foram "gostados"
    history_opinions = torch.ones(len(history_indices), 1, device=device)
    
    # Adiciona a dimensão de batch (1, N_hist, D_item)
    item_hist_tensor = history_features.unsqueeze(0)
    opinions_tensor = history_opinions.unsqueeze(0)
    
    return item_hist_tensor, opinions_tensor


# --- FUNÇÃO DE AVALIAÇÃO SEM HISTÓRICO (BASELINE) ---
def evaluate_no_history(model, test_dataset, features_tensor, device):
    print("\n--- Iniciando Avaliação 1: Sem Histórico (Baseline) ---")
    y_true, y_prob = [], []
    
    for sample in tqdm(test_dataset, desc="Avaliando (Baseline)"):
        idx = sample["item_idx"].unsqueeze(0).to(device)
        r_true = sample["rating"].item()
        neigh = sample["neigh_idxs"].unsqueeze(0).to(device)
        
        target_feats = features_tensor[idx]
        neigh_feats = features_tensor[neigh]
        
        # Para o UserModel, passamos o próprio item como um histórico de 1
        item_hist = target_feats.unsqueeze(1)
        # Usamos uma opinião neutra para não vazar a resposta
        neutral_opinion = torch.tensor([[[0.5]]], device=device, dtype=torch.float)
        
        with torch.no_grad():
            logit = model(item_hist, neutral_opinion, target_feats, neigh_feats)
            prob = torch.sigmoid(logit).item()
            
        y_true.append(r_true)
        y_prob.append(prob)
        
    return np.array(y_true), np.array(y_prob)


# --- FUNÇÃO DE AVALIAÇÃO COM PERSONAS ---
def evaluate_with_personas(model, test_dataset, features_tensor, personas, device):
    print("\n--- Iniciando Avaliação 2: Com Personas de Usuário ---")
    results = {name: {"y_true": [], "y_prob": []} for name in personas.keys()}
    
    for sample in tqdm(test_dataset, desc="Avaliando (Personas)"):
        idx = sample["item_idx"].unsqueeze(0).to(device)
        r_true = sample["rating"].item()
        neigh = sample["neigh_idxs"].unsqueeze(0).to(device)
        
        target_feats = features_tensor[idx]
        neigh_feats = features_tensor[neigh]
        
        for name, persona_data in personas.items():
            item_hist = persona_data["item_hist_tensor"]
            opinions = persona_data["opinions_tensor"]
            
            with torch.no_grad():
                logit = model(item_hist, opinions, target_feats, neigh_feats)
                prob = torch.sigmoid(logit).item()
            
            results[name]["y_true"].append(r_true)
            results[name]["y_prob"].append(prob)
    
    # Converte listas para numpy arrays
    for name in results:
        results[name]["y_true"] = np.array(results[name]["y_true"])
        results[name]["y_prob"] = np.array(results[name]["y_prob"])
        
    return results

# --- FUNÇÃO PARA IMPRIMIR MÉTRICAS ---
def print_metrics(title, y_true, y_prob):
    print(f"\n--- Resultados para: {title} ---")
    y_pred = (y_prob >= 0.5).astype(int)
    
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan')
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    mean_like_prob = y_prob[y_true == 1].mean() if (y_true == 1).any() else float('nan')
    mean_dislike_prob = y_prob[y_true == 0].mean() if (y_true == 0).any() else float('nan')
    
    print(f"  AUC      : {auc:.3f}")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  Média Prob (Likes)   : {mean_like_prob:.3f}")
    print(f"  Média Prob (Dislikes): {mean_dislike_prob:.3f}")


# --- FUNÇÃO PRINCIPAL ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")
    
    # Carrega o modelo treinado
    print("[INFO] Carregando modelo GraphRec...")
    features_np = np.load(FEATS_PATH)
    model = GraphRec(d_item=features_np.shape[1]).to(device)
    model.load_state_dict(torch.load(MODEL_CKPT_PATH, map_location=device))
    model.eval()
    print("  [OK] Modelo carregado.")
    
    # Prepara datasets e tensores
    features_tensor = torch.from_numpy(features_np).float().to(device)
    test_dataset = GraphRecDataset(FEATS_PATH, TEST_LOGS_PATH, k_social=10)
    
    # Carrega e prepara as personas
    print("[INFO] Carregando e preparando personas...")
    personas = {
        "Consistente": prepare_persona_history(PATH_PERSONA_CONSISTENT, features_np, device),
        "Inconsistente": prepare_persona_history(PATH_PERSONA_INCONSISTENT, features_np, device)
    }
    # Renomeia para clareza no loop
    personas = {name: {"item_hist_tensor": data[0], "opinions_tensor": data[1]} for name, data in personas.items()}
    print("  [OK] Personas prontas para avaliação.")

    # --- Executa e imprime avaliação baseline ---
    y_true_base, y_prob_base = evaluate_no_history(model, test_dataset, features_tensor, device)
    print_metrics("Baseline (Sem Histórico)", y_true_base, y_prob_base)
    
    # --- Executa e imprime avaliação com personas ---
    persona_results = evaluate_with_personas(model, test_dataset, features_tensor, personas, device)
    for name, results_data in persona_results.items():
        print_metrics(f"Persona: {name}", results_data["y_true"], results_data["y_prob"])
        
    print("\n[SUCESSO] Avaliação comparativa concluída.")

if __name__ == "__main__":
    main()
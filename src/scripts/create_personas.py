# src/scripts/create_personas.py

import json
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

print("[INFO] Iniciando a criação de personas de usuário...")

# --- CONFIGURAÇÃO ---
ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data/processed/combined_features.npy"
OUTPUT_DIR = ROOT / "data/processed"
HISTORY_SIZE = 20  # Número de itens no histórico de cada persona
ANCHOR_PROFILE_ID = 100 # Perfil usado como base para o usuário consistente

# --- Garante que o diretório de saída exista ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Carrega as features ---
print(f"[INFO] Carregando features de {FEATURES_PATH}...")
if not FEATURES_PATH.exists():
    print(f"[ERRO] Arquivo de features não encontrado. Execute o pipeline de processamento primeiro.")
    exit()
    
features = np.load(FEATURES_PATH)
n_profiles = features.shape[0]
print(f"  [OK] {n_profiles} perfis carregados.")


# --- 1. Persona Consistente ---
# Gosta de perfis muito similares a um perfil "âncora".
print(f"\n[INFO] Gerando Persona Consistente (baseado no perfil {ANCHOR_PROFILE_ID})...")
anchor_vector = features[ANCHOR_PROFILE_ID].reshape(1, -1)
similarities = cosine_similarity(anchor_vector, features).flatten()

# Pega os N mais similares, excluindo o próprio âncora
# np.argsort retorna índices do menor para o maior, então pegamos do final
consistent_indices = np.argsort(similarities)[-(HISTORY_SIZE + 1):-1]
# Inverte para ter o mais similar primeiro (opcional, mas lógico)
consistent_history = consistent_indices[::-1].tolist()

path_consistent = OUTPUT_DIR / "persona_consistent_history.json"
with open(path_consistent, "w") as f:
    json.dump(consistent_history, f)
print(f"  [OK] Histórico da persona consistente salvo em: {path_consistent}")


# --- 2. Persona Inconsistente ---
# Gosta de perfis completamente aleatórios, sem correlação.
print("\n[INFO] Gerando Persona Inconsistente...")
# Garante que não vamos selecionar o mesmo perfil duas vezes
inconsistent_history = np.random.choice(n_profiles, HISTORY_SIZE, replace=False).tolist()

path_inconsistent = OUTPUT_DIR / "persona_inconsistent_history.json"
with open(path_inconsistent, "w") as f:
    json.dump(inconsistent_history, f)
print(f"  [OK] Histórico da persona inconsistente salvo em: {path_inconsistent}")

print("\n[SUCESSO] Personas criadas com sucesso.")
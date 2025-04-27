# src/scripts/test_ranking.py
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

ORIGINAL = True  # ou False para a versão otimizada

# Carrega features
X = np.load(Path("data/processed/multimodal_features.npy"))
model = PreferenceModel(n_features=X.shape[1])

n_like = 1000
n_dis  = 1000
n_total = min(len(X), n_like + n_dis)

if ORIGINAL:
    print("▶ Versão ORIGINAL: update() + fit() a cada exemplo")
    for i in tqdm(range(n_like), desc="Likes", unit="ex"):
        model.update(X[i], like=True)
    for i in tqdm(range(n_like, n_total), desc="Dislikes", unit="ex"):
        model.update(X[i], like=False)
else:
    print("▶ Versão OTIMIZADA: acumula histórico e fit() único")
    for i in tqdm(range(n_like), desc="Likes", unit="ex"):
        model.X_hist.append(X[i].reshape(1, -1)); model.y_hist.append(1)
    for i in tqdm(range(n_like, n_total), desc="Dislikes", unit="ex"):
        model.X_hist.append(X[i].reshape(1, -1)); model.y_hist.append(0)
    print("▶ Treinando modelo uma única vez…")
    model._fit()

print("▶ Executando ranking…")
candidates = X[n_total:]
idx_top, probs_top = rank_candidates(model, candidates, top_k=5, explore_frac=0.2)

print("\nTop-5 índices (rel. aos candidatos):", idx_top)
print("Probabilidades                     :", probs_top.round(3))

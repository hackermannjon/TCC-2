# src/scripts/test_ranking.py
import numpy as np
from pathlib import Path
from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

X = np.load(Path("data/processed/multimodal_features.npy"))
model = PreferenceModel(n_features=X.shape[1])

# treino simplificado (mesmo de antes)
for i in range(100):
    model.update(X[i], like=True)
for i in range(100, 200):
    model.update(X[i], like=False)

# candidatos = todos fora do treino
candidates = X[200:]
idx_top, probs_top = rank_candidates(model, candidates, top_k=5, explore_frac=0.2)
print("Índices (200 +):", idx_top)
print("Probabilidades :", probs_top.round(3))

# src/recommender/core.py
"""
Funções de ranking e seleção de perfis.
"""
import numpy as np
from typing import Tuple
from src.models.preference_model import PreferenceModel


def rank_candidates(
    model: PreferenceModel,
    candidate_features: np.ndarray,
    top_k: int = 10,
    explore_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna:
      • idx_top   – índices (relativos a candidate_features) dos top_k perfis
      • probs_top – probabilidades correspondentes

    explore_frac = 0 → ranking puro
    explore_frac = 0.1 → embaralha 10 % dos top_k para explorar
    """
    # 1) Probabilidades de like para todos os candidatos
    probs = model.predict(candidate_features)

    # 2) Ordena do maior para o menor
    sorted_idx = np.argsort(-probs)

    # 3) Seleciona top_k
    idx_top = sorted_idx[:top_k]

    # 4) Exploração: embaralha fração do top_k
    if explore_frac > 0:
        n_explore = max(1, int(top_k * explore_frac))
        explore_part = np.random.choice(idx_top, n_explore, replace=False)
        np.random.shuffle(explore_part)
        # substitui as posições escolhidas pelo bloco embaralhado
        idx_top[:n_explore] = explore_part

    return idx_top, probs[idx_top]

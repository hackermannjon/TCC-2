# src/recommender/core.py
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
    Ordena perfis pelo score de like, com exploração real.
    - probs: probabilidades de like para cada candidato
    - sorted_idx: índices de candidatos ordenados do maior pro menor score
    - top_k: número de perfis a retornar
    - explore_frac: fração de top_k a substituir por candidatos logo abaixo (sem duplicar)
    """
    probs = model.predict(candidate_features)
    sorted_idx = np.argsort(-probs)
    
    # take the top_k
    top_idx = list(sorted_idx[:top_k])

    # compute how many to explore
    n_explore = int(top_k * explore_frac)
    if n_explore > 0:
        # candidates for exploration: slots immediately abaixo do top_k
        pool = list(sorted_idx[top_k : top_k + n_explore])
        # pick n_explore from this pool (or fewer, se pool pequeno)
        explore_picks = np.random.choice(pool, min(n_explore, len(pool)), replace=False)
        # randomly choose positions in top_idx to replace
        replace_positions = np.random.choice(range(top_k), len(explore_picks), replace=False)
        for pos, new_cand in zip(replace_positions, explore_picks):
            top_idx[pos] = new_cand

    # convert to np.array and get their probs
    top_idx = np.array(top_idx)
    return top_idx, probs[top_idx]

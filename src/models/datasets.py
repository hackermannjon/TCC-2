from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import torch
from torch.utils.data import Dataset

class GraphRecDataset(Dataset):
    """
    Dataset para GraphRec:
    - Carrega características combinadas (imagem, texto, demografia, personalidade, social)
    - Lê interações do usuário (likes/dislikes)
    - Constrói vizinhança social via KNN para cada item

    Cada amostra retorna:
        item_idx: índice do item (profile_id)
        rating:   rótulo (1.0 para like, 0.0 para dislike)
        neigh_idxs: tensor long com até k vizinhos sociais
    """
    def __init__(self,
                 features_path: Path,
                 interactions_path: Path,
                 k_social: int = 10):
        # Carrega features combinadas (N_items, D)
        self.features = np.load(features_path)
        self.n_items = self.features.shape[0]

        # Lê log de interações
        logs = pd.read_csv(interactions_path)
        # Garante que exista pelo menos duas classes
        logs = logs[logs['profile_id'].between(0, self.n_items - 1)]
        self.interactions = list(zip(
            logs['profile_id'].astype(int).tolist(),
            logs['like'].astype(int).tolist()
        ))

        # Constroi grafo social sintético via KNN em features
        knn = NearestNeighbors(n_neighbors=k_social + 1, metric='cosine')
        knn.fit(self.features)
        neigh = knn.kneighbors(self.features, return_distance=False)
        # Remove o próprio nó e mantém apenas k vizinhos
        self.social_neighbors = [
            [j for j in neigh[i] if j != i][:k_social]
            for i in range(self.n_items)
        ]

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        item_idx, rating = self.interactions[idx]
        # converte para tensores
        item_idx_t = torch.tensor(item_idx, dtype=torch.long)
        rating_t   = torch.tensor(rating, dtype=torch.float)
        neigh_idxs = torch.tensor(self.social_neighbors[item_idx], dtype=torch.long)
        return {
            'item_idx': item_idx_t,
            'rating': rating_t,
            'neigh_idxs': neigh_idxs
        }

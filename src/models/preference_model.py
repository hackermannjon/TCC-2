# src/models/preference_model.py

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.neighbors import NearestNeighbors

from src.models.graphrec import GraphRec
from src.models.datasets import GraphRecDataset

class PreferenceModel:
    """
    Integra o GraphRec pré-treinado para preferência online.

    Métodos:
      - update(item_idx, like): registra feedback (CSV + memória)
      - train(epochs, batch_size, lr): treina GraphRec sobre histórico
      - predict(X): retorna probabilidades para vetores em X (batch)
    """
    def __init__(
        self,
        feats_path: Path = Path("data/processed/combined_features.npy"),
        logs_path : Path = Path("data/logs/interactions.csv"),
        ckpt_path : Path = Path("data/models/graphrec.pth"),
        device    : str  | None = None,
    ):
        # dispositivo (GPU se disponível)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"▶ PreferenceModel em: {self.device}")

        # carrega dataset para features e grafo social
        ds = GraphRecDataset(feats_path, logs_path, k_social=10)
        self.features         = ds.features           # np.ndarray (N_items, D_item)
        self.social_neighbors = ds.social_neighbors   # list[list[int]]
        self.logs_path        = logs_path

        # carrega GraphRec pré-treinado
        D_ITEM = self.features.shape[1]
        self.model = GraphRec(d_item=D_ITEM).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # pré-treina um NN para achar idx mais rápido
        self._nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self._nn.fit(self.features)

        # histórico de interações
        self.h_feats: list[torch.Tensor] = []
        self.h_ops  : list[float]        = []

    def update(self, item_idx: int, like: bool) -> None:
        """Registra feedback no CSV e adiciona ao histórico."""
        pd.DataFrame([{
            "profile_id": item_idx,
            "like": int(like),
            "timestamp": pd.Timestamp.utcnow()
        }]).to_csv(
            self.logs_path,
            mode="a",
            header=not self.logs_path.exists(),
            index=False
        )
        # armazena o vetor de features e o label
        self.h_feats.append(torch.from_numpy(self.features[item_idx]).float())
        self.h_ops.append(float(like))

    def _prep_static_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Empilha todo o histórico em tensores para train/predict."""
        feats = torch.stack(self.h_feats).to(self.device)            # (N_hist, D_item)
        ops   = torch.tensor(self.h_ops, device=self.device)         # (N_hist,)
        item_hist = feats.unsqueeze(0)                               # (1, N_hist, D_item)
        opinions  = ops.unsqueeze(0).unsqueeze(-1)                   # (1, N_hist, 1)
        return item_hist, opinions

    def train(self, epochs: int = 5, batch_size: int = 32, lr: float = 3e-4) -> None:
        """
        Treina o GraphRec em minibatches sobre todo o histórico de interações.
        """
        if len(set(self.h_ops)) < 2:
            print("⇢ Precisa de pelo menos 1 like e 1 dislike para treinar.")
            return

        item_hist, opinions = self._prep_static_tensors()            # (1,N,D), (1,N,1)
        feats_all = torch.stack(self.h_feats).to(self.device)         # (N, D_item)
        labels    = torch.tensor(self.h_ops, device=self.device)     # (N,)
        idxs      = torch.arange(len(self.h_feats), dtype=torch.long)  # (N,)

        dataset = torch.utils.data.TensorDataset(feats_all, labels, idxs)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        optim   = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(1, epochs+1):
            total_loss = 0.0
            for batch_feats, batch_lbls, batch_idxs in loader:
                B = batch_feats.size(0)
                # expande histórico p/ cada item do batch
                hist_b = item_hist.repeat(B, 1, 1)   # (B, N_hist, D_item)
                ops_b  = opinions.repeat(B, 1, 1)    # (B, N_hist, 1)

                # monta vizinhos sociais
                neigh_list = []
                for idx in batch_idxs.tolist():
                    neigh_ids = self.social_neighbors[idx]
                    neigh_list.append(torch.from_numpy(self.features[neigh_ids]).float())
                neigh_b = torch.stack(neigh_list).to(self.device)  # (B, K, D_item)

                optim.zero_grad()
                logits = self.model(hist_b, ops_b, batch_feats.to(self.device), neigh_b)  # (B,)
                loss   = loss_fn(logits, batch_lbls.to(self.device))
                loss.backward()
                optim.step()
                total_loss += loss.item() * B

            avg_loss = total_loss / len(dataset)
            print(f"[epoch {ep}/{epochs}] loss={avg_loss:.4f}")

        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Recebe X de forma (M, D_item) e retorna um array de probabilidades (M,).
        Todo o batch é inferido de uma vez (sem loops gigantes de memória).
        """
        if not self.h_feats:
            raise RuntimeError("Use update() + train() antes de predict().")

        # encontra, para cada vetor em X, o idx mais próximo
        idxs = self._nn.kneighbors(X, return_distance=False).reshape(-1)  # (M,)
        M = len(idxs)

        # cria tensores de target e vizinhos
        tgt = torch.from_numpy(self.features[idxs]).float().to(self.device)  # (M, D_item)
        neigh_list = [torch.from_numpy(self.features[self.social_neighbors[i]]).float()
                      for i in idxs]
        neigh = torch.stack(neigh_list).to(self.device)                     # (M, K, D_item)

        # empilha histórico para todos de uma vez
        item_hist, opinions = self._prep_static_tensors()                   # (1,N, D), (1,N,1)
        hist_b = item_hist.repeat(M, 1, 1)                                  # (M, N, D_item)
        ops_b  = opinions.repeat(M, 1, 1)                                   # (M, N, 1)

        with torch.no_grad():
            logits = self.model(hist_b, ops_b, tgt, neigh)  # (M,)
            probs  = torch.sigmoid(logits).cpu().numpy()

        return probs

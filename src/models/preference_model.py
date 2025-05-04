from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.graphrec import GraphRec
from src.models.datasets import GraphRecDataset


class PreferenceModel:
    """
    • update(...)   – armazena cliques em memória e CSV (instantâneo)
    • train(...)    – treina o GraphRec em lote sobre todo o histórico
    • predict(X)    – gera probabilidades de like para vetores em X
    """
    def __init__(
        self,
        feats_path: Path = Path("data/processed/combined_features.npy"),
        logs_path : Path = Path("data/logs/interactions.csv"),
        ckpt_path : Path = Path("data/models/graphrec.pth"),
        device    : str  | None = None,
    ):
        # dispositivo
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"▶ PreferenceModel em: {self.device}")

        # carrega grafo e features
        ds = GraphRecDataset(feats_path, logs_path, k_social=10)
        self.features         = ds.features
        self.social_neighbors = ds.social_neighbors
        self.logs_path        = logs_path

        # carrega modelo pré-treinado
        d_item = self.features.shape[1]
        self.model = GraphRec(d_item=d_item).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # histórico de interações
        self.h_feats: list[torch.Tensor] = []  # (D,) cada
        self.h_ops  : list[float]        = []  # 1.0 ou 0.0

    def update(self, item_idx: int, like: bool) -> None:
        """
        Armazena um clique (like/dislike) em memória e no CSV.
        """
        df = pd.DataFrame([{"profile_id": item_idx,
                             "like": int(like),
                             "timestamp": pd.Timestamp.utcnow()}])
        df.to_csv(
            self.logs_path,
            mode="a", header=not self.logs_path.exists(), index=False
        )
        feat = torch.from_numpy(self.features[item_idx]).float()
        self.h_feats.append(feat)
        self.h_ops.append(float(like))

    def _prep_static_tensors(self):
        """
        Empilha todo o histórico em tensores fixos para uso em train e predict.
        Retorna: item_hist (1,N,D), opinions (1,N,1), neigh_tensor (N,K,D)
        """
        feats = torch.stack(self.h_feats).to(self.device)          # (N, D)
        ops   = torch.tensor(self.h_ops, device=self.device).unsqueeze(1)  # (N,1)
        item_hist = feats.unsqueeze(0)                             # (1, N, D)
        opinions  = ops.unsqueeze(0)                               # (1, N, 1)

        # vizinhos de cada histórico
        N, D = feats.shape
        K = len(self.social_neighbors[0])
        neigh_tensor = torch.empty((N, K, D), device=self.device)
        for i, f in enumerate(self.h_feats):
            idx = int(np.argmin(np.linalg.norm(self.features - f.cpu().numpy(), axis=1)))
            neigh_tensor[i] = torch.from_numpy(
                self.features[self.social_neighbors[idx]]
            ).float().to(self.device)

        return item_hist, opinions, neigh_tensor

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 3e-4
    ) -> None:
        """
        Treina o GraphRec usando TODO o histórico de interações coletado.
        Exige pelo menos 1 like e 1 dislike.
        """
        if len(set(self.h_ops)) < 2:
            print("⇢ Necessário ao menos um like e um dislike para treinar.")
            return

        item_hist, opinions, _ = self._prep_static_tensors()
        feats_all = torch.stack(self.h_feats).to(self.device)        # (N, D)
        labels_all = torch.tensor(self.h_ops, device=self.device)    # (N,)

        ds = TensorDataset(feats_all, labels_all, torch.arange(len(self.h_feats)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        self.model.train()
        optim   = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for ep in range(1, epochs+1):
            total_loss = 0.0
            for batch_feats, batch_lbls, batch_idxs in dl:
                B = batch_feats.size(0)
                item_hist_b = item_hist.repeat(B, 1, 1)    # (B, N, D)
                opinions_b  = opinions.repeat(B, 1, 1)     # (B, N, 1)

                # vizinhos por amostra
                neigh_list = []
                for idx in batch_idxs.tolist():
                    neigh_ids = self.social_neighbors[idx]
                    neigh_list.append(
                        torch.from_numpy(self.features[neigh_ids]).float()
                    )
                neigh_b = torch.stack(neigh_list).to(self.device)  # (B, K, D)

                optim.zero_grad()
                logits = self.model(
                    item_hist_b, opinions_b,
                    batch_feats.to(self.device), neigh_b
                )  # (B,)
                loss = loss_fn(logits, batch_lbls.to(self.device))
                loss.backward()
                optim.step()
                total_loss += loss.item() * B

            avg = total_loss / len(ds)
            print(f"[epoch {ep}/{epochs}] loss={avg:.4f}")

        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna array de probabilidades de like para cada vetor em X (M, D).
        Requer update() + train() prévios.
        """
        if not self.h_feats:
            raise RuntimeError("Use update() e train() antes de predict().")

        item_hist, opinions, _ = self._prep_static_tensors()
        probs = []
        for vec in X:
            idx = int(np.argmin(np.linalg.norm(self.features - vec, axis=1)))
            tgt = torch.from_numpy(self.features[idx]).float().to(self.device).unsqueeze(0)   # (1, D)
            neigh = torch.from_numpy(
                self.features[self.social_neighbors[idx]]
            ).float().to(self.device).unsqueeze(0)  # (1, K, D)
            with torch.no_grad():
                p = torch.sigmoid(self.model(item_hist, opinions, tgt, neigh)).item()
            probs.append(p)
        return np.array(probs)

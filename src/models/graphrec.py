"""
GraphRec adaptado para o seu TCC
--------------------------------
• UserModel: atenção sobre itens interagidos (like/dislike)
• ItemModel: atenção sobre vizinhos sociais (KNN)
• Predictor : MLP sobre concat([z_u ; z_j])  → probabilidade de like
"""
from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserModel(nn.Module):
    def __init__(self, d_item: int, d_user: int):
        super().__init__()
        self.opinion_fc = nn.Linear(d_item + 1, d_user)
        self.att_fc      = nn.Linear(d_user, 1)

    def forward(self, item_feats: torch.Tensor,
                opinions: torch.Tensor) -> torch.Tensor:
        """
        item_feats : (B, N_int, D_item)
        opinions   : (B, N_int, 1)  -- 1.0 like / 0.0 dislike
        Return     : (B, D_user)
        """
        x = torch.cat([item_feats, opinions], dim=-1)          # opinion‑aware
        h = torch.tanh(self.opinion_fc(x))                     # (B, N_int, d_user)
        α = torch.softmax(self.att_fc(h), dim=1)               # (B, N_int, 1)
        z_u = (α * h).sum(dim=1)                               # (B, d_user)
        return z_u


class ItemModel(nn.Module):
    def __init__(self, d_item: int):
        super().__init__()
        self.self_fc   = nn.Linear(d_item, d_item)
        self.neigh_fc  = nn.Linear(d_item, d_item)
        self.att_fc    = nn.Linear(d_item, 1)

    def forward(self, self_feat: torch.Tensor,
                neigh_feats: torch.Tensor) -> torch.Tensor:
        """
        self_feat   : (B, D_item)
        neigh_feats : (B, K, D_item)
        Return      : (B, D_item)
        """
        h_self  = torch.tanh(self.self_fc(self_feat))          # (B, D)
        h_neigh = torch.tanh(self.neigh_fc(neigh_feats))       # (B, K, D)
        β = torch.softmax(self.att_fc(h_neigh), dim=1)         # (B, K, 1)
        agg = (β * h_neigh).sum(dim=1)                         # (B, D)
        z_j = h_self + agg
        return z_j


class GraphRec(nn.Module):
    def __init__(self, d_item: int, d_user: int = 128):
        super().__init__()
        self.user_model = UserModel(d_item, d_user)
        self.item_model = ItemModel(d_item)
        self.predictor  = nn.Sequential(
            nn.Linear(d_user + d_item, 256),
            nn.ReLU(),
            nn.Linear(256, 1)          # logit
        )

    def forward(self,
                item_feats: torch.Tensor,       # (B, N_int, D_item)
                opinions: torch.Tensor,         # (B, N_int, 1)
                target_feats: torch.Tensor,     # (B, D_item)
                neigh_feats: torch.Tensor       # (B, K, D_item)
               ) -> torch.Tensor:
        z_u = self.user_model(item_feats, opinions)            # (B, d_user)
        z_j = self.item_model(target_feats, neigh_feats)       # (B, D_item)
        out = self.predictor(torch.cat([z_u, z_j], dim=-1))    # (B, 1)
        return out.squeeze(-1)                                 # (B,)

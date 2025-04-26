# src/models/preference_model.py
"""
PreferenceModel — Logistic Regression com histórico completo
• Armazena todo feedback (likes/dislikes)
• Escala features via StandardScaler
• Retreina LogisticRegression liblinear sempre que existem
  pelo menos 1 exemplo de cada classe
"""

from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class PreferenceModel:
    def __init__(self, n_features: int, random_state: int = 42):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(
            solver="liblinear",
            random_state=random_state,
            max_iter=1000,
        )
        self.X_hist: list[np.ndarray] = []
        self.y_hist: list[int] = []
        self._trained = False

    # ---------- treinamento interno ----------
    def _fit(self) -> None:
        """Treina/re-treina se há pelo menos 1 exemplo de cada classe."""
        if len(set(self.y_hist)) < 2:
            # ainda não temos as duas classes → adia o ajuste
            return
        X = np.vstack(self.X_hist)
        y = np.array(self.y_hist)
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self._trained = True

    # ---------- API pública ----------
    def update(self, x: np.ndarray, like: bool) -> None:
        """Armazena o feedback e re-treina quando possível."""
        self.X_hist.append(x.reshape(1, -1))
        self.y_hist.append(1 if like else 0)
        self._fit()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probabilidade de like para cada vetor em X."""
        if not self._trained:
            raise RuntimeError("Modelo ainda não possui exemplos das duas classes.")
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

    # ---------- persistência ----------
    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceModel":
        return joblib.load(path)

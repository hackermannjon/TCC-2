# src/models/preference_model.py
"""
PreferenceModel — Logistic Regression com retreinamento completo

• Armazena histórico completo de interações (X, y)
• Aplica StandardScaler
• Usa LogisticRegression regularizado (C=0.01)
• Retreina o modelo a cada novo feedback
"""
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
            C=0.01,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        )
        self.X_hist: list[np.ndarray] = []
        self.y_hist: list[int] = []
        self._trained = False

    def _fit(self):
        # só treina quando há exemplos de ambas classes
        if len(set(self.y_hist)) < 2:
            return
        X = np.vstack(self.X_hist)
        y = np.array(self.y_hist)
        Xs = self.scaler.fit_transform(X)
        self.clf.fit(Xs, y)
        self._trained = True

    def update(self, x: np.ndarray, like: bool) -> None:
        """Adiciona novo feedback e retreina o modelo."""
        self.X_hist.append(x.reshape(1, -1))
        self.y_hist.append(1 if like else 0)
        self._fit()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade de like para cada vetor em X."""
        if not self._trained:
            raise RuntimeError("Modelo não treinado com duas classes.")
        Xs = self.scaler.transform(X)
        return self.clf.predict_proba(Xs)[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PreferenceModel":
        return joblib.load(path)

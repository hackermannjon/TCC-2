# src/scripts/evaluate_model.py

import numpy as np
import pandas as pd
from pathlib import Path
from src.models.preference_model import PreferenceModel
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score

# --- Paths ---
FEAT_PATH = Path("data/processed/multimodal_features.npy")
LOG_PATH  = Path("data/logs/interactions.csv")

# --- Carrega dados ---
X = np.load(FEAT_PATH)
logs = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
logs = logs.sort_values("timestamp").reset_index(drop=True)

# --- Split cronológico 80/20 ---
n = len(logs)
n_train = int(n * 0.8)
train_logs = logs.iloc[:n_train]
test_logs  = logs.iloc[n_train:]

# --- Treina modelo ---
model = PreferenceModel(n_features=X.shape[1])
for _, row in train_logs.iterrows():
    idx = int(row["profile_id"])
    like = bool(row["like"])
    model.update(X[idx], like=like)

# --- Previsões no teste ---
test_idxs = test_logs["profile_id"].astype(int).to_numpy()
y_true    = test_logs["like"].astype(int).to_numpy()
y_probs   = model.predict(X[test_idxs])

# --- Métricas ---
auc   = roc_auc_score(y_true, y_probs)
acc   = accuracy_score(y_true, y_probs >= 0.5)
prec  = precision_score(y_true, y_probs >= 0.5)

print(f"Test set size: {len(test_logs)}")
print(f"AUC       : {auc:.3f}")
print(f"Accuracy  : {acc:.3f}")
print(f"Precision : {prec:.3f}")

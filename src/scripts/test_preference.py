# src/scripts/test_preference.py
import numpy as np
from pathlib import Path
from src.models.preference_model import PreferenceModel

X = np.load(Path("data/processed/multimodal_features.npy"))
model = PreferenceModel(n_features=X.shape[1])

# 100 likes + 100 dislikes
for i in range(100):
    model.update(X[i], like=True)
for i in range(100, 200):
    model.update(X[i], like=False)

print("Likes simulados   :", model.predict(X[:5]).round(3))
print("Dislikes simulados:", model.predict(X[100:105]).round(3))

# src/scripts/test_ranking.py

from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.models.preference_model import PreferenceModel

# carrega todas as features
X = np.load(Path("data/processed/combined_features.npy"))

model = PreferenceModel(
    feats_path=Path("data/processed/combined_features.npy"),
    logs_path=Path("data/logs/interactions.csv"),
    ckpt_path=Path("data/models/graphrec.pth")
)

# simula 100 likes e 100 dislikes
for i in tqdm(range(100), desc="likes", unit="it"):
    model.update(i, like=True)
for i in tqdm(range(100, 200), desc="dislikes", unit="it"):
    model.update(i, like=False)

print("▶ treino offline …")
model.train(epochs=5, batch_size=32, lr=3e-4)

# agora ranking em batch único
candidates = X[200:]
probs = model.predict(candidates)
top5 = np.argsort(-probs)[:5]

print("\nTop-5 idx (candidates):", top5)
print("Probabilidades        :", np.round(probs[top5], 3))

from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.models.preference_model import PreferenceModel

X = np.load(Path("data/processed/combined_features.npy"))

model = PreferenceModel(
    feats_path=Path("data/processed/combined_features.npy"),
    logs_path=Path("data/logs/interactions.csv"),
    ckpt_path=Path("data/models/graphrec.pth")
)

for i in tqdm(range(100), desc="likes"):
    model.update(i, like=True)
for i in tqdm(range(100, 200), desc="dislikes"):
    model.update(i, like=False)

print("▶ treino offline …")
model.train(epochs=5, batch_size=32, lr=3e-4)

candidates = X[200:]
probs = [model.predict(np.array([v]))[0] for v in tqdm(candidates, desc="ranking")]
probs = np.array(probs)

top5 = np.argsort(-probs)[:5]
print("\nTop‑5 idx (candidates) :", top5)
print("Probabilidades         :", np.round(probs[top5], 3))

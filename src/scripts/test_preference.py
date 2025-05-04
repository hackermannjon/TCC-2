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

likes_probs    = model.predict(X[:5])
dislikes_probs = model.predict(X[100:105])

print("\nProb. primeiros 5 (likes)   :", np.round(likes_probs, 3))
print("Prob. next 5 (dislikes)     :", np.round(dislikes_probs, 3))

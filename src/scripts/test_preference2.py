# src/scripts/test_preference_epochs.py

import numpy as np
from pathlib import Path
from src.models.preference_model import PreferenceModel

# Carrega features multimodais
X = np.load(Path("data/processed/multimodal_features.npy"))

# Instancia o modelo original (sem buffer)
model = PreferenceModel(n_features=X.shape[1])

# Treina 5 epochs sobre o mesmo conjunto de 200 exemplos
for epoch in range(5):
    # prints opcionais para ver o progresso:
    print(f"[Epoch {epoch+1}/5] treinando likes e dislikes…")
    for i in range(100):
        model.update(X[i], like=True)
    for i in range(100, 200):
        model.update(X[i], like=False)

# Após o treino múltiplo, faz previsões
likes_probs    = model.predict(X[:5]).round(3)
dislikes_probs = model.predict(X[100:105]).round(3)

print("Probabilidades sobre os 5 primeiros (esperado alto):", likes_probs)
print("Probabilidades sobre 100–104 (esperado baixo):", dislikes_probs)

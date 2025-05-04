"""
Concatena o vetor multimodal (2439 dims) com o vetor social (64 dims),
gerando o vetor final de 2503 dimensões.

Salva em: data/processed/combined_features.npy
"""
from pathlib import Path
import numpy as np

ROOT   = Path(__file__).resolve().parents[2]
DATA_P = ROOT / "data" / "processed"

MULTI = DATA_P / "multimodal_features.npy"   # 2439 dims
SOC   = DATA_P / "social_embeddings.npy"     # 64 dims
OUT   = DATA_P / "combined_features.npy"     # 2503 dims

print("▶ carregando multimodal…", MULTI)
X_multi = np.load(MULTI)
print("   shape", X_multi.shape)

print("▶ carregando social…", SOC)
X_soc = np.load(SOC)
print("   shape", X_soc.shape)

assert X_multi.shape[0] == X_soc.shape[0], "Mismatch N linhas"

print("▶ concatenando…")
X_combined = np.hstack([X_multi, X_soc]).astype("float32")
print("   new shape", X_combined.shape)

OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT, X_combined)
print("✅ combined_features salvo:", OUT)

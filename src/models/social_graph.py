"""
Gera um grafo sintético de “conhecidos em comum”.

• Usa K-Nearest Neighbors (k = 10) sobre os embeddings multimodais
• Extrai vetores sociais (64 dims) com Node2Vec
• Salva em: data/processed/social_embeddings.npy   (N, 64)
"""
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from node2vec import Node2Vec           # pip install node2vec

ROOT      = Path(__file__).resolve().parents[2]
DATA_PROC = ROOT / "data" / "processed"
MULTI     = DATA_PROC / "multimodal_features.npy"
SOC_OUT   = DATA_PROC / "social_embeddings.npy"

print("▶ carregando multimodal…")
X = np.load(MULTI)          # (N, 2439)
N = X.shape[0]

print("▶ construindo grafo KNN (k=10)…")
knn = NearestNeighbors(n_neighbors=10, metric="cosine").fit(X)
neighbors = knn.kneighbors(return_distance=False)

G = nx.Graph()
G.add_nodes_from(range(N))
for i, neigh_list in enumerate(neighbors):
    for j in neigh_list:
        if i != j:
            G.add_edge(i, j)
print(f"   nós: {G.number_of_nodes()}   arestas: {G.number_of_edges()}")

print("▶ rodando Node2Vec (dim=64)…")
node2vec = Node2Vec(G, dimensions=64, walk_length=20,
                    num_walks=10, workers=4, quiet=True)
model = node2vec.fit(window=10, min_count=1, batch_words=4)

social_vecs = np.zeros((N, 64), dtype="float32")
for n in G.nodes():
    social_vecs[n] = model.wv[str(n)]

SOC_OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(SOC_OUT, social_vecs)
print("[OK] social_embeddings salvo:", SOC_OUT, "shape", social_vecs.shape)

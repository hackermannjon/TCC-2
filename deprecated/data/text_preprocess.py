"""
Pré-processamento textual para MatchPredict-AI.

Carrega o CSV de perfis, aplica limpeza básica e gera embeddings
BERT/Sentence-BERT para cada `self_summary` (ou campo escolhido).
Salva um `.npy` com shape (N_perfis, dim_embedding).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ---------- #
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_FILE  = DATA_DIR / "raw" / "okcupid_profiles.csv"
EMB_FILE  = DATA_DIR / "processed" / "text_embeddings.npy"
TEXT_COL  = "essay0"          # coluna com a bio principal
MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim SBERT, leve e rápido
BATCH_SIZE = 64
# ---------------------------- #

def clean_text(text: str) -> str:
    """Remoções simples – links, múltiplos espaços etc."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = text.replace("<br />", " ").replace("\n", " ")
    text = " ".join(text.split())  # colapsa espaços
    return text

def load_profiles(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].apply(clean_text)
    return df[TEXT_COL]

def make_embeddings(texts, model_name=MODEL_NAME, batch=BATCH_SIZE):
    model = SentenceTransformer(model_name)
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch), desc="Embeddings"):
        chunk = texts[i : i + batch].tolist()
        vecs = model.encode(chunk, show_progress_bar=False, batch_size=batch, normalize_embeddings=True)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

def main():
    print("▶ carregando perfis…")
    texts = load_profiles(RAW_FILE)
    print(f"▶ {len(texts)} bios lidas.")
    print("▶ gerando embeddings SBERT…")
    emb = make_embeddings(texts)
    EMB_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, emb)
    print(f"[OK] embeddings salvos em {EMB_FILE} – shape={emb.shape}")

if __name__ == "__main__":
    main()

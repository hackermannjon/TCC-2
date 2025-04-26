# src/models/features.py
"""
Concatena embeddings de texto e imagem para criar features multimodais.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Caminhos
BASE_DIR          = Path(__file__).resolve().parents[2]
TEXT_EMB_PATH     = BASE_DIR / "data" / "processed" / "text_embeddings.npy"
IMG_FEAT_PATH     = BASE_DIR / "data" / "processed" / "image_features.npy"
OUTPUT_FEAT_PATH  = BASE_DIR / "data" / "processed" / "multimodal_features.npy"
CSV_INPUT_PATH    = BASE_DIR / "data" / "raw" / "okcupid_profiles.csv"
CSV_OUTPUT_PATH   = BASE_DIR / "data" / "processed" / "selected_profiles.csv"

def main():
    # 1) Carrega embeddings
    text_emb = np.load(TEXT_EMB_PATH)
    img_feat = np.load(IMG_FEAT_PATH)

    # 2) Alinha número de instâncias
    n = min(text_emb.shape[0], img_feat.shape[0])
    text_emb = text_emb[:n]
    img_feat = img_feat[:n]

    # 3) Concatena
    multimodal = np.concatenate([text_emb, img_feat], axis=1)

    # 4) Salva features multimodais
    OUTPUT_FEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_FEAT_PATH, multimodal)
    print(f"✅ Multimodal features salvas em {OUTPUT_FEAT_PATH} com shape {multimodal.shape}")

    # 5) Exporta metadados correspondentes
    df = pd.read_csv(CSV_INPUT_PATH)
    df_selected = df.iloc[:n]
    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_selected.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"✅ Metadados dos primeiros {n} perfis salvos em {CSV_OUTPUT_PATH}")

if __name__ == "__main__":
    main()

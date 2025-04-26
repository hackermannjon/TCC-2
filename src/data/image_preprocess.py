"""
Pré-processamento de imagens para MatchPredict-AI.

Carrega todas as imagens de data/raw/images/ProfilesDataSet/,
redimensiona para 224×224, extrai features da penúltima camada
da ResNet50 pré-treinada e salva em data/processed/image_features.npy.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torchvision import models

# ---------- CONFIG ---------- #
DATA_DIR    = Path(__file__).resolve().parents[2] / "data"
IMG_DIR     = DATA_DIR / "raw" / "images" / "ProfilesDataSet"
OUT_FILE    = DATA_DIR / "processed" / "image_features.npy"
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------------- #

def load_image_paths(img_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]

def build_model():
    # Carrega ResNet50 sem a cabeça de classificação
    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]  # remove última camada (fc)
    model = torch.nn.Sequential(*modules)
    model.eval().to(DEVICE)
    return model

def preprocess_image(img_path: Path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img)

def main():
    paths = load_image_paths(IMG_DIR)
    if not paths:
        print(f"❌ Não encontrou imagens em {IMG_DIR}")
        return

    # Transformações padrão da ResNet
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std= [0.229, 0.224, 0.225]),
    ])

    model = build_model()
    all_feats = []

    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Extraindo features"):
            batch_paths = paths[i : i + BATCH_SIZE]
            imgs = [preprocess_image(p, transform) for p in batch_paths]
            batch = torch.stack(imgs).to(DEVICE)      # (B,3,224,224)
            feats = model(batch)                     # (B,2048,1,1)
            feats = feats.view(feats.size(0), -1)    # (B,2048)
            all_feats.append(feats.cpu().numpy())

    feats_array = np.vstack(all_feats)  # (N,2048)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_FILE, feats_array)
    print(f"✅ Extraídas {feats_array.shape[0]} features e salvas em {OUT_FILE}")

if __name__ == "__main__":
    main()

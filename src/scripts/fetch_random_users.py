"""
fetch_random_users.py

Busca N perfis da Random Data API e salva:
 - data/raw/random_users.json   (metadados de todos os perfis)
 - data/raw/images/{uuid}.jpg   (avatares de cada perfil)
"""

import os
import json
import time
import requests
from pathlib import Path
from tqdm import tqdm

# ---------- CONFIGURAÇÃO ---------- #
API_URL        = "https://random-data-api.com/api/users/random_user"
TOTAL_PROFILES = 3000           # quantos perfis queremos
BATCH_SIZE     = 500            # quantos por requisição
DATA_DIR       = Path(__file__).resolve().parents[2] / "data"
RAW_DIR        = DATA_DIR / "raw"
IMG_DIR        = RAW_DIR / "images"
OUT_JSON       = RAW_DIR / "random_users.json"
MAX_RETRIES    = 3              # tentativas por lote
RETRY_DELAY    = 5              # segundos entre tentativas
# ----------------------------------- #

def fetch_batch(n):
    """Faz uma chamada à API para 'n' perfis."""
    params = {"size": n}
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()
    return resp.json()  # já é lista de dicts

def fetch_all_profiles(total, batch_size):
    """Agrega 'total' perfis em lotes de 'batch_size', com retry."""
    profiles = []
    batches = (total + batch_size - 1) // batch_size
    for i in range(batches):
        want = min(batch_size, total - len(profiles))
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                batch = fetch_batch(want)
                profiles.extend(batch)
                break
            except Exception as e:
                print(f"⚠️ Erro lote {i+1}/{batches}, tentativa {attempt}: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    raise
    return profiles

def save_metadata(profiles, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadados salvos em {path} (total: {len(profiles)} perfis)")

def download_images(profiles, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(profiles, desc="Baixando avatares"):
        uuid = p["id"]  # Random Data API fornece 'id' ou 'uid'; ajuste se necessário
        url  = p["avatar"]
        ext  = os.path.splitext(url)[1] or ".jpg"
        img_path = img_dir / f"{uuid}{ext}"
        if img_path.exists():
            continue
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(resp.content)
        else:
            print(f"⚠️ Falha ao baixar {url}")
    total = len(list(img_dir.iterdir()))
    print(f"✅ Avatares salvos em {img_dir} (total: {total})")

def main():
    print(f"▶ Iniciando fetch de {TOTAL_PROFILES} perfis em batches de {BATCH_SIZE}...")
    profiles = fetch_all_profiles(TOTAL_PROFILES, BATCH_SIZE)
    save_metadata(profiles, OUT_JSON)
    print("▶ Baixando avatares de perfil...")
    download_images(profiles, IMG_DIR)

if __name__ == "__main__":
    main()

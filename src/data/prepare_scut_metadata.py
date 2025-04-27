# src/data/prepare_scut_metadata.py

import pandas as pd
from pathlib import Path

def main():
    # local do labels.csv da SCUT-FBP5500 v2
    raw = Path("data/raw/scut_fbp5500")
    labels = raw / "labels.csv"       # este CSV vem no dataset
    images = raw / "images"           # dentro já estão 0.jpg … 5499.jpg

    # lê o labels.csv (cada linha: image_name,beauty_score)
    df = pd.read_csv(labels)
    # garante que o campo image_name termine em .jpg
    if not df["image_name"].str.endswith(".jpg").all():
        df["image_name"] = df["image_name"].astype(str) + ".jpg"

    # salva só os campos que precisamos
    out = Path("data/processed")
    out.mkdir(exist_ok=True)
    df.to_csv(out / "profiles_metadata.csv", index=False)
    print(f"✅ Metadados SCUT salvos em {out/'profiles_metadata.csv'}")

if __name__ == "__main__":
    main()

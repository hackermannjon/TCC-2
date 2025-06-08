"""
Geração de features multimodais
• Imagem  : embeddings da CNN (2048 dims)
• Texto   : embeddings SBERT (384 dims)
• Sexo    : 1 float  (F=1 , M=0 , desconhecido=-1)
• Idade   : 1 float  (normalizado: (idade-18)/50)   → ~0-1
• Personalidade: 5 floats (Big-5 via lexicon)  → ~0-1 cada
Total = 2048 + 384 + 1 + 1 + 5 = 2439 dimensões
Salva:
    data/processed/multimodal_features.npy
    data/processed/selected_profiles.csv   (com profile_id, sex, age, img_file, essay0)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import torchvision.models as models
import torchvision.transforms as T
from src.models.personality import big5_from_text

# --- paths ---
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
IMG_DIR = DATA / "raw" / "images" / "ProfilesDataSet"
CSV_RAW = DATA / "raw" / "okcupid_profiles.csv"        # original Kaggle file
CSV_OUT = DATA / "processed" / "selected_profiles.csv"
NPY_OUT = DATA / "processed" / "multimodal_features.npy"

# --- modelos ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# ResNet50 sem a última camada fully-connected
cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device).eval()
cnn = torch.nn.Sequential(*list(cnn.children())[:-1])

# SBERT para embedding de texto
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Transformações de imagem
tfm = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- carrega CSV raw, filtra quem tem imagem ---
df = pd.read_csv(CSV_RAW)
df = df.reset_index().rename(columns={"index": "profile_id"})

def guess_img(row):
    idx = row.profile_id
    sex = str(row.get("sex", "")).strip().lower()[:1]  # 'm' ou 'f'
    sexes = {"m": "M", "f": "F"}
    races = ("C", "A")  # Caucasian / Asian
    if sex not in sexes:
        return None
    for r in races:
        candidate = IMG_DIR / f"{r}{sexes[sex]}{idx}.jpg"
        if candidate.exists():
            # retorna caminho relativo para salvar no CSV
            return candidate.relative_to(DATA).as_posix()
    return None

# aplica filtro
df["img_file"] = df.apply(guess_img, axis=1)
df = df[df["img_file"].notna()].reset_index(drop=True)

# --- prepara matriz de features ---
n = len(df)
D_IMG = 2048
D_TXT = 384
D_EXTRA = 1 + 1  # sexo + idade
D_BIG5 = 5
TOTAL_DIM = D_IMG + D_TXT + D_EXTRA + D_BIG5
features = np.zeros((n, TOTAL_DIM), dtype="float32")

# --- loop de geração ---
for i, row in tqdm(df.iterrows(), total=n, desc="Gerando features"):
    # imagem
    img_path = DATA / row.img_file
    img = Image.open(img_path).convert("RGB")
    tensor = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        vec_img = cnn(tensor).squeeze().cpu().numpy()  # (2048,)
    # texto
    bio = str(row.get("essay0", "")).strip()
    vec_txt = sbert.encode(bio)  # (384,)
    # extras
    sex_val = 1.0 if row.sex == "f" else 0.0
    age_val = (row.age - 18) / 50 if not np.isnan(row.age) else -1.0
    vec_extra = np.array([sex_val, age_val], dtype="float32")  # (2,)
    # big5
    vec_big5 = big5_from_text(bio)  # (5,)
    # concatena
    features[i] = np.hstack([vec_img, vec_txt, vec_extra, vec_big5])

# --- salva resultados ---
NPY_OUT.parent.mkdir(parents=True, exist_ok=True)
np.save(NPY_OUT, features)
# salva subset de colunas
df[["profile_id", "sex", "age", "img_file", "essay0"]].to_csv(CSV_OUT, index=False)

print("[OK] Multimodal salvo:", NPY_OUT, "shape", features.shape)
print("[OK] CSV salvo:", CSV_OUT)

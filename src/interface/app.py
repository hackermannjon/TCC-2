# src/interface/app.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Ajuste caminho raiz
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

DATA_DIR  = ROOT / "data"
IMG_DIR   = DATA_DIR / "raw" / "images" / "ProfilesDataSet"
FEAT_PATH = DATA_DIR / "processed" / "combined_features.npy"
CSV_PATH  = DATA_DIR / "processed" / "selected_profiles.csv"
LOG_PATH  = DATA_DIR / "logs" / "interactions.csv"
TOP_K     = 20
EXP_BASE  = 0.3

# --- Carregamento cache ---
@st.cache_resource
def load_embeddings() -> np.ndarray:
    return np.load(FEAT_PATH)

@st.cache_resource
def load_profiles() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)

X  = load_embeddings()
df = load_profiles()

# --- Inicializa sessão ---
if "model" not in st.session_state:
    st.session_state.model = PreferenceModel(n_features=X.shape[1])
if "pool" not in st.session_state:
    st.session_state.pool = list(range(len(X)))
if "current" not in st.session_state:
    st.session_state.current = st.session_state.pool.pop(0)
if "likes" not in st.session_state:
    st.session_state.likes = 0
    st.session_state.dislikes = 0

# --- Funções utilitárias ---
def log_interaction(idx: int, like: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"profile_id": idx, "like": int(like), "timestamp": pd.Timestamp.utcnow()}
    ]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def pick_next():
    if not st.session_state.pool:
        st.success("🎉 Fim dos perfis!")
        st.stop()
    n_fb = st.session_state.likes + st.session_state.dislikes
    explore = EXP_BASE / (1 + 0.002 * n_fb)
    pool_idx = np.array(st.session_state.pool[: TOP_K * 2])
    if st.session_state.model._trained:
        feats = X[pool_idx]
        idx_top, _ = rank_candidates(st.session_state.model, feats, 1, explore)
        chosen = int(pool_idx[idx_top[0]])
    else:
        chosen = int(pool_idx[0])
    st.session_state.pool.remove(chosen)
    st.session_state.current = chosen


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# --- Sidebar ---
st.sidebar.title("MatchPredict-AI")
if st.sidebar.button("🔄 Reiniciar Sessão"):
    reset_session()

st.sidebar.metric("Likes", st.session_state.likes)
st.sidebar.metric("Dislikes", st.session_state.dislikes)
progress = (st.session_state.likes + st.session_state.dislikes) / len(X)
st.sidebar.progress(progress, text=f"{len(X)-len(st.session_state.pool)} / {len(X)} vistos")

if st.session_state.model._trained:
    prob = st.session_state.model.predict(X[st.session_state.current: st.session_state.current+1])[0]
    st.sidebar.metric("Prob. de Like", f"{prob:.3f}")
else:
    st.sidebar.info("Dê ao menos 1 Like e 1 Dislike para ativar o modelo.")

# --- Similaridade social ---
vec_curr = X[st.session_state.current].reshape(1, -1)
sims = cosine_similarity(vec_curr, X).flatten()
sims[st.session_state.current] = -1
# Top10 e threshold
top10 = np.sort(sims)[-10:]
mean10 = top10.mean()
count80 = int((sims > 0.8).sum())
st.sidebar.metric("SimTop10 (média)", f"{mean10:.2f}")
st.sidebar.metric("N sim >0.8", count80)

# --- Perfil Atual ---
row = df.iloc[st.session_state.current]
idx = st.session_state.current
# determina caminho da imagem conforme sexo
sex_val = str(row.get("sex", "m")).strip().lower()[:1]
file_candidates = []
if sex_val == "m":
    for race in ("C", "A"): file_candidates.append(IMG_DIR / f"{race}M{idx}.jpg")
else:
    for race in ("C", "A"): file_candidates.append(IMG_DIR / f"{race}F{idx}.jpg")
img_path = next((p for p in file_candidates if p.exists()), None)

col_img, col_txt = st.columns([1,2])
with col_img:
    if img_path:
        st.image(Image.open(img_path), width=260)
    else:
        st.image(Image.new("RGB", (260,260), "gray"), caption="Imagem não disponível")

with col_txt:
    sexo = row.get("sex", "?").capitalize()
    orient = row.get("orientation", "?")
    idade = int(row.get("age", -1)) if not pd.isna(row.get("age")) else "?"
    location = row.get("location", "N/D")
    st.subheader(f"{sexo} – {orient}")
    st.write(f"**Idade:** {idade}  |  **Local:** {location}")
    st.write(row.get("essay0", "(bio não disponível.)"))

like_col, dislike_col = st.columns(2)
with like_col:
    if st.button("👍 Like", use_container_width=True):
        st.session_state.model.update(X[idx], like=True)
        st.session_state.likes += 1
        log_interaction(idx, True)
        pick_next()
        st.rerun()
with dislike_col:
    if st.button("👎 Dislike", use_container_width=True):
        st.session_state.model.update(X[idx], like=False)
        st.session_state.dislikes += 1
        log_interaction(idx, False)
        pick_next()
        st.rerun()

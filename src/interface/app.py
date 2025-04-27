# src/interface/app.py
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

DATA_DIR   = ROOT / "data"
IMG_DIR    = DATA_DIR / "raw" / "images" / "ProfilesDataSet"
FEAT_PATH  = DATA_DIR / "processed" / "multimodal_features.npy"
CSV_PATH   = DATA_DIR / "processed" / "selected_profiles.csv"
LOG_PATH   = DATA_DIR / "logs" / "interactions.csv"
TOP_K      = 20
EXP_BASE   = 0.3

@st.cache_resource
def load_embeddings():
    return np.load(FEAT_PATH)

@st.cache_resource
def load_profiles():
    return pd.read_csv(CSV_PATH)

X  = load_embeddings()
df = load_profiles()

if "model" not in st.session_state:
    st.session_state.model = PreferenceModel(n_features=X.shape[1])
if "pool" not in st.session_state:
    st.session_state.pool = list(range(len(X)))
if "current" not in st.session_state:
    st.session_state.current = st.session_state.pool.pop(0)
if "likes" not in st.session_state:
    st.session_state.likes = 0
    st.session_state.dislikes = 0

# ------------------------------------------------------------

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
        idx, _ = rank_candidates(st.session_state.model, feats, 1, explore)
        chosen = int(pool_idx[idx[0]])
    else:
        chosen = int(pool_idx[0])
    st.session_state.pool.remove(chosen)
    st.session_state.current = chosen


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ---------------- Sidebar ------------------------
st.sidebar.title("MatchPredict‑AI")
if st.sidebar.button("🔄 Reiniciar Sessão"):
    reset_session()

st.sidebar.metric("Likes", st.session_state.likes)
st.sidebar.metric("Dislikes", st.session_state.dislikes)
progress = (st.session_state.likes + st.session_state.dislikes) / len(X)
st.sidebar.progress(progress, text=f"{len(X)-len(st.session_state.pool)} / {len(X)} vistos")

if st.session_state.model._trained:
    p_like = st.session_state.model.predict(X[st.session_state.current: st.session_state.current+1])[0]
    st.sidebar.metric("Prob. de Like", f"{p_like:.3f}")
else:
    st.sidebar.info("Dê pelo menos 1 Like e 1 Dislike para ativar o modelo.")

# ---------------- Perfil Atual -------------------
row = df.iloc[st.session_state.current]
img_path = IMG_DIR / f"{st.session_state.current}.jpg"

col_img, col_txt = st.columns([1,2])
with col_img:
    if img_path.exists():
        st.image(Image.open(img_path), width=260)
    else:
        st.image(Image.new("RGB", (260,260), "gray"))

with col_txt:
    sexo       = row.get("sex", "?")
    orient     = row.get("orientation", "?")
    idade      = int(row.get("age", -1)) if not pd.isna(row.get("age")) else "?"
    location   = row.get("location", "N/D") if "location" in row else "N/D"
    st.subheader(f"{sexo.capitalize()} – {orient}")
    st.write(f"**Idade:** {idade}  |  **Local:** {location}")
    st.write(row.get("essay0", "(bio não disponível)"))

like_col, dislike_col = st.columns(2)
with like_col:
    if st.button("👍 Like", use_container_width=True):
        st.session_state.model.update(X[st.session_state.current], like=True)
        st.session_state.likes += 1
        log_interaction(st.session_state.current, True)
        pick_next()
        st.rerun()

with dislike_col:
    if st.button("👎 Dislike", use_container_width=True):
        st.session_state.model.update(X[st.session_state.current], like=False)
        st.session_state.dislikes += 1
        log_interaction(st.session_state.current, False)
        pick_next()
        st.rerun()

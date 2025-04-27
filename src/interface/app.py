# src/interface/app.py
# Streamlit front‑end completo para MatchPredict‑AI
# • Exibe foto, nome, idade, bio (se existir)
# • Botões Like / Dislike registram feedback e treinam modelo
# • Índice de progresso, score previsto, contadores
# • Log de interações em data/logs/interactions.csv
# ---------------------------------------------------------------------------
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# --- garante import de pacote src ----------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

# ------------------- CONFIG ----------------------------
DATA_DIR = ROOT / "data"
IMG_DIR  = DATA_DIR / "raw" / "images" / "ProfilesDataSet"
FEAT_PATH  = DATA_DIR / "processed" / "multimodal_features.npy"
CSV_PATH   = DATA_DIR / "processed" / "selected_profiles.csv"
LOG_PATH   = DATA_DIR / "logs" / "interactions.csv"
TOP_K      = 20     # candidatos avaliados por rodada
EXP_BASE   = 0.3    # exploração inicial

# ------------------- LOAD DATA -------------------------
@st.cache_resource
def load_embeddings() -> np.ndarray:
    return np.load(FEAT_PATH)

@st.cache_resource
def load_profiles() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    return df

X  = load_embeddings()
df = load_profiles()

# ------------------- SESSION STATE ---------------------
if "model" not in st.session_state:
    st.session_state.model = PreferenceModel(n_features=X.shape[1])

if "pool" not in st.session_state:
    st.session_state.pool = list(range(len(X)))

if "current" not in st.session_state:
    st.session_state.current = st.session_state.pool.pop(0)

if "likes" not in st.session_state:
    st.session_state.likes = 0
    st.session_state.dislikes = 0

# -------------------------------------------------------

def log_interaction(idx: int, like: bool):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"profile_id": idx, "like": int(like), "timestamp": pd.Timestamp.utcnow()}]
    ).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)


def pick_next():
    if not st.session_state.pool:
        st.success("⚡️ Você chegou ao fim dos perfis! Reinicie a sessão para recomeçar.")
        st.stop()

    n_feedback = st.session_state.likes + st.session_state.dislikes
    explore_frac = EXP_BASE / (1 + 0.002 * n_feedback)

    # seleciona um lote inicial
    pool_idx = np.array(st.session_state.pool[: TOP_K * 2])
    feats = X[pool_idx]
    top_idx, _ = rank_candidates(
        st.session_state.model, feats, top_k=1, explore_frac=explore_frac
    )
    chosen = int(pool_idx[top_idx[0]])
    st.session_state.pool.remove(chosen)
    st.session_state.current = chosen


def reset_session():
    for k in ["model", "pool", "current", "likes", "dislikes"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ------------------- UI LAYOUT -------------------------

st.sidebar.title("MatchPredict‑AI")
if st.sidebar.button("🔄 Reiniciar Sessão"):
    reset_session()

st.sidebar.metric("Likes", st.session_state.likes)
st.sidebar.metric("Dislikes", st.session_state.dislikes)
progress = (st.session_state.likes + st.session_state.dislikes) / len(X)
st.sidebar.progress(progress, text=f"{len(X) - len(st.session_state.pool)} / {len(X)} vistos")

if st.session_state.model._trained:
    prob_curr = st.session_state.model.predict(X[st.session_state.current : st.session_state.current + 1])[0]
    st.sidebar.metric("Prob. prevista de Like", f"{prob_curr:.3f}")
else:
    st.sidebar.info("O modelo ainda precisa de um Like e um Dislike para calibrar.")

# --------------- exibe perfil atual --------------------
row = df.iloc[st.session_state.current] if not df.empty else {}
img_path = IMG_DIR / f"{st.session_state.current}.jpg"

col_img, col_txt = st.columns([1, 2])
with col_img:
    if img_path.exists():
        st.image(Image.open(img_path), width=260)
    else:
        st.image(Image.new("RGB", (260, 260), "gray"), caption="Imagem não disponível")

with col_txt:
    st.subheader(row.get("name", "Usuário"))
    if "age" in row:
        st.write(f"**Idade:** {row['age']}")
    if "location" in row:
        st.write(f"**Local:** {row['location']}")
    st.write(row.get("bio", "Bio não disponível."))

# --------------- botões de ação ------------------------
like_col, dislike_col = st.columns(2)
with like_col:
    if st.button("👍 Like", use_container_width=True):
        st.session_state.model.update(X[st.session_state.current], like=True)
        st.session_state.likes += 1
        log_interaction(st.session_state.current, like=True)
        pick_next()
        st.experimental_rerun()

with dislike_col:
    if st.button("👎 Dislike", use_container_width=True):
        st.session_state.model.update(X[st.session_state.current], like=False)
        st.session_state.dislikes += 1
        log_interaction(st.session_state.current, like=False)
        pick_next()
        st.experimental_rerun()

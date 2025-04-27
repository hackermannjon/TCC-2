import sys
from pathlib import Path

# garante que src/ seja reconhecido como pacote
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from src.models.preference_model import PreferenceModel
from src.recommender.core import rank_candidates

# --- Configuração de caminhos ---
DATA_DIR    = ROOT / "data"
FEAT_PATH   = DATA_DIR / "processed" / "multimodal_features.npy"
META_PATH   = DATA_DIR / "processed" / "profiles_metadata.csv"
IMG_DIR     = DATA_DIR / "raw" / "images" / "ProfilesDataSet"
TOP_K       = 20
EXP_FRAC    = 0.2

@st.cache_resource
def load_data():
    X = np.load(FEAT_PATH)
    df = pd.read_csv(META_PATH)   # agora usa o CSV da SCUT
    return X, df

X, df_meta = load_data()

# inicializa sessão
if "model" not in st.session_state:
    st.session_state.model = PreferenceModel(n_features=X.shape[1])
if "pool" not in st.session_state:
    st.session_state.pool = list(df_meta.index)
    st.session_state.current = st.session_state.pool.pop(0)
if "likes" not in st.session_state:
    st.session_state.likes = 0
    st.session_state.dislikes = 0

def pick_next():
    if not st.session_state.pool:
        st.balloons(); st.stop()
    # define um minibatch para ranking
    batch = st.session_state.pool[: TOP_K * 2]
    feats  = X[batch]
    idxs, _ = rank_candidates(
        st.session_state.model, feats, top_k=1, explore_frac=EXP_FRAC
    )
    chosen = batch[idxs[0]]
    st.session_state.pool.remove(chosen)
    st.session_state.current = chosen

# UI
st.title("MatchPredict-AI – Demo SCUT-FBP5500")

row = df_meta.iloc[st.session_state.current]
img_file = IMG_DIR / row["image_name"]

col1, col2 = st.columns([1, 2])
with col1:
    if img_file.exists():
        st.image(Image.open(img_file), width=240)
    else:
        st.image(Image.new("RGB", (240,240), "gray"), caption="Imagem não encontrada")
with col2:
    st.subheader(f"Imagem: {row['image_name']}")
    st.write(f"**Beauty score:** {row['beauty_score']:.2f}")

# botões Like / Dislike
lcol, dcol = st.columns(2)
with lcol:
    if st.button("👍 Like"):
        st.session_state.model.update(X[st.session_state.current], like=True)
        st.session_state.likes += 1
        pick_next()
        st.experimental_rerun()
with dcol:
    if st.button("👎 Dislike"):
        st.session_state.model.update(X[st.session_state.current], like=False)
        st.session_state.dislikes += 1
        pick_next()
        st.experimental_rerun()

# sidebar
st.sidebar.header("Estatísticas")
st.sidebar.metric("Likes", st.session_state.likes)
st.sidebar.metric("Dislikes", st.session_state.dislikes)
if st.session_state.model._trained:
    prob = st.session_state.model.predict(
        X[[st.session_state.current]]
    )[0]
    st.sidebar.metric("Score previsto", f"{prob:.3f}")
else:
    st.sidebar.info("Aguardando ambos os tipos de feedback…")

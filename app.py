"""Streamlit app: draw one Bangla word, get the whole predicted word.

Run with:  streamlit run app.py
"""

import os

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from ocr_utils import ctc_greedy, encode_word, ids_to_word, lexicon_rank, load_labels, nfc, preprocess_drawing

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.keras")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.json")

st.set_page_config(page_title="Bangla Word Recognition", page_icon="✍️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    import keras
    return keras.models.load_model(path, compile=False)


@st.cache_data
def get_labels(path):
    return load_labels(path)


st.title("Bangla Handwritten Word Recognition")
st.caption("Draw one Bangla word in the box, for example মানুষ or জল. "
           "The model reads the whole word and shows the joined word.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Run `python train.py` first.")
    st.stop()

model = load_model(MODEL_PATH)
labels = get_labels(LABELS_PATH)
lexicon = labels.get("lexicon", [])

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Decoding", ["Dictionary (recommended)", "Free (any word)"],
                    help="Dictionary mode picks the closest word from the word list. "
                         "Free mode spells out whatever the model reads.")
    stroke = st.slider("Pen size", 4, 24, 10)
    extra = st.text_area("Extra dictionary words (one per line)", "")
    show_input = st.checkbox("Show model input image", False)
    st.markdown(f"Dictionary size: **{len(lexicon)}** words")

extra_words = [nfc(w.strip()) for w in extra.splitlines() if w.strip()]
bad_extra = [w for w in extra_words if encode_word(w) is None]
if bad_extra:
    st.sidebar.warning("These words use letters the model cannot write and were skipped: "
                       + ", ".join(bad_extra))
words = list(dict.fromkeys(lexicon + [w for w in extra_words if w not in bad_extra]))

canvas = st_canvas(
    stroke_width=stroke,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=240,
    width=700,
    drawing_mode="freedraw",
    return_image_data=True,
    key="canvas",
)

if canvas.image_data is not None:
    rgba = canvas.image_data.astype(np.uint8)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    x = preprocess_drawing(gray) if (gray < 128).sum() > 30 else None

    if x is None:
        st.info("Draw a word above to see the prediction.")
    else:
        logits = model.predict(x[None], verbose=0)[0]
        free_word = ids_to_word(ctc_greedy(logits))
        ranked = lexicon_rank(logits, words, top_k=5) if words else []

        if mode.startswith("Dictionary") and ranked:
            word, conf = ranked[0]
        else:
            word, conf = free_word, None

        st.subheader("Predicted word")
        st.markdown(f"<div style='font-size:64px;font-weight:600;line-height:1.2'>{word or '?'}</div>",
                    unsafe_allow_html=True)
        if conf is not None:
            st.caption(f"Dictionary confidence: {conf:.1%}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Free reading**")
            st.write(free_word or "(nothing read)")
        with col2:
            if ranked:
                st.markdown("**Top dictionary matches**")
                for w, p in ranked:
                    st.write(f"{w}  ·  {p:.1%}")

        if show_input:
            st.image((x[:, :, 0] * 255).astype(np.uint8), caption="Model input (64 x 256)",
                     clamp=True)

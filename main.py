import os
import re
import base64

import numpy as np
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# --- Constants (must match training in SimpleRNN/simplernn.ipynb) ---
VOCAB_SIZE = 10000      # num_words used in imdb.load_data
MAX_LEN = 500           # maxlen used in pad_sequences
INDEX_OFFSET = 3        # IMDB reserves 0=padding, 1=start, 2=unknown
PAD, START, OOV = 0, 1, 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "SimpleRNN", "simple_rnn_imdb.h5")
BG_IMAGE_PATH = os.path.join(BASE_DIR, "Movie Review.jpg")

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="centered")


# --- Cached resources: loaded once, not on every button click ---
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    word_index = imdb.get_word_index()
    return model, word_index


@st.cache_data
def get_base64_of_local_image(image_path):
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


model, word_index = load_resources()


# --- Text preprocessing (mirrors how the IMDB training data was encoded) ---
def tokenize(text):
    # Lowercase and keep only letters, digits and apostrophes,
    # so "fantastic." and "fantastic" map to the same word.
    return re.findall(r"[a-z0-9']+", text.lower())


def encode_word(word):
    idx = word_index.get(word)
    if idx is None:
        return OOV                      # word never seen in IMDB vocabulary
    idx += INDEX_OFFSET
    if idx >= VOCAB_SIZE:
        return OOV                      # rare word outside the top-10k used in training
    return idx


def preprocess_text(text):
    words = tokenize(text)
    encoded = [START] + [encode_word(w) for w in words]   # training sequences start with 1
    padded = sequence.pad_sequences(
        [encoded], maxlen=MAX_LEN, padding="pre", truncating="pre", value=PAD
    )
    known = sum(1 for idx in encoded[1:] if idx != OOV)
    return padded, len(words), known


# --- Prediction ---
def predict_sentiment(text):
    data, n_words, n_known = preprocess_text(text)
    prob_positive = float(model.predict(data, verbose=0)[0][0])
    sentiment = "Positive" if prob_positive > 0.5 else "Negative"
    confidence = prob_positive if sentiment == "Positive" else 1.0 - prob_positive
    return sentiment, confidence, prob_positive, n_words, n_known


# --- Styling ---
bg_image = get_base64_of_local_image(BG_IMAGE_PATH)
bg_css = (
    f"""
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    """
    if bg_image
    else "background-color: #141414;"
)

st.markdown(f"""
    <style>
    .stApp {{
        {bg_css}
        color: white;
    }}
    .title-box {{
        background-color: rgba(30, 30, 30, 0.85);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem auto 1.5rem auto;
        width: 90%;
        max-width: 800px;
    }}
    .title-box h1 {{
        font-size: 2.5rem;
        color: #FFD700;
        margin-bottom: 0.5rem;
    }}
    .description {{
        font-size: 1rem;
        color: #ddd;
    }}
    .stTextArea textarea {{
        font-size: 1.1rem !important;
    }}
    .stButton>button {{
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
    }}
    .footer {{
        text-align: center;
        padding: 2rem 1rem 1rem;
        font-size: 0.9rem;
        color: #cccccc;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("""
    <div class="title-box">
        <h1>🎬 Movie Review Sentiment Analyzer</h1>
        <div class="description">
            A SimpleRNN model trained on 25,000 IMDB reviews.<br>
            Enter any movie review to see whether it's classified as <strong>Positive</strong> or <strong>Negative</strong>.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Input and prediction ---
user_input = st.text_area("📝 Enter your movie review:", height=200)

if st.button("Analyze Review"):
    if not user_input.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        sentiment, confidence, prob_positive, n_words, n_known = predict_sentiment(user_input)

        if n_known == 0:
            st.warning("None of the words in this review are in the model's vocabulary, "
                       "so the prediction is not meaningful.")

        if sentiment == "Positive":
            st.success(f"Sentiment: **{sentiment}** 😊")
        else:
            st.error(f"Sentiment: **{sentiment}** 😞")

        st.progress(confidence, text=f"Confidence: {confidence:.1%}")

        with st.expander("Details"):
            st.write(f"P(positive) from sigmoid output: `{prob_positive:.4f}`")
            st.write(f"Words in review: `{n_words}`  |  recognised by model: `{n_known}`")
            if n_words > MAX_LEN - 1:
                st.write(f"Review was longer than {MAX_LEN - 1} words, so only the last "
                         f"{MAX_LEN - 1} were used.")

# --- Footer ---
st.markdown("""
    <div class="footer">
        © 2025 Movie Sentiment Analyzer | Built with Streamlit & TensorFlow | Harsh Sharma, IIT Bhubaneswar
    </div>
""", unsafe_allow_html=True)

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import base64

# --- Load base64 background image ---
def get_base64_of_local_image(image_path):
    with open(image_path, "rb") as img_file:
        data = img_file.read()
    return base64.b64encode(data).decode()

bg_image = get_base64_of_local_image("Movie Review.jpg")  # Local image path

# --- Load IMDB word index and model ---
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}
model = load_model('SimpleRNN/simple_rnn_imdb.h5')  # Adjust if path is different

# --- Text Preprocessing ---
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    return sequence.pad_sequences([encoded], maxlen=500)

# --- Sentiment Prediction ---
def predict_sentiment(text):
    data = preprocess_text(text)
    score = float(model.predict(data)[0][0])
    return ('Positive' if score > 0.5 else 'Negative'), score

# --- Streamlit Page Config ---
st.set_page_config(page_title="🎬 Movie Review Sentiment Analyzer", layout="centered")

# --- Background and Styling ---
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
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

# --- Title Box ---
st.markdown("""
    <div class="title-box">
        <h1>🎬 Movie Review Sentiment Analyzer</h1>
        <div class="description">
            This analyzer uses a SimpleRNN model trained on IMDB data.<br>
            Enter any movie review to see whether it’s classified as <strong>Positive</strong> or <strong>Negative</strong>.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Input Field and Prediction ---
user_input = st.text_area("📝 Enter your movie review:", height=200)

if st.button("Analyze Review"):
    if user_input.strip():
        sentiment, score = predict_sentiment(user_input)
        st.success(f"Sentiment: **{sentiment}**")
        st.info(f"Confidence Score: `{score:.4f}`")
    else:
        st.warning("Please enter a review before analyzing.")

# --- Footer ---
st.markdown("""
    <div class="footer">
        © 2025 Movie Sentiment Analyzer | Developed using Streamlit & TensorFlow | created by Suryansh Tripathi
    </div>
""", unsafe_allow_html=True)

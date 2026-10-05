# 🎬 Movie Sentiment Analyzer

A Streamlit web app that classifies movie reviews as **Positive** or **Negative** using a **Simple Recurrent Neural Network (SimpleRNN)** trained on the IMDB reviews dataset with TensorFlow/Keras.

🌐 **Live app:** https://movie-sentiment-analyzer-001.streamlit.app/

## 🧠 Features

- Type any movie review and get a real-time sentiment prediction
- Shows a confidence score and the raw sigmoid probability
- Robust preprocessing: punctuation handling, out-of-vocabulary words mapped to the unknown token, same encoding as training
- Model and vocabulary cached so they load only once
- End-to-end pipeline: data loading → preprocessing → training → evaluation → deployment

## 📂 Project structure

```
Movie-Sentiment-analyzer/
├── main.py                     # Streamlit app (inference)
├── evaluate.py                 # Evaluates the saved model on the IMDB test set
├── SimpleRNN/
│   ├── simplernn.ipynb         # Data prep, model building, training
│   ├── embedding.ipynb         # Exploration of one-hot encoding and word embeddings
│   ├── prediction.ipynb        # Loading the model and testing predictions
│   └── simple_rnn_imdb.h5      # Trained model
├── Movie Review.jpg            # App background image
├── requirements.txt
├── runtime.txt
└── README.md
```

## 🔄 How it works

### Training (`SimpleRNN/simplernn.ipynb`)

1. **Data:** IMDB dataset via `keras.datasets.imdb` — 25,000 training and 25,000 test reviews, balanced 50/50, restricted to the 10,000 most frequent words.
2. **Padding:** every review padded/truncated to 500 tokens (pre-padding, so the RNN reads real words last).
3. **Model:**

| Layer | Output shape | Parameters |
|---|---|---|
| Embedding (10,000 → 128) | (None, 500, 128) | 1,280,000 |
| SimpleRNN (128 units, ReLU) | (None, 128) | 32,896 |
| Dense (1, sigmoid) | (None, 1) | 129 |
| **Total** | | **1,313,025** |

4. **Training:** Adam optimizer, binary cross-entropy loss, batch size 32, 20% validation split, `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`.
5. **Saved** as `simple_rnn_imdb.h5`.

### Inference (`main.py`)

1. User enters a review in the Streamlit text area.
2. Text is lowercased and tokenized (punctuation removed).
3. Each word is mapped to its IMDB index + 3; unknown or rare (outside top 10k) words map to the unknown token `2`; a start token `1` is prepended — exactly as in the training data.
4. Sequence is padded to 500 tokens.
5. The model outputs P(positive). Score > 0.5 → **Positive**, otherwise **Negative**. Confidence = P for positive, 1 − P for negative.

## 📊 Results

- **Best validation accuracy:** 85.4% (epoch 3; early stopping restored these weights)
- **Test accuracy:** run `python evaluate.py` to compute accuracy, confusion matrix and precision/recall on the 25,000-review test set

Training accuracy continued rising to ~95% after epoch 3 while validation loss increased, indicating overfitting — which early stopping prevented from reaching the saved model.

## 💻 Run locally

```bash
git clone https://github.com/harsh3-web/Movie-Sentiment-analyzer-main.git
cd Movie-Sentiment-analyzer-main

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
streamlit run main.py
```

To evaluate the model on the test set:

```bash
python evaluate.py
```

## 📝 Example

**Input:** "This movie had stunning visuals and a brilliant performance by the lead actor."
**Output:** ✅ Positive

## ⚠️ Limitations and future work

- **Vanishing gradients:** SimpleRNN struggles to retain information over 500 timesteps; LSTM or GRU layers would capture long-range context better.
- **Training stability:** the ReLU-activated RNN showed a very high first-epoch loss; `tanh` activation or gradient clipping would stabilize training.
- **Overfitting:** dropout / recurrent dropout could improve generalization.
- **Stronger models:** a bidirectional LSTM or a fine-tuned transformer (e.g. DistilBERT) typically reaches 90%+ on IMDB.
- **Domain:** trained only on IMDB movie reviews; may not generalize to other kinds of text, sarcasm, or very short inputs.

## 🛠️ Tech stack

Python 3.10 · TensorFlow / Keras 2.15 · NumPy · scikit-learn (evaluation) · Streamlit

## ✍️ Author

**Harsh Sharma** — B.Tech/M.Tech Dual Degree, Civil Engineering, IIT Bhubaneswar

"""Evaluate the saved SimpleRNN model on the held-out IMDB test set.

Run:  python evaluate.py
Use the printed numbers in the README / resume instead of validation accuracy.
"""
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

VOCAB_SIZE = 10000
MAX_LEN = 500

(_, _), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_LEN)

model = load_model("SimpleRNN/simple_rnn_imdb.h5")
probs = model.predict(X_test, batch_size=256, verbose=1).ravel()
preds = (probs > 0.5).astype(int)

print(f"\nTest accuracy: {accuracy_score(y_test, preds):.4f}\n")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, preds))
print("\nClassification report:")
print(classification_report(y_test, preds, target_names=["Negative", "Positive"], digits=4))

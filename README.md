# 🎬 Movie Sentiment Analyzer 🎭

A Streamlit web application that performs **sentiment analysis** on movie reviews using a trained machine learning model. The tool predicts whether a review expresses a **positive** or **negative** sentiment.

## 🚀 Demo

🌐 Live App: https://movie-sentiment-analyzer-001.streamlit.app/
## 🧠 Features

- Accepts user input as plain text movie reviews.
- Analyzes and classifies sentiment in real-time.
- Uses a pre-trained `TfidfVectorizer` and `LogisticRegression` model.
- User-friendly interface built with **Streamlit**.

## 📂 Project Structure

Movie-Sentiment-analyzer/
├── main.py # Streamlit app
├── movie_sentiment.pkl # Trained logistic regression model
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Dependencies
└── README.md # Project documentation

markdown
Copy code

## 🧪 How It Works

1. The user inputs a movie review.
2. The text is transformed using a `TfidfVectorizer`.
3. The resulting features are passed to a `LogisticRegression` classifier.
4. The app displays whether the sentiment is **Positive** or **Negative**.

## 📌 Technologies Used

- **Python 3.10**
- **Streamlit** – for frontend deployment
- **Scikit-learn** – for training model and vectorizer
- **Pandas, Numpy** – for data handling
- **Pickle** – to load serialized models

## 💻 Run Locally

1. Clone the repository:

```bash
git clone https://github.com/surya01t/Movie-Sentiment-analyzer.git
cd Movie-Sentiment-analyzer
(Optional) Create and activate a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
Install the dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run main.py
🔍 Model Information
Vectorizer: TF-IDF

Classifier: Logistic Regression

Dataset: Preprocessed dataset of labeled movie reviews

Accuracy: ~87% on validation set

📝 Example Usage
Input:

"This movie had stunning visuals and a brilliant performance by the lead actor."

Output:

✅ Positive

✍️ Author
Harsh Sharma
🎓 IIT Bhubaneswar
📬 Your Email or LinkedIn

🌟 Support
If you find this project useful, consider giving it a ⭐️ star!

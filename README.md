# 🎬 Movie Sentiment Analyzer

A machine learning web app that predicts whether a movie review is 
positive or negative.

## 🚀 Live Demo
[Click here to try it live!](https://movie-sentiment-analyzer-v4k7rzwgqy2rweqvbdvjjt.streamlit.app/)

## 📊 Model Performance
- **Dataset:** 50,000 IMDB movie reviews
- **Algorithm:** Logistic Regression + TF-IDF
- **Accuracy:** 89.47% on unseen test data

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Streamlit

## 💡 How It Works
1. Raw review text is cleaned (HTML tags removed, lowercased)
2. Text is converted to numbers using TF-IDF vectorization
3. Logistic Regression model predicts positive or negative
4. Confidence score is displayed with a visual bar

## 📦 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Dataset
Download the IMDB dataset from:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Sentiment Analysis - Flask Web App

A web application for Twitter sentiment analysis powered by a machine learning model.

## About
- Dataset: Sentiment140 (1.6M tweets, subsampled to 100k)
- Models tested: Logistic Regression, Naive Bayes, KNN, LinearSVC
- Best model: Logistic Regression (accuracy: 0.758)
- Deployment: Flask web application

## Tech Stack
- Python, Scikit-learn, NLTK, Flask
- CountVectorizer (max_features=5000)
- PorterStemmer for text preprocessing

## Installation
```bash
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 in your browser.

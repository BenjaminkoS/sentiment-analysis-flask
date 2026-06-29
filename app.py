from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download("stopwords")
import pickle
app = Flask(__name__)

cv=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

def clean_text(text):
    all_stopwords = stopwords.words("english")
    all_stopwords.remove('not')
    all_stopwords.remove("isn't")
    all_stopwords.remove("don't")
    all_stopwords.remove("haven't")
    all_stopwords.remove("couldn't")
    all_stopwords.remove("wouldn't")
    ps=PorterStemmer()
    clean=re.sub('[^a-zA-Z]',' ',text)
    clean=clean.lower()
    clean = clean.split()
    clean = [ps.stem(word) for word in clean if not word in set(all_stopwords)]
    return " ".join(clean)
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    print(f"Method: {request.method}")
    if request.method == 'POST':
        text = request.form['message']
        cleaned = clean_text(text)
        vec = cv.transform([cleaned]).toarray()
        prediction = model.predict(vec)[0]
        print(f"Original: {text}")
        print(f"Cleaned: {cleaned}")
        print(f"Prediction: {prediction}")
        result = 'Positive' if prediction == 4 else 'Negative'
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)
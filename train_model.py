import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib

# Download necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(txt):
    # Lowercase
    txt = txt.lower()
    # Remove punctuation
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    txt = ''.join([i for i in txt if not i.isdigit()])
    # Remove non-ascii (emojis etc)
    txt = txt.encode('ascii', 'ignore').decode('ascii')
    # Remove stopwords
    words = txt.split()
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)

# Load data
df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

# Preprocess
df['text'] = df['text'].apply(preprocess_text)

# Vectorize
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['text'])
y = df['emotion']

# Train Model
svc = LinearSVC()
svc.fit(X, y)

# Save
joblib.dump(svc, 'svc_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and Vectorizer saved successfully!")

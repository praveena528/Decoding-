# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from textblob import TextBlob

# Load dataset
df = pd.read_csv('social_media_posts.csv')

# Preprocess text
df['clean_text'] = df['TextContent'].astype(str).str.lower()
df['clean_text'] = df['clean_text'].str.replace(r'[^\w\s]', '', regex=True)

# Generate sentiment scores
df['polarity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['SentimentLabel']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)  # Ensures convergence
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Sample prediction
sample_text = "I'm really excited about the new product launch!"
sample_vector = vectorizer.transform([sample_text])
prediction = model.predict(sample_vector)[0]
print(f"Predicted Sentiment: {prediction}")

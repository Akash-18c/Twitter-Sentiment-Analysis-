import pandas as pd
import re
import nltk
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)          # remove URLs
    text = re.sub(r'@\w+', '', text)                     # remove mentions
    text = re.sub(r'#\w+', '', text)                     # remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)              # keep only letters
    text = text.lower().strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)

def train_and_save():
    df_train = pd.read_csv('twitter_training.csv', header=None,
                           names=['id', 'topic', 'sentiment', 'text'],
                           encoding='latin-1')
    df_val = pd.read_csv('twitter_validation.csv', header=None,
                         names=['id', 'topic', 'sentiment', 'text'],
                         encoding='latin-1')

    df = pd.concat([df_train, df_val], ignore_index=True)
    df.dropna(subset=['text', 'sentiment'], inplace=True)
    df['clean'] = df['text'].apply(clean_text)
    df = df[df['clean'].str.strip() != '']

    X, y = df['clean'], df['sentiment']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=5, solver='lbfgs'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)

    pipeline.fit(X_train, y_train)
    print(classification_report(y_test, pipeline.predict(X_test)))
    joblib.dump(pipeline, 'sentiment_model.pkl')
    print("Model saved to sentiment_model.pkl")
    return pipeline

def load_model():
    return joblib.load('sentiment_model.pkl')

def predict(texts, model):
    cleaned = [clean_text(t) for t in texts]
    preds = model.predict(cleaned)
    probas = model.predict_proba(cleaned)
    classes = model.classes_
    results = []
    for text, pred, prob in zip(texts, preds, probas):
        conf = round(float(max(prob)) * 100, 1)
        results.append({
            'text': text,
            'sentiment': pred,
            'confidence': conf,
            'scores': {c: round(float(p) * 100, 1) for c, p in zip(classes, prob)}
        })
    return results

if __name__ == '__main__':
    train_and_save()

# Twitter Sentiment Analysis

## Live Demo
**[https://twitter-sentiment-analysis-m52d.onrender.com](https://twitter-sentiment-analysis-m52d.onrender.com)**

A machine learning web app that analyzes the sentiment of tweets using NLP.

## Features
- Analyze any pasted tweet text instantly
- Fetch and analyze tweets by Twitter username
- 4 sentiment classes: Positive, Negative, Neutral, Irrelevant
- 86% model accuracy trained on 73,000+ real tweets
- Clean dark UI with animated loading screen

## Tech Stack
- **Backend:** Python, Flask
- **ML:** Scikit-learn, TF-IDF + Logistic Regression
- **NLP:** NLTK
- **Frontend:** HTML, CSS, JavaScript

## Setup

1. Clone the repo
```bash
git clone https://github.com/Akash-18c/Twitter-Sentiment-Analysis-.git
cd Twitter-Sentiment-Analysis-
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `.env` file
```
RAPIDAPI_KEY=your_rapidapi_key_here
```

4. Train the model
```bash
python model.py
```

5. Run the app
```bash
python app.py
```

6. Open `http://127.0.0.1:5000`

## Dataset
Trained on `twitter_training.csv` and `twitter_validation.csv` containing labeled tweets across 4 sentiment classes.

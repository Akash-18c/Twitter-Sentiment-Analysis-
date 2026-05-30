import os
import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from model import load_model, predict, train_and_save
from collections import Counter

load_dotenv()

app = Flask(__name__)

MODEL_PATH = 'sentiment_model.pkl'
if not os.path.exists(MODEL_PATH):
    print("Model not found. Training now... this may take a minute.")
    model = train_and_save()
else:
    model = load_model()
    print("Model loaded successfully.")

RAPID_API_KEY = os.getenv('RAPIDAPI_KEY', '')


def get_tweets_by_username(username: str, count: int = 10):
    """Fetch tweets via RapidAPI Twitter-API45 endpoint - free tier available."""
    if not RAPID_API_KEY:
        return None, "RapidAPI key not set in .env file."
    try:
        headers = {
            "x-rapidapi-key":  RAPID_API_KEY,
            "x-rapidapi-host": "twitter-api45.p.rapidapi.com"
        }

        # Step 1: get user info
        r = requests.get(
            "https://twitter-api45.p.rapidapi.com/screenname.php",
            headers=headers,
            params={"screenname": username},
            timeout=15
        )
        r.raise_for_status()
        user_data = r.json()
        user_id = user_data.get("id") or user_data.get("rest_id")
        if not user_id:
            return None, f"User @{username} not found."

        # Step 2: get tweets
        r2 = requests.get(
            "https://twitter-api45.p.rapidapi.com/timeline.php",
            headers=headers,
            params={"screenname": username, "count": str(count)},
            timeout=15
        )
        r2.raise_for_status()
        data2 = r2.json()

        tweets = []
        for item in data2.get("timeline", []):
            text = item.get("text") or item.get("full_text", "")
            if text and not text.startswith("RT "):
                tweets.append(text)
            if len(tweets) >= count:
                break

        if not tweets:
            return None, f"No tweets found for @{username}."
        return tweets[:count], None

    except requests.exceptions.HTTPError as e:
        return None, f"API error {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Could not fetch tweets: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = (data or {}).get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    results = predict([text], model)
    return jsonify(results[0])


@app.route('/analyze/username', methods=['POST'])
def analyze_username():
    data = request.get_json()
    username = (data or {}).get('username', '').strip().lstrip('@')
    count = min(int((data or {}).get('count', 10)), 50)
    if not username:
        return jsonify({'error': 'No username provided.'}), 400

    tweets, error = get_tweets_by_username(username, count)
    if error:
        return jsonify({'error': error}), 400

    results = predict(tweets, model)
    counts  = Counter(r['sentiment'] for r in results)
    total   = len(results)
    summary = {s: round(c / total * 100, 1) for s, c in counts.items()}

    return jsonify({'username': username, 'summary': summary, 'tweets': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


import joblib
import numpy as np
from preprocess import process_review
from flask import Flask, request, jsonify

# Create Flask application
app = Flask(__name__)

# Load the saved machine learning model
model = joblib.load('model.joblib')

# Load the saved TF-IDF vectorizer
vectorizer = joblib.load('vectorizer.joblib')


@app.route("/predict", methods=["POST"])
def predict():

    # Read incoming JSON data
    data = request.get_json()

    # Extract customer review
    reviews = data['reviews']

    # Preprocess the review
    process_rev = process_review(reviews)

    # Convert review into numerical features
    process_rev = vectorizer.transform([process_rev])

    prediction = model.predict(process_rev)

    return jsonify({
        "prediction": "positive" if prediction[0] == 1 else "negative"
    })
@app.route("/health", methods=["GET"])
def health():

    return jsonify({"status": "ok"})


# Run Flask application
if __name__ == "__main__":

    import os

    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000))
    )

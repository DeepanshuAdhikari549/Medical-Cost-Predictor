from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model safely
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "model_joblib_gb.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "Medical Cost Predictor is LIVE 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Example input format (adjust based on your model)
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)

    return jsonify({
        "prediction": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

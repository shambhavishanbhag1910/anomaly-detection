from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"


app = Flask(__name__)


scaler = joblib.load(
    MODEL_DIR / "scaler.pkl"
)

iso_model = joblib.load(
    MODEL_DIR / "isolation_forest.pkl"
)

autoencoder = load_model(
    MODEL_DIR / "autoencoder.keras"
)


with open(
    MODEL_DIR / "metadata.json",
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


FEATURES = metadata["features"]
AE_THRESHOLD = metadata[
    "autoencoder_threshold"
]


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": True
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        values = {
            feature: float(payload[feature])
            for feature in FEATURES
        }

        input_df = pd.DataFrame(
            [values]
        )

        X = scaler.transform(
            input_df[FEATURES]
        )


        # Isolation Forest

        iso_raw = iso_model.predict(X)[0]

        iso_anomaly = int(
            iso_raw == -1
        )

        iso_score = float(
            -iso_model.decision_function(X)[0]
        )


        # Autoencoder

        reconstruction = autoencoder.predict(
            X,
            verbose=0
        )

        reconstruction_error = float(
            np.mean(
                np.square(
                    X - reconstruction
                )
            )
        )

        ae_anomaly = int(
            reconstruction_error
            > AE_THRESHOLD
        )


        # Agreement logic

        if iso_anomaly and ae_anomaly:
            priority = "High"

        elif iso_anomaly or ae_anomaly:
            priority = "Medium"

        else:
            priority = "Normal"


        return jsonify({
            "priority": priority,
            "isolation_forest_anomaly": iso_anomaly,
            "isolation_forest_score": iso_score,
            "autoencoder_anomaly": ae_anomaly,
            "reconstruction_error": reconstruction_error,
            "model_agreement": bool(
                iso_anomaly == ae_anomaly
            )
        })


    except (KeyError, TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400


if __name__ == "__main__":
    app.run(
        debug=True
    )
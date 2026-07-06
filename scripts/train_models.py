from pathlib import Path
import json

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_data,
    fit_scaler,
    transform_features,
)

from src.ml_model import (
    build_isolation_forest,
    train_isolation_forest,
)

from src.dl_model import (
    build_autoencoder,
    train_autoencoder,
    reconstruction_errors,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "predictive_maintenance.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)


df = load_data(DATA_PATH)


train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Machine failure"]
)


normal_train = train_df[
    train_df["Machine failure"] == 0
]


X_train, scaler = fit_scaler(normal_train)
X_test = transform_features(test_df, scaler)


# Isolation Forest

iso_model = build_isolation_forest()

iso_model = train_isolation_forest(
    iso_model,
    X_train
)


# Autoencoder
X_ae_train, X_ae_val = train_test_split(
    X_train,
    test_size=0.2,
    random_state=42,
)

autoencoder = build_autoencoder(
    input_dim=X_train.shape[1]
)

train_autoencoder(
    autoencoder,
    X_ae_train
)

val_errors = reconstruction_errors(
    autoencoder,
    X_ae_val
)

threshold = float(
    np.percentile(
        val_errors,
        95
    )
)


# Save artifacts

joblib.dump(
    scaler,
    MODEL_DIR / "scaler.pkl"
)

joblib.dump(
    iso_model,
    MODEL_DIR / "isolation_forest.pkl"
)

autoencoder.save(
    MODEL_DIR / "autoencoder.keras"
)


metadata = {
    "autoencoder_threshold": threshold,
    "features": [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
}

with open(
    MODEL_DIR / "metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=2)


print("Training completed successfully.")
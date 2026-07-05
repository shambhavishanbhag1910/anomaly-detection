from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load and validate the predictive maintenance dataset."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def fit_scaler(df: pd.DataFrame):
    """Fit a StandardScaler using model input features."""

    X = df[FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler
):
    """Apply an existing scaler to input features."""

    X = df[FEATURES].copy()

    return scaler.transform(X)
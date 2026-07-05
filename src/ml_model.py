import numpy as np
from sklearn.ensemble import IsolationForest


def build_isolation_forest(
    contamination: float = 0.034,
    random_state: int = 42
) -> IsolationForest:
    """Create an Isolation Forest model."""

    return IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )


def train_isolation_forest(model, X_train):
    """Train the Isolation Forest."""

    model.fit(X_train)

    return model


def predict_isolation_forest(model, X):
    """Return anomaly labels and anomaly scores."""

    raw_predictions = model.predict(X)

    labels = np.where(
        raw_predictions == -1,
        1,
        0
    )

    scores = -model.decision_function(X)

    return labels, scores
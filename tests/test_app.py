import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")

    assert response.status_code == 200


def test_predict_normal_input(client):
    payload = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "priority" in data
    assert "isolation_forest_anomaly" in data
    assert "autoencoder_anomaly" in data
    assert "model_agreement" in data


def test_predict_missing_field(client):
    payload = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400
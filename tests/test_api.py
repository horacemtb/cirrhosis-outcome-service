"""Check HTTP responses with a real model and an in-process client."""

import pytest
from fastapi.testclient import TestClient

from cirrhosis_service.api import app
from cirrhosis_service.predict import load_model, predict_record


def test_health_and_prediction(model_path, record, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    expected = predict_record(load_model(model_path), record)

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json=record)

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert prediction.status_code == 200
    assert prediction.json() == expected


@pytest.mark.parametrize("endpoint", ["/predict", "/explain"])
def test_invalid_requests_return_422(model_path, record, monkeypatch, endpoint):
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as client:
        assert client.post(endpoint, json={}).status_code == 422

        response = client.post(
            endpoint,
            content='{"Bilirubin": 1e309}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

        record["Age"] = "21532"
        assert client.post(endpoint, json=record).status_code == 422

        record["Age"] = 21532
        record["extra_field"] = 1
        assert client.post(endpoint, json=record).status_code == 422


def test_explanation_matches_prediction(model_path, record, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    with TestClient(app) as client:
        prediction = client.post("/predict", json=record).json()
        response = client.post("/explain", json=record)

    assert response.status_code == 200
    explanation = response.json()
    assert explanation["predicted_status"] == prediction["predicted_status"]
    assert explanation["threshold"] == prediction["threshold"]
    assert explanation["probability_d"] == pytest.approx(prediction["probability_d"])
    assert len(explanation["contributions"]) == 17
    assert explanation["base_value"] + sum(
        row["shap_value"] for row in explanation["contributions"]
    ) == pytest.approx(prediction["probability_d"], abs=1e-6)


def test_missing_model_prevents_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass

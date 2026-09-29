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


def test_invalid_requests_return_422(model_path, record, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as client:
        assert client.post("/predict", json={}).status_code == 422

        response = client.post(
            "/predict",
            content='{"Bilirubin": 1e309}',
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

        record["Age"] = "21532"
        assert client.post("/predict", json=record).status_code == 422

        record["Age"] = 21532
        record["extra_field"] = 1
        assert client.post("/predict", json=record).status_code == 422


def test_missing_model_prevents_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass

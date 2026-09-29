"""Simple checks for data preparation and model prediction."""

import json

import pandas as pd
import pytest

from cirrhosis_service.predict import load_model, predict_record
from cirrhosis_service.preprocessing import validate_features
from cirrhosis_service.train import load_training_data


def test_unknown_category_is_rejected(record):
    record["Drug"] = "unknown"
    with pytest.raises(ValueError, match="Drug"):
        validate_features(pd.DataFrame([record]))


def test_missing_field_is_rejected(record):
    del record["Age"]
    with pytest.raises(ValueError, match="missing"):
        validate_features(pd.DataFrame([record]))


def test_training_removes_excluded_rows_and_columns(tmp_path, record):
    rows = []
    for status in ["C", "D", "C", "D", "CL"]:
        row = record.copy()
        row.update(Status=status, id=1, N_Days=999)
        rows.append(row)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    features, target = load_training_data(csv_path)

    assert target.tolist() == [0, 1, 0, 1]
    assert "Status" not in features
    assert "id" not in features
    assert "N_Days" not in features


def test_saved_model_predicts_consistently(model_path, record):
    first_result = predict_record(load_model(model_path), record)
    second_result = predict_record(load_model(model_path), record)
    metrics_path = model_path.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert first_result == second_result
    assert 0 <= first_result["probability_d"] <= 1
    assert first_result["threshold"] == 0.35
    assert metrics["train_rows"] == 16
    assert metrics["test_rows"] == 4


def test_probability_equal_to_threshold_is_positive(model_path, record):
    model = load_model(model_path)
    result = predict_record(model, record)
    model["threshold"] = result["probability_d"]

    assert predict_record(model, record)["predicted_class"] == 1

"""Small sample inputs shared by the model and API tests."""

import json
from pathlib import Path

import pandas as pd
import pytest

from cirrhosis_service.train import train


@pytest.fixture
def record():
    """Read a fresh copy of the example request for each test."""
    path = Path(__file__).resolve().parents[1] / "examples/predict_request.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def training_csv(tmp_path, record):
    """Write 20 labelled rows for training and tuning tests."""
    rows = []
    for number in range(20):
        row = record.copy()
        row["Bilirubin"] = number + 1
        row["Status"] = "C" if number < 10 else "D"
        rows.append(row)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def model_path(tmp_path, training_csv):
    """Train a real model in a temporary folder."""
    train(training_csv, tmp_path)

    return tmp_path / "model.joblib"

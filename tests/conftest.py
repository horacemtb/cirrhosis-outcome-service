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
def model_path(tmp_path, record):
    """Train a real model on 20 sample rows in a temporary folder."""
    rows = []
    for number in range(20):
        row = record.copy()
        row["Bilirubin"] = number + 1
        row["Status"] = "C" if number < 10 else "D"
        rows.append(row)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    train(csv_path, tmp_path)

    return tmp_path / "model.joblib"

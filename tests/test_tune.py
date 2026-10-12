"""Check optional tuning and reuse of the selected parameters."""

import json
import sys

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from cirrhosis_service.predict import load_model, predict_record
from cirrhosis_service.train import (
    DEFAULT_RF_PARAMS,
    load_training_data,
    main as train_main,
)

pytest.importorskip("optuna")
from cirrhosis_service import tune as tuning  # noqa: E402


def test_recall_uses_the_service_threshold():
    class Model:
        def predict_proba(self, features):
            return np.array([[0.7, 0.3], [0.65, 0.35], [0.6, 0.4]])

    assert tuning.recall_at_threshold(Model(), None, [1, 1, 0]) == 0.5


def test_tuning_keeps_test_rows_out(training_csv, tmp_path, monkeypatch):
    features, target = load_training_data(training_csv)
    _, test_features, _, _ = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )

    def check_training_rows(model, features, target, **kwargs):
        assert len(features) == len(target) == 16
        assert set(features.index).isdisjoint(test_features.index)
        assert kwargs["scoring"] is tuning.recall_at_threshold
        assert kwargs["cv"].n_splits == 2
        assert model.named_steps["sampler"].random_state == 42
        return np.array([0.8, 0.9])

    monkeypatch.setattr(tuning, "cross_val_score", check_training_rows)
    output = tmp_path / "tuning"
    result = tuning.tune(training_csv, output, trials=1, folds=2)

    assert result["best_cv_recall"] == pytest.approx(0.85)
    assert result["train_rows"] == 16
    assert result["trials"] == 1
    assert json.loads((output / "best_params.json").read_text()) == DEFAULT_RF_PARAMS
    assert json.loads((output / "tuning.json").read_text()) == result


def test_tuned_params_train_and_predict(training_csv, tmp_path, record, monkeypatch):
    monkeypatch.setattr(
        tuning,
        "DEFAULT_RF_PARAMS",
        {**DEFAULT_RF_PARAMS, "n_estimators": 50, "max_depth": 3},
    )
    tuning_output = tmp_path / "tuning"
    tuning.tune(training_csv, tuning_output, trials=1, folds=2)
    params_path = tuning_output / "best_params.json"
    params = json.loads(params_path.read_text())
    output = tmp_path / "model"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(training_csv),
            "--params",
            str(params_path),
            "--output",
            str(output),
        ],
    )
    train_main()
    model = load_model(output / "model.joblib")
    forest_params = model["pipeline"].named_steps["rf"].get_params()

    for name, value in params.items():
        assert forest_params[name] == value
    assert forest_params["n_estimators"] == 50
    assert forest_params["max_depth"] == 3
    assert forest_params["bootstrap"] is True
    assert forest_params["random_state"] == 42
    prediction = predict_record(model, record)
    assert 0 <= prediction["probability_d"] <= 1
    assert prediction["threshold"] == 0.35


@pytest.mark.parametrize("trials,folds", [(0, 2), (-1, 2), (1, 1), (1, 20)])
def test_invalid_search_size_is_rejected(training_csv, tmp_path, trials, folds):
    with pytest.raises(ValueError, match="Trials|Folds"):
        tuning.tune(training_csv, tmp_path / "tuning", trials=trials, folds=folds)


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"unknown": 1}', '{"random_state": 1}']
)
def test_invalid_params_file_fails(training_csv, tmp_path, monkeypatch, content):
    params_path = tmp_path / "params.json"
    params_path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(training_csv),
            "--params",
            str(params_path),
            "--output",
            str(tmp_path / "model"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        train_main()
    assert error.value.code == 1
    assert not (tmp_path / "model/model.joblib").exists()

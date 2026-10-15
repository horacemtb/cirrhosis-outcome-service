"""Check that SHAP explains the same score as the prediction pipeline."""

import math

import pytest

from cirrhosis_service.explain import explain_record, make_explainer
from cirrhosis_service.predict import load_model, predict_record
from cirrhosis_service.preprocessing import CATEGORIES, NUMERIC_FEATURES


@pytest.mark.parametrize("bilirubin", [2.3, 20.0])
def test_contributions_match_prediction(model_path, record, bilirubin):
    record["Bilirubin"] = bilirubin
    model = load_model(model_path)
    model["threshold"] = 0.45
    result = explain_record(model, make_explainer(model), record)
    expected = predict_record(model, record)

    assert result["predicted_class"] == expected["predicted_class"]
    assert result["predicted_status"] == expected["predicted_status"]
    assert result["threshold"] == expected["threshold"]
    assert result["probability_d"] == pytest.approx(expected["probability_d"])
    contributions = result["contributions"]
    assert [row["feature"] for row in contributions] == (
        list(CATEGORIES) + ["Stage"] + NUMERIC_FEATURES
    )
    for row in contributions:
        assert row["value"] == record[row["feature"]]
        assert math.isfinite(row["shap_value"])
    assert math.isfinite(result["base_value"])
    reconstructed = result["base_value"] + sum(
        row["shap_value"] for row in contributions
    )
    assert reconstructed == pytest.approx(result["probability_d"], abs=1e-6)

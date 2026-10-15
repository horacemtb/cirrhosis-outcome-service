"""Explain the class-D score for one patient with Tree SHAP."""

import pandas as pd
import shap

from cirrhosis_service.predict import predict_record
from cirrhosis_service.preprocessing import PatientData


def make_explainer(model: dict) -> shap.TreeExplainer:
    return shap.TreeExplainer(
        model["pipeline"].named_steps["rf"],
        feature_perturbation="tree_path_dependent",
        model_output="raw",
    )


def explain_record(
    model: dict, explainer: shap.TreeExplainer, record: dict | PatientData
) -> dict:
    patient = PatientData.model_validate(record).model_dump()
    preprocessor = model["pipeline"].named_steps["preprocess"]
    features = preprocessor.transform(pd.DataFrame([patient]))
    names = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]

    # Random Forest has one output per class; index 1 corresponds to D.
    values = explainer.shap_values(features)[0, :, 1]
    return {
        **predict_record(model, patient),
        "base_value": float(explainer.expected_value[1]),
        "contributions": [
            {"feature": name, "value": patient[name], "shap_value": float(value)}
            for name, value in zip(names, values)
        ],
    }

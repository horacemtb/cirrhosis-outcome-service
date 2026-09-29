"""Load a trained model and predict an outcome."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from cirrhosis_service.preprocessing import PatientData

CLASS_LABELS = {0: "C", 1: "D"}


def load_model(path: str | Path) -> dict:
    """Load the pipeline and threshold saved by train.py."""
    return joblib.load(path)


def predict_record(model: dict, record: dict | PatientData) -> dict:
    patient = PatientData.model_validate(record)
    features = pd.DataFrame([patient.model_dump()])
    probability = float(model["pipeline"].predict_proba(features)[0, 1])
    threshold = model["threshold"]
    predicted_class = int(probability >= threshold)

    return {
        "predicted_class": predicted_class,
        "predicted_status": CLASS_LABELS[predicted_class],
        "probability_d": probability,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict a cirrhosis outcome.")
    parser.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    try:
        model = load_model(args.model)
        record = json.loads(args.input.read_text(encoding="utf-8"))
        result = predict_record(model, record)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Prediction failed: {error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

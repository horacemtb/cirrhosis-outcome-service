"""Train Random Forest and save test metrics."""

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from cirrhosis_service.preprocessing import make_preprocessor, validate_features

RANDOM_STATE = 42
THRESHOLD = 0.35


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, dtype={"Age": "Int64", "Stage": "Int64"})

    if not data["Status"].isin(["C", "D", "CL"]).all():
        raise ValueError("Status must contain only C, D, or CL.")

    data = data[data["Status"] != "CL"].reset_index(drop=True)
    target = data["Status"].map({"C": 0, "D": 1})
    features = data.drop(columns=["Status", "id", "N_Days"], errors="ignore")
    return validate_features(features), target


def make_pipeline() -> Pipeline:
    forest = RandomForestClassifier(
        n_estimators=473,
        max_depth=20,
        min_samples_split=7,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        bootstrap=True,
        max_samples=0.6095011330566877,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("sampler", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("rf", forest)
    ])


def train(data_path: str | Path, output_dir: str | Path) -> dict:
    features, target = load_training_data(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )

    if set(y_train) != {0, 1} or set(y_test) != {0, 1}:
        raise ValueError("Both train and test must contain C and D.")

    logging.info("Training on %d rows; testing on %d.", len(y_train), len(y_test))

    pipeline = make_pipeline()
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    metrics = {
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "class_order": [0, 1],
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = {"pipeline": pipeline, "threshold": THRESHOLD}
    joblib.dump(model, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved model and metrics to %s.", output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the cirrhosis outcome model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        metrics = train(args.data, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Training failed: {error}\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

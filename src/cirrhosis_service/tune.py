"""Choose Random Forest parameters with Optuna on the training data."""

import argparse
import json
import logging
from pathlib import Path

import optuna
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cirrhosis_service.train import (
    DEFAULT_RF_PARAMS,
    RANDOM_STATE,
    THRESHOLD,
    load_training_data,
    make_pipeline,
    split_training_data,
)


def recall_at_threshold(model, features, target) -> float:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = probabilities >= THRESHOLD
    return float(recall_score(target, predictions, zero_division=0))


def tune(
    data_path: str | Path,
    output_dir: str | Path,
    trials: int = 30,
    folds: int = 5,
) -> dict:
    if trials < 1:
        raise ValueError("Trials must be at least 1.")
    if folds < 2:
        raise ValueError("Folds must be at least 2.")

    features, target = load_training_data(data_path)
    x_train, _, y_train, _ = split_training_data(features, target)

    if folds > y_train.value_counts().min():
        raise ValueError("Folds cannot exceed the smallest training class count.")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 800),
            "max_depth": trial.suggest_categorical(
                "max_depth", [3, 5, 7, 10, 15, 20, None]
            ),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", [None, "sqrt", "log2"]
            ),
            "class_weight": trial.suggest_categorical(
                "class_weight", [None, "balanced", "balanced_subsample"]
            ),
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
        }

        scores = cross_val_score(
            make_pipeline(params),
            x_train,
            y_train,
            scoring=recall_at_threshold,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )

        return float(scores.mean())

    logging.info(
        "Tuning on %d rows with %d trials and %d folds.", len(y_train), trials, folds
    )
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.enqueue_trial(DEFAULT_RF_PARAMS)
    study.optimize(objective, n_trials=trials, n_jobs=1)

    result = {
        "best_cv_recall": study.best_value,
        "threshold": THRESHOLD,
        "folds": folds,
        "trials": len(study.trials),
        "random_state": RANDOM_STATE,
        "train_rows": len(y_train),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_params.json").write_text(
        json.dumps(study.best_params, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "tuning.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    logging.info("Saved tuning results to %s.", output_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Tune the cirrhosis outcome model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuning"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        result = tune(args.data, args.output, args.trials, args.folds)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Tuning failed: {error}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

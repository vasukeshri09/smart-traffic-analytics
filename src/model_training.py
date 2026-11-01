
"""Train and evaluate congestion and accident prediction models."""

from pathlib import Path
import warnings
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    label_binarize
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TRAIN_END = pd.Timestamp("2024-03-01 00:00:00")

TARGETS = {
    "congestion": "congestion_target",
    "accident": "accident_target",
}

# These columns must never be used as model inputs.
LEAKAGE_COLUMNS = [
    "congestion_target",
    "accident_target",
    "next_speed",
    "next_speed_ratio",
    "accident_count",
    "timestamp",
]


def load_data():
    """Load the Gold modelling dataset."""

    path = GOLD / "model_dataset.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run data engineering first."
        )

    data = pd.read_parquet(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    return data


def build_preprocessor(X):
    """Create preprocessing for numeric and categorical columns."""

    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical = [
        column for column in X.columns
        if column not in categorical
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numerical),
        ("categorical", categorical_pipeline, categorical),
    ])

    return preprocessor


def get_models(target_name, y_train):
    """Return the available classification models."""

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBClassifier

        if target_name == "accident":
            model = XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
        else:
            model = XGBClassifier(
                n_estimators=120,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="multi:softprob",
                num_class=len(np.unique(y_train)),
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )

        models["xgboost"] = model

    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost.")

    return models


def evaluate_model(model, X_test, y_test, target_name, model_name):
    """Calculate metrics and save evaluation outputs."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="weighted", zero_division=0
    )

    result = {
        "task": target_name,
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
    }

    # Precision-recall evaluation.
    if target_name == "accident":
        positive_probability = probabilities[:, 1]

        result["pr_auc"] = average_precision_score(
            y_test, positive_probability
        )

        result["no_skill_baseline"] = float(
            np.mean(y_test == 1)
        )

        # Capture rate among the 10% highest-risk predictions.
        number_selected = max(1, int(len(y_test) * 0.10))

        top_indices = np.argsort(positive_probability)[-number_selected:]

        actual_accidents = np.sum(np.asarray(y_test) == 1)

        captured = np.sum(
            np.asarray(y_test)[top_indices] == 1
        )

        result["top_10_percent_accident_capture"] = (
            captured / actual_accidents if actual_accidents else 0
        )

    else:
        classes = model.named_steps["classifier"].classes_
        y_binary = label_binarize(y_test, classes=classes)

        result["pr_auc_macro"] = average_precision_score(
            y_binary,
            probabilities,
            average="macro",
        )

    # Save classification report.
    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True,
    )

    report_path = REPORT_DIR / (
        f"{target_name}_{model_name}_classification_report.json"
    )

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    # Save confusion matrix.
    matrix = confusion_matrix(y_test, predictions)

    matrix_path = REPORT_DIR / (
        f"{target_name}_{model_name}_confusion_matrix.csv"
    )

    pd.DataFrame(matrix).to_csv(matrix_path, index=False)

    return result


def train_task(data, target_name, target_column):
    """Train and evaluate models for one prediction task."""

    print("\n" + "=" * 55)
    print(f"TRAINING: {target_name.upper()} PREDICTION")
    print("=" * 55)

    # Chronological split.
    train = data[data["timestamp"] < TRAIN_END].copy()
    test = data[data["timestamp"] >= TRAIN_END].copy()

    # Exclude rows without target labels.
    train = train.dropna(subset=[target_column])
    test = test.dropna(subset=[target_column])

    if train.empty or test.empty:
        raise ValueError(f"Empty train or test dataset for {target_name}")

    # Accident target is integer; congestion target is categorical.
    y_train = train[target_column]
    y_test = test[target_column]

    # Select only predictor columns.
    feature_columns = [
        column for column in data.columns
        if column not in LEAKAGE_COLUMNS
    ]

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # Remove any columns that are entirely missing in training.
    usable_columns = [
        column for column in X_train.columns
        if not X_train[column].isna().all()
    ]

    X_train = X_train[usable_columns]
    X_test = X_test[usable_columns]

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Features:", len(usable_columns))

    preprocessor = build_preprocessor(X_train)
    models = get_models(target_name, y_train)

    results = []

    
    for model_name, classifier in models.items():

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", classifier),
        ])

        # XGBoost requires integer class labels.
        label_encoder = None
        fit_y_train = y_train.copy()
        eval_y_test = y_test.copy()

        if model_name == "xgboost" and target_name == "congestion":

            label_encoder = LabelEncoder()

            fit_y_train = label_encoder.fit_transform(y_train)
            eval_y_test = label_encoder.transform(y_test)

        if model_name == "xgboost":

            weights = compute_sample_weight(
                class_weight="balanced",
                y=fit_y_train,
            )

            pipeline.fit(
                X_train,
                fit_y_train,
                classifier__sample_weight=weights,
            )

        else:
            pipeline.fit(X_train, fit_y_train)

        result = evaluate_model(
            pipeline,
            X_test,
            eval_y_test,
            target_name,
            model_name,
        )

        results.append(result)

        # Save model and its associated label encoder.
        model_path = MODEL_DIR / (
            f"{target_name}_{model_name}.joblib"
        )

        joblib.dump(
            {
                "pipeline": pipeline,
                "features": usable_columns,
                "target": target_column,
                "label_encoder": label_encoder,
            },
            model_path,
        )

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1 score: {result['f1_weighted']:.4f}")

        if target_name == "accident":
            print(f"PR-AUC: {result['pr_auc']:.4f}")
            print(
                "Top 10% accident capture:",
                f"{result['top_10_percent_accident_capture']:.4f}"
            )
        else:
            print(f"Macro PR-AUC: {result['pr_auc_macro']:.4f}")

        print("Saved:", model_path.name)


    return results


def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Gold dataset...")
    data = load_data()

    print("Total records:", len(data))
    print("Training period: Before 1 March 2024")
    print("Testing period: March 2024")

    all_results = []

    for task_name, target_column in TARGETS.items():
        results = train_task(
            data,
            task_name,
            target_column,
        )
        all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETED")
    print("=" * 55)

    print(results_df.to_string(index=False))

    print("\nModels saved in:", MODEL_DIR)
    print("Reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()

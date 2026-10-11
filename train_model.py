
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_lifestyle_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "stress_model.joblib"

FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]

TARGET = "Stress_Level"
VALID_CLASSES = {"Low", "Moderate", "High"}


def load_and_audit_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}\n"
            "Place the CSV inside the data folder and use "
            "the filename student_lifestyle_dataset.csv."
        )

    df = pd.read_csv(DATA_FILE)

    print("\n========== DATASET AUDIT ==========")
    print("Dataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    required = FEATURES + [TARGET]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    print("\nMissing values before cleaning:")
    print(df[required].isnull().sum())

    print("\nDuplicate full rows:", df.duplicated().sum())

    df = df[required].copy()

    for column in FEATURES:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()

    invalid_labels = sorted(
        set(df[TARGET].dropna().unique()) - VALID_CLASSES
    )

    if invalid_labels:
        raise ValueError(
            f"Unexpected stress labels: {invalid_labels}"
        )

    df = df.dropna(subset=required).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows remain after cleaning.")

    if df[TARGET].nunique() != 3:
        raise ValueError(
            "The cleaned dataset must contain Low, Moderate "
            "and High classes."
        )

    if df[TARGET].value_counts().min() < 5:
        raise ValueError(
            "At least five records per class are needed "
            "for the planned stratified evaluation."
        )

    print("\nRows retained after cleaning:", len(df))
    print("\nStress class distribution:")
    print(df[TARGET].value_counts())

    print("\nFeature statistics:")
    print(df[FEATURES].describe().round(2))

    return df


def build_models():
    return {
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=1
            ))
        ]),

        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }


def train_and_evaluate(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = build_models()

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    print("\n========== BASELINE ==========")
    print(
        "Baseline accuracy:",
        round(accuracy_score(y_test, baseline_predictions), 4)
    )
    print(
        "Baseline macro F1:",
        round(
            f1_score(
                y_test,
                baseline_predictions,
                average="macro",
                zero_division=0
            ),
            4
        )
    )

    folds = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    print("\n========== CROSS-VALIDATION ==========")

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=folds,
            scoring="f1_macro",
            n_jobs=1
        )

        mean_score = scores.mean()

        results.append({
            "Model": name,
            "Mean CV Macro F1": mean_score,
            "Standard Deviation": scores.std()
        })

        print(
            f"{name}: "
            f"mean macro F1 = {mean_score:.4f}, "
            f"standard deviation = {scores.std():.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "Mean CV Macro F1",
        ascending=False
    )

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]

    print("\nSelected model:", best_name)

    best_model.fit(X_train, y_train)

    predictions = best_model.predict(X_test)

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4)
    )
    print(
        "Macro F1:",
        round(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ),
            4
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Low", "Moderate", "High"],
            zero_division=0
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["Low", "Moderate", "High"]
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    reference_stats = {}

    for column in FEATURES:
        reference_stats[column] = {
            "mean": float(X_train[column].mean()),
            "median": float(X_train[column].median()),
            "std": float(X_train[column].std()),
            "min": float(X_train[column].min()),
            "max": float(X_train[column].max()),
            "q25": float(X_train[column].quantile(0.25)),
            "q75": float(X_train[column].quantile(0.75))
        }

    bundle = {
        "model": best_model,
        "features": FEATURES,
        "classes": ["Low", "Moderate", "High"],
        "selected_model_name": best_name,
        "reference_stats": reference_stats,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": 42
    }

    joblib.dump(bundle, MODEL_FILE)

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    metrics = {
        "selected_model": best_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "test_rows": len(X_test)
    }

    with open(
        MODEL_DIR / "evaluation_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nSaved model to:", MODEL_FILE)
    print("Saved comparison to:", MODEL_DIR / "model_comparison.csv")
    print(
        "Saved evaluation metrics to:",
        MODEL_DIR / "evaluation_metrics.json"
    )


def main():
    df = load_and_audit_data()
    train_and_evaluate(df)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error)

    input("\nPress Enter to close...")

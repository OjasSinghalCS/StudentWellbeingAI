
from pathlib import Path

import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_lifestyle_dataset.csv"

ALL_FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]

TWO_FEATURES = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day"
]

TARGET = "Stress_Level"


def main():
    df = pd.read_csv(DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        df[ALL_FEATURES],
        df[TARGET],
        test_size=0.20,
        random_state=42,
        stratify=df[TARGET]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for name, features in [
        ("All five lifestyle features", ALL_FEATURES),
        ("Study hours and sleep hours only", TWO_FEATURES)
    ]:
        model = DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        )

        cv_scores = cross_val_score(
            model,
            X_train[features],
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1
        )

        model.fit(X_train[features], y_train)
        predictions = model.predict(X_test[features])

        print("\n================================")
        print(name)
        print("================================")
        print("CV macro F1:", round(cv_scores.mean(), 4))
        print("CV standard deviation:", round(cv_scores.std(), 4))
        print("Test accuracy:", round(
            accuracy_score(y_test, predictions), 4
        ))
        print("Test macro F1:", round(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            ), 4
        ))

        print("\nFeature importances:")
        for feature, importance in zip(
            features, model.feature_importances_
        ):
            print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()

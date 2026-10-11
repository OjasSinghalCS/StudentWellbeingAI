
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "student_lifestyle_dataset.csv"

df = pd.read_csv(DATA_FILE)

features = [
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day"
]

target = "Stress_Level"

print("\n=== STRESS LABELS BY STUDY HOURS ===")
print(
    df.groupby("Stress_Level")["Study_Hours_Per_Day"]
    .agg(["min", "max", "mean", "median"])
    .round(2)
)

print("\n=== STRESS LABELS BY SLEEP HOURS ===")
print(
    df.groupby("Stress_Level")["Sleep_Hours_Per_Day"]
    .agg(["min", "max", "mean", "median"])
    .round(2)
)

print("\n=== COMBINED STUDY AND SLEEP PATTERNS ===")
print(
    df.groupby("Stress_Level")[features]
    .mean()
    .round(2)
)

print("\n=== LABEL CONSISTENCY FOR IDENTICAL INPUT PAIRS ===")

pair_counts = (
    df.groupby(features)[target]
    .nunique()
)

print("Distinct study/sleep pairs:", len(pair_counts))
print(
    "Pairs associated with multiple stress labels:",
    int((pair_counts > 1).sum())
)

print("\n=== SAMPLE RECORDS BY STRESS CATEGORY ===")

for label in ["Low", "Moderate", "High"]:
    print(f"\n{label}:")
    print(
        df.loc[df[target] == label, features]
        .head(10)
        .to_string(index=False)
    )


from pathlib import Path

import joblib
import math
import pandas as pd
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "stress_model.joblib"

app = Flask(__name__)

FEATURES = [
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day"
]

FIELD_DETAILS = [
    {
        "name": "Study_Hours_Per_Day",
        "label": "Study hours",
        "step": "0.1",
        "placeholder": "e.g. 7.5"
    },
    {
        "name": "Extracurricular_Hours_Per_Day",
        "label": "Extracurricular hours",
        "step": "0.1",
        "placeholder": "e.g. 2"
    },
    {
        "name": "Sleep_Hours_Per_Day",
        "label": "Sleep hours",
        "step": "0.1",
        "placeholder": "e.g. 8"
    },
    {
        "name": "Social_Hours_Per_Day",
        "label": "Social hours",
        "step": "0.1",
        "placeholder": "e.g. 3"
    },
    {
        "name": "Physical_Activity_Hours_Per_Day",
        "label": "Physical activity hours",
        "step": "0.1",
        "placeholder": "e.g. 3.5"
    }
]

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

model_bundle = joblib.load(MODEL_FILE)
model = model_bundle["model"]
REFERENCE_STATS = model_bundle.get("reference_stats", {})
MODEL_NAME = model_bundle.get(
    "selected_model_name", "Trained classifier"
)


def get_lifestyle_insights(values):
    insights = []

    for feature in FEATURES:
        value = values[feature]
        stats = REFERENCE_STATS.get(feature, {})

        lower = stats.get("q25")
        upper = stats.get("q75")

        if lower is None or upper is None:
            continue

        label = feature.replace("_", " ").replace(
            " Per Day", ""
        )

        if value < lower:
            insights.append(
                f"{label}: Your reported time is below the "
                f"middle-50% reference range in the dataset "
                f"({lower:.1f}–{upper:.1f} hours per day)."
            )
        elif value > upper:
            insights.append(
                f"{label}: Your reported time is above the "
                f"middle-50% reference range in the dataset "
                f"({lower:.1f}–{upper:.1f} hours per day)."
            )

    return insights


def get_suggestions(values):
    suggestions = []

    study = values["Study_Hours_Per_Day"]
    extracurricular = values["Extracurricular_Hours_Per_Day"]
    sleep = values["Sleep_Hours_Per_Day"]
    social = values["Social_Hours_Per_Day"]
    activity = values["Physical_Activity_Hours_Per_Day"]

    if study < 2:
        suggestions.append(
            "Study: Your reported study time is below the "
            "project's reference range. If you have learning "
            "goals to meet, consider building a manageable "
            "study routine."
        )
    elif study > 8:
        suggestions.append(
            "Study: Your reported study time is above the "
            "project's reference range. Consider balancing "
            "study with breaks, sleep and other activities."
        )

    if extracurricular > 3:
        suggestions.append(
            "Extracurricular activities: You have entered "
            "more than 3 hours per day. Check whether this "
            "fits comfortably alongside your other commitments."
        )

    if sleep < 8:
        suggestions.append(
            "Sleep: Your reported sleep is below the usual "
            "8–10 hours recommended for teenagers. If possible, "
            "make more time for sleep in your routine."
        )
    elif sleep > 10:
        suggestions.append(
            "Sleep: Your reported sleep exceeds the usual "
            "8–10 hour recommendation for teenagers. Individual "
            "needs vary; if you regularly sleep this long and "
            "still feel tired, consider speaking with a trusted "
            "adult or healthcare professional."
        )

    if social < 1:
        suggestions.append(
            "Social time: You have entered less than 1 hour. "
            "If you would like more connection, consider making "
            "time for friends, family or people you trust."
        )
    elif social > 4:
        suggestions.append(
            "Social time: You have entered more than 4 hours. "
            "This is not necessarily a problem; check whether "
            "your routine still leaves enough time for your "
            "other priorities."
        )

    if activity < 1:
        suggestions.append(
            "Physical activity: Your reported activity is "
            "below the commonly recommended 60 minutes of "
            "moderate-to-vigorous activity per day for teenagers. "
            "If appropriate for you, try including enjoyable "
            "movement such as walking, cycling or a sport."
        )
    elif activity > 3:
        suggestions.append(
            "Physical activity: You have entered more than "
            "3 hours. This is not automatically a problem; "
            "make sure your routine allows enough recovery, "
            "food and sleep."
        )

    if not suggestions:
        suggestions.append(
            "No lifestyle flags were identified from the "
            "reference ranges used in this project. These "
            "rules do not establish whether a routine is "
            "healthy or whether someone is experiencing stress."
        )

    suggestions.append(
        "Important: This feedback is based on your entered "
        "routine, not the model's predicted stress category. "
        "The project cannot measure actual psychological stress "
        "or diagnose a health condition."
    )

    return suggestions


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            fields=FIELD_DETAILS,
            form_data={},
            error=None
        )

    values = {}

    try:
        for field in FEATURES:
            raw_value = request.form.get(field, "").strip()

            if not raw_value:
                raise ValueError(
                    "Please enter a value for every activity."
                )

            value = float(raw_value)

            if not math.isfinite(value):
                raise ValueError(
                    "Please enter finite numerical values only."
                )

            if not 0 <= value <= 24:
                raise ValueError(
                    "Each activity must be between 0 and 24 hours."
                )

            values[field] = value

    except (ValueError, TypeError):
        return render_template(
            "index.html",
            fields=FIELD_DETAILS,
            form_data=request.form,
            error=(
                "Enter a valid number between 0 and 24 "
                "for every activity. Leave no fields blank."
            )
        ), 400

    total_hours = sum(values.values())

    if total_hours > 24:
        return render_template(
            "index.html",
            fields=FIELD_DETAILS,
            form_data=request.form,
            error=(
                f"Your activities total {total_hours:.1f} hours. "
                "The total cannot exceed 24 hours per day."
            )
        ), 400

    input_data = pd.DataFrame([values], columns=FEATURES)

    prediction = str(model.predict(input_data)[0]).title()

    insights = get_lifestyle_insights(values)
    suggestions = get_suggestions(values)

    result = {
        "prediction": prediction,
        "inputs": values,
        "insights": insights,
        "suggestions": suggestions,
        "model_name": MODEL_NAME,
        "total_hours": round(total_hours, 1)
    }

    return render_template("results.html", result=result)


if __name__ == "__main__":
    app.run(debug=False)

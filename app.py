"""
STEP 7: BUILDING THE FLASK WEB APPLICATION
--------------------------------------------
Serves:
    GET  /            -> Home page + prediction input form (indexnew.html)
    POST /predict      -> Handles the form submission, runs the model,
                           renders the result page (resultnew.html)
    GET  /country/<c>  -> Optional helper: pre-fills the form from the
                           dataset for a selected country (dropdown)

Run with:
    python app.py
Then open http://127.0.0.1:5000
"""

import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# --------------------------------------------------------------- Load model
with open("model/HDI.pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
FEATURES = artifact["features"]

# --------------------------------------------------------------- Load dataset (for the country dropdown)
df = pd.read_csv("data/hdi_dataset.csv")
COUNTRIES = sorted(df["Country"].dropna().unique().tolist())


def score_to_category(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


@app.route("/")
def home():
    return render_template("indexnew.html", countries=COUNTRIES)


@app.route("/country/<country_name>")
def country_data(country_name):
    """Returns a country's stored feature values as JSON, used by the
    frontend to auto-fill the form when a country is picked from the
    dropdown."""
    row = df[df["Country"] == country_name]
    if row.empty:
        return jsonify({"error": "Country not found"}), 404
    row = row.iloc[0]
    return jsonify(
        {
            "Life_Expectancy": row["Life_Expectancy"],
            "Mean_Years_Schooling": row["Mean_Years_Schooling"],
            "Expected_Years_Schooling": row["Expected_Years_Schooling"],
            "GNI_Per_Capita": row["GNI_Per_Capita"],
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    # ---------------------------------------------------- Input validation
    errors = []
    values = {}
    for feature in FEATURES:
        raw = request.form.get(feature, "").strip()
        try:
            values[feature] = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{feature.replace('_', ' ')}' must be a number.")
            continue

    if not errors:
        if not (0 <= values["Life_Expectancy"] <= 100):
            errors.append("Life expectancy should be between 0 and 100 years.")
        if not (0 <= values["Mean_Years_Schooling"] <= 25):
            errors.append("Mean years of schooling should be between 0 and 25.")
        if not (0 <= values["Expected_Years_Schooling"] <= 25):
            errors.append("Expected years of schooling should be between 0 and 25.")
        if values.get("GNI_Per_Capita", 0) < 0:
            errors.append("GNI per capita cannot be negative.")

    if errors:
        return render_template(
            "indexnew.html", countries=COUNTRIES, errors=errors, form=request.form
        )

    # ---------------------------------------------------- Prediction
    country_name = request.form.get("country", "Custom Input")
    feature_vector = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    predicted_score = float(model.predict(feature_vector)[0])
    predicted_score = max(0.0, min(1.0, predicted_score))
    category = score_to_category(predicted_score)

    return render_template(
        "resultnew.html",
        country=country_name,
        inputs=values,
        score=round(predicted_score, 3),
        category=category,
    )


if __name__ == "__main__":
    app.run(debug=True)

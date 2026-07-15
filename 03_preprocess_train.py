"""
STEP 4: DATA PREPROCESSING & FEATURE ENGINEERING
STEP 5: MACHINE LEARNING MODEL BUILDING (Linear Regression)
STEP 6: MODEL SAVING & SERIALIZATION (Pickle)
-------------------------------------------------------------------------
Loads data/hdi_dataset.csv, cleans it, trains a Linear Regression model to
predict HDI_Score from the four input features, evaluates it, and saves the
trained model + label encoder + feature list to model/HDI.pkl so the Flask
app can load it at runtime without retraining.
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------- Load data
df = pd.read_csv("data/hdi_dataset.csv")
print("Raw shape:", df.shape)
print("Missing values before cleaning:\n", df.isnull().sum())

# --------------------------------------------------------- 4a. Null handling
# Fill numeric nulls with the column mean (as called out in the project spec)
numeric_features = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
]
for col in numeric_features:
    df[col] = df[col].fillna(df[col].mean())

print("\nMissing values after mean imputation:\n", df.isnull().sum())

# --------------------------------------------------------- 4b. Label encoding
# HDI_Category is a categorical label; encode it for reference/analysis
# (the regression itself predicts the continuous HDI_Score).
label_encoder = LabelEncoder()
# Fit in a fixed, meaningful order so the encoded values are ordinal
category_order = ["Low", "Medium", "High", "Very High"]
label_encoder.fit(category_order)
df["HDI_Category_Encoded"] = label_encoder.transform(df["HDI_Category"])

# --------------------------------------------------------- 4c. Feature/label selection
FEATURES = numeric_features  # independent variables
TARGET = "HDI_Score"         # dependent variable

X = df[FEATURES]
y = df[TARGET]

# --------------------------------------------------------- 4d. Train/test split (75/25)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# --------------------------------------------------------- 5. Model training
model = LinearRegression()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# --------------------------------------------------------- 5b. Evaluation
train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)
test_mae = mean_absolute_error(y_test, test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("\n--- Model Evaluation ---")
print(f"Train R-squared : {train_r2:.4f}")
print(f"Test  R-squared : {test_r2:.4f}")
print(f"Test  MAE       : {test_mae:.4f}")
print(f"Test  RMSE      : {test_rmse:.4f}")

print("\nFeature coefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:28s} {coef: .6f}")
print(f"  {'Intercept':28s} {model.intercept_: .6f}")

# --------------------------------------------------------- 5c. Actual vs predicted plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_pred, color="#2563eb", alpha=0.7)
lims = [0, 1]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual HDI Score")
plt.ylabel("Predicted HDI Score")
plt.title(f"Actual vs Predicted HDI Score (Test R2={test_r2:.3f})")
plt.legend()
plt.tight_layout()
plt.savefig("static/plots/actual_vs_predicted.png", dpi=130)
plt.close()

# --------------------------------------------------------- 6. Save with pickle
artifact = {
    "model": model,
    "features": FEATURES,
    "label_encoder": label_encoder,
    "category_order": category_order,
    "metrics": {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    },
}

with open("model/HDI.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\nSaved trained model to model/HDI.pkl")


def score_to_category(score: float) -> str:
    """HDI categories per UNDP convention (used by the Flask app too)."""
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


# Quick sanity check
sample = X_test.iloc[[0]]
pred_score = model.predict(sample)[0]
print(
    f"\nSanity check -> predicted HDI: {pred_score:.3f} "
    f"({score_to_category(pred_score)}), actual: {y_test.iloc[0]:.3f}"
)

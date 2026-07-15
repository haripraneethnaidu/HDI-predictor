"""
STEP 2: DATASET COLLECTION & UNDERSTANDING
-------------------------------------------
This script builds data/hdi_dataset.csv, the dataset used throughout the
project.

NOTE ON THE DATA SOURCE
------------------------
In the real project you will download the HDI dataset from an open source
repository such as Kaggle, e.g.:
    https://www.kaggle.com/datasets/iamsouravbanerjee/human-development-index-dataset
    https://www.kaggle.com/datasets/undp/human-development

Since this environment has no internet access, this script instead BUILDS a
representative dataset for ~110 countries using the same feature set and the
official UNDP HDI methodology (geometric mean of a Life Expectancy Index, an
Education Index, and an Income Index). This lets every downstream step
(EDA, preprocessing, model training, Flask app) run exactly as it would on
the real Kaggle file.

TO USE THE REAL KAGGLE DATA INSTEAD:
    1. Download the CSV from Kaggle.
    2. Rename/rearrange its columns to match:
       Country, Life_Expectancy, Mean_Years_Schooling,
       Expected_Years_Schooling, GNI_Per_Capita, HDI_Score
    3. Save it as data/hdi_dataset.csv, overwriting the generated file.
    4. Re-run 02_eda.py and 03_preprocess_train.py — nothing else changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Base data: (Country, Life Expectancy, Mean Years Schooling,
#             Expected Years Schooling, GNI per Capita)
# Values are approximate, representative figures used to build a realistic
# teaching dataset (not an official UNDP release).
countries_raw = [
    ("Norway", 83.2, 13.0, 18.2, 66000),
    ("Switzerland", 84.0, 13.9, 16.5, 69000),
    ("Ireland", 82.0, 12.7, 18.9, 76000),
    ("Germany", 81.3, 14.2, 17.0, 54000),
    ("Iceland", 83.1, 13.8, 19.2, 55000),
    ("Hong Kong", 85.5, 12.2, 17.3, 62000),
    ("Australia", 83.3, 12.7, 21.1, 49000),
    ("Sweden", 83.0, 12.6, 19.4, 52000),
    ("Netherlands", 82.3, 12.6, 18.6, 55000),
    ("Denmark", 81.4, 13.0, 18.9, 58000),
    ("Finland", 82.2, 12.9, 19.1, 47000),
    ("Singapore", 83.9, 11.9, 16.5, 90000),
    ("United Kingdom", 81.3, 13.4, 17.3, 45000),
    ("Belgium", 82.3, 12.0, 19.8, 49000),
    ("New Zealand", 82.5, 12.8, 19.2, 40000),
    ("Canada", 82.7, 13.4, 16.3, 46000),
    ("United States", 77.2, 13.7, 16.3, 64000),
    ("Austria", 81.6, 12.5, 16.3, 54000),
    ("Japan", 84.8, 13.4, 15.2, 42000),
    ("Israel", 83.0, 13.2, 16.2, 39000),
    ("South Korea", 83.7, 12.5, 16.3, 44000),
    ("France", 82.9, 11.6, 15.6, 47000),
    ("Slovenia", 81.6, 12.6, 17.7, 39000),
    ("Spain", 83.2, 10.3, 17.9, 39000),
    ("Czechia", 79.4, 12.7, 17.4, 39000),
    ("Italy", 83.2, 10.7, 16.3, 42000),
    ("Malta", 83.5, 11.6, 15.4, 41000),
    ("Estonia", 78.7, 13.2, 16.5, 38000),
    ("Cyprus", 81.0, 12.3, 15.1, 39000),
    ("Poland", 76.9, 12.8, 16.7, 33000),
    ("United Arab Emirates", 79.0, 12.1, 15.5, 68000),
    ("Greece", 81.7, 10.5, 17.7, 29000),
    ("Portugal", 81.5, 9.5, 16.7, 34000),
    ("Saudi Arabia", 76.9, 10.4, 16.9, 46000),
    ("Slovakia", 77.9, 12.7, 15.0, 30000),
    ("Chile", 79.4, 10.6, 16.6, 24000),
    ("Croatia", 78.5, 11.9, 15.1, 30000),
    ("Qatar", 80.0, 10.4, 13.4, 90000),
    ("Argentina", 75.4, 11.1, 17.7, 21000),
    ("Hungary", 76.4, 12.0, 15.0, 33000),
    ("Latvia", 75.4, 13.2, 16.0, 30000),
    ("Bahrain", 79.3, 9.8, 16.3, 41000),
    ("Montenegro", 76.9, 11.6, 15.4, 21000),
    ("Romania", 75.6, 11.6, 13.9, 30000),
    ("Kuwait", 78.6, 7.6, 13.6, 60000),
    ("Russia", 72.9, 12.8, 15.6, 27000),
    ("Bulgaria", 75.1, 11.4, 14.6, 24000),
    ("Costa Rica", 78.6, 8.7, 15.7, 18000),
    ("Uruguay", 77.9, 9.0, 16.7, 21000),
    ("Serbia", 74.3, 11.2, 14.6, 17000),
    ("Malaysia", 76.2, 10.6, 13.7, 26000),
    ("Panama", 78.5, 10.4, 13.5, 25000),
    ("Turkey", 77.6, 8.1, 17.6, 28000),
    ("Mexico", 70.9, 9.1, 14.6, 18000),
    ("China", 78.6, 8.1, 14.2, 17000),
    ("Thailand", 79.3, 8.7, 14.7, 17000),
    ("Brazil", 73.4, 8.1, 15.4, 14000),
    ("Georgia", 73.6, 13.1, 14.7, 15000),
    ("Colombia", 73.7, 8.9, 14.4, 14000),
    ("Peru", 72.9, 9.9, 15.0, 11000),
    ("Ukraine", 68.2, 11.3, 15.1, 12000),
    ("Dominican Republic", 74.1, 8.7, 13.5, 17000),
    ("Ecuador", 76.8, 9.2, 14.9, 10000),
    ("Sri Lanka", 77.0, 10.8, 13.9, 12000),
    ("Vietnam", 73.6, 8.3, 12.9, 8000),
    ("Egypt", 70.2, 7.4, 13.4, 11000),
    ("Indonesia", 68.0, 8.6, 13.7, 11000),
    ("Philippines", 71.0, 9.4, 13.1, 8000),
    ("Jordan", 74.5, 10.5, 13.0, 9000),
    ("Jamaica", 71.3, 9.9, 12.9, 9000),
    ("Tunisia", 73.6, 7.3, 15.0, 10000),
    ("South Africa", 65.3, 10.2, 13.6, 12000),
    ("Morocco", 74.0, 5.7, 12.4, 7000),
    ("Bolivia", 65.4, 9.2, 14.2, 8000),
    ("Iraq", 71.0, 7.6, 11.3, 9000),
    ("India", 67.7, 6.7, 12.0, 6500),
    ("Bangladesh", 72.4, 6.2, 11.6, 5300),
    ("Bhutan", 72.0, 4.2, 13.1, 9500),
    ("Nepal", 68.4, 5.1, 12.9, 3800),
    ("Cambodia", 69.6, 5.0, 11.1, 4200),
    ("Kenya", 61.4, 6.6, 12.6, 4400),
    ("Ghana", 64.1, 7.3, 12.1, 5500),
    ("Myanmar", 63.7, 5.0, 9.9, 4300),
    ("Pakistan", 66.1, 4.5, 8.3, 4600),
    ("Zambia", 61.2, 7.2, 11.4, 3400),
    ("Tanzania", 65.5, 6.4, 8.7, 2600),
    ("Rwanda", 66.6, 4.6, 11.3, 2100),
    ("Papua New Guinea", 65.3, 4.7, 10.0, 3700),
    ("Nigeria", 55.4, 7.2, 10.0, 5100),
    ("Senegal", 68.0, 3.4, 9.7, 3400),
    ("Haiti", 64.0, 5.6, 9.4, 2900),
    ("Sudan", 65.7, 3.9, 8.1, 3800),
    ("Ethiopia", 66.6, 3.1, 8.9, 2100),
    ("Uganda", 63.4, 5.2, 11.2, 2200),
    ("Malawi", 63.8, 4.8, 11.0, 1500),
    ("Afghanistan", 62.9, 3.9, 10.3, 1600),
    ("Mozambique", 59.3, 3.5, 9.7, 1300),
    ("Yemen", 63.8, 3.2, 9.0, 1500),
    ("Guinea", 59.8, 2.6, 9.6, 2500),
    ("Sierra Leone", 60.7, 4.1, 10.2, 1900),
    ("Burkina Faso", 59.9, 1.6, 6.9, 2100),
    ("Mali", 58.9, 2.4, 7.6, 2300),
    ("Chad", 53.0, 2.5, 7.7, 1600),
    ("Niger", 61.9, 2.1, 6.6, 1300),
    ("Central African Republic", 54.1, 4.6, 7.6, 1000),
    ("South Sudan", 55.7, 5.7, 5.0, 1500),
    ("Burundi", 63.7, 3.3, 11.5, 800),
    ("DR Congo", 60.4, 7.0, 9.9, 1200),
    ("Somalia", 57.5, 1.9, 5.7, 1400),
]

df = pd.DataFrame(
    countries_raw,
    columns=[
        "Country",
        "Life_Expectancy",
        "Mean_Years_Schooling",
        "Expected_Years_Schooling",
        "GNI_Per_Capita",
    ],
)

# --- Compute HDI sub-indices using the standard UNDP goalpost method -----
LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15          # Mean years of schooling goalpost
EYS_MAX = 18           # Expected years of schooling goalpost
GNI_MIN, GNI_MAX = 100, 75000

life_index = (df["Life_Expectancy"] - LE_MIN) / (LE_MAX - LE_MIN)

education_index = (
    (df["Mean_Years_Schooling"] / MYS_MAX)
    + (df["Expected_Years_Schooling"] / EYS_MAX)
) / 2

income_index = (np.log(df["GNI_Per_Capita"]) - np.log(GNI_MIN)) / (
    np.log(GNI_MAX) - np.log(GNI_MIN)
)

hdi = (life_index * education_index * income_index) ** (1 / 3)

# Small realistic noise so the relationship isn't perfectly deterministic
noise = np.random.normal(0, 0.01, size=len(df))
df["HDI_Score"] = np.clip(hdi + noise, 0, 1).round(3)


def categorize(score):
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


df["HDI_Category"] = df["HDI_Score"].apply(categorize)

# Introduce a handful of missing values on purpose, to be handled in
# preprocessing (step 4) — this mirrors real-world messy data.
rng = np.random.default_rng(7)
for col in ["Life_Expectancy", "Mean_Years_Schooling", "GNI_Per_Capita"]:
    missing_idx = rng.choice(df.index, size=3, replace=False)
    df.loc[missing_idx, col] = np.nan

out_path = "data/hdi_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(df.head(10))
print("\nMissing values per column:")
print(df.isnull().sum())

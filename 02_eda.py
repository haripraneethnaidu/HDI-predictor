"""
STEP 3: DATA VISUALIZATION & ANALYSIS (EDA)
--------------------------------------------
Loads data/hdi_dataset.csv and generates the standard set of exploratory
plots into static/plots/, which the Flask app can also reuse.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
PLOT_DIR = "static/plots"

df = pd.read_csv("data/hdi_dataset.csv")

print("Shape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

numeric_cols = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
    "HDI_Score",
]

# ---------------------------------------------------------------- Distplots
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(22, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2563eb")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/distplots.png", dpi=130)
plt.close()

# ---------------------------------------------------------------- Strip plots
fig, axes = plt.subplots(1, len(numeric_cols) - 1, figsize=(22, 5))
for ax, col in zip(axes, numeric_cols[:-1]):
    sns.stripplot(x="HDI_Category", y=col, data=df, ax=ax,
                  order=["Low", "Medium", "High", "Very High"],
                  palette="viridis")
    ax.set_title(f"{col} by HDI Category")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/stripplots.png", dpi=130)
plt.close()

# ---------------------------------------------------------------- Heatmap / correlation matrix
plt.figure(figsize=(7, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/heatmap.png", dpi=130)
plt.close()

# ---------------------------------------------------------------- Scatter plots vs HDI Score
fig, axes = plt.subplots(1, len(numeric_cols) - 1, figsize=(22, 5))
for ax, col in zip(axes, numeric_cols[:-1]):
    sns.scatterplot(x=col, y="HDI_Score", data=df, ax=ax, hue="HDI_Category",
                     palette="viridis", legend=False)
    ax.set_title(f"HDI Score vs {col}")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/scatterplots.png", dpi=130)
plt.close()

# ---------------------------------------------------------------- HDI Category counts
plt.figure(figsize=(6, 4))
order = ["Low", "Medium", "High", "Very High"]
sns.countplot(x="HDI_Category", data=df, order=order, palette="viridis")
plt.title("Number of Countries per HDI Category")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/category_counts.png", dpi=130)
plt.close()

print(f"\nSaved 5 plots to {PLOT_DIR}/")

import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression


# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame


# Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# Create K-Fold
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Create model
model = LinearRegression()


# Cross-validation
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="r2"
)


# Results
print("Fold R² Scores:")
print(scores)

print("\nAverage R²:", scores.mean())
print("Standard Deviation:", scores.std())

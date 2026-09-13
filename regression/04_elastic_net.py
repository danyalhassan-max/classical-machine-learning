import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame


# Separate features and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Elastic Net pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5))
])


# Train model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)


print("\nElastic Net Regression Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)


# Show coefficients
elastic_model = model.named_steps["model"]

print("\nCoefficients:")
print(elastic_model.coef_)

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

housing = fetch_california_housing(as_frame=True)

df = housing.frame

print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = LinearRegression()
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
print("\nFirst 10 Predictions:")
print(y_pred[:10])

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

print(model.coef_)
print(model.intercept_)


A = df[["MedInc"]]
B = df["MedHouseVal"]

A_train, A_test, B_train, B_test = train_test_split(
    A,
    B,
    test_size=0.2,
    random_state=42
)

poly = PolynomialFeatures(degree=2)

A_train_poly = poly.fit_transform(A_train)
A_test_poly = poly.transform(A_test)

Model = LinearRegression()

Model.fit(A_train_poly, B_train)
B_pred = Model.predict(A_test_poly)

mae = mean_absolute_error(B_test, B_pred)
mse = mean_squared_error(B_test, B_pred)
rmse = mse ** 0.5
r2 = r2_score(B_test, B_pred)

print("\nPolynomial Regression Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

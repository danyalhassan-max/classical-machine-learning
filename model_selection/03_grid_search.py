import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv("bank-full.csv", sep=";")


# Separate features and target
X = df.drop("y", axis=1)
y = df["y"]


# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42
    ))
])


# Hyperparameters to test
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, 20]
}


# Stratified K-Fold
skfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# F1 scorer
f1_scorer = make_scorer(
    f1_score,
    pos_label="yes"
)


# GridSearchCV
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=skfold,
    scoring=f1_scorer,
    n_jobs=-1
)


# Run Grid Search
grid_search.fit(X, y)


# Results
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest F1 Score:")
print(grid_search.best_score_)

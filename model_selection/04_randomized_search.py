import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
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


# Hyperparameter search space
param_grid = {
    "model__n_estimators": [50, 100, 150, 200, 300, 400],
    "model__max_depth": [5, 10, 15, 20, 25, 30, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
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


# Randomized Search
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=10,
    cv=skfold,
    scoring=f1_scorer,
    random_state=42,
    n_jobs=-1
)


# Run search
random_search.fit(X, y)


# Results
print("Best Parameters:")
print(random_search.best_params_)

print("\nBest F1 Score:")
print(random_search.best_score_)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("bank-full.csv", sep=";")

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numerical_features = X.select_dtypes(include=["int64"]).columns
categorical_features = X.select_dtypes(include=["str"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

tree_pipeline = Pipeline([

    ("Preprocessor" , preprocessor),
    ("model" , DecisionTreeClassifier(random_state=42))
]
)

tree_pipeline.fit(X_train , y_train)
y_pred_tree = tree_pipeline.predict(X_test)

print(y_pred_tree)


accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, pos_label="yes")
recall_tree = recall_score(y_test, y_pred_tree, pos_label="yes")
f1_tree = f1_score(y_test, y_pred_tree, pos_label="yes")

print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)
print("F1-Score:", f1_tree)

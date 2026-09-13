import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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

knn_pipeline = Pipeline([

    ("Preprocessor" , preprocessor),
    ("model" , KNeighborsClassifier(n_neighbors=5))
]
)

knn_pipeline.fit(X_train , y_train)
y_pred_knn = knn_pipeline.predict(X_test)

print("Predicition" , y_pred_knn)


accuracy_knn = accuracy_score(y_test, y_pred_knn)
precision_knn = precision_score(y_test, y_pred_knn, pos_label="yes")
recall_knn = recall_score(y_test, y_pred_knn, pos_label="yes")
f1_knn = f1_score(y_test, y_pred_knn, pos_label="yes")

print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1-Score:", f1_knn)

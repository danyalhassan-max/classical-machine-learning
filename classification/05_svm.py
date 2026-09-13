import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
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

svm_pipeline = Pipeline([

      ("Preprocessor" , preprocessor),
      ("model" , SVC())
    
])

svm_pipeline.fit(X_train , y_train)
y_pred_svm = svm_pipeline.predict(X_test)

print("Prediction:", y_pred_svm)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, pos_label="yes")
recall_svm = recall_score(y_test, y_pred_svm, pos_label="yes")
f1_svm = f1_score(y_test, y_pred_svm, pos_label="yes")

print("Accuracy:", accuracy_svm)
print("Precision:", precision_svm)
print("Recall:", recall_svm)
print("F1-Score:", f1_svm)

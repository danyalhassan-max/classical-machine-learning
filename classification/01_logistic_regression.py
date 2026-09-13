import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score



df = pd.read_csv("bank-full.csv" , sep=";")

print(df.head())
print(df.info())
print(df["y"].value_counts())

x = df.drop("y" , axis=1)
y = df["y"]

print(x.head())
print(y.head())


x_train , x_test , y_train , y_test = train_test_split(
   x,
   y,
   test_size=0.2,
   random_state=42,
stratify=y

)
print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

numerical_features = x.select_dtypes(include=["int64"]).columns
categorical_features = x.select_dtypes(include=["str"]).columns

print("Numerical" , numerical_features)
print("Categorical" , categorical_features)

preprocessor = ColumnTransformer(

transformers=[
   ("num" , StandardScaler() , numerical_features) ,
   ("cat" , OneHotEncoder(handle_unknown="ignore"), categorical_features)
]
)

pipeline = Pipeline(
    [
        ("preprocessor" , preprocessor) ,
        ("model" , LogisticRegression())

    ]
)

pipeline.fit(x_train ,y_train)
y_pred = pipeline.predict(x_test)

print(y_pred[:10])
print("Prediction:" ,len(y_pred))
print("Actual" , len(y_test))

accuracy = accuracy_score(y_test , y_pred)
print("Accuracy:", accuracy)


cm = confusion_matrix(y_test , y_pred)
print("Confusion_matric" , cm)

precision = precision_score(y_test, y_pred, pos_label="yes")
print("Precision:", precision)

r = recall_score(y_test , y_pred , pos_label="yes")
print("recall" , r)

f1= f1_score(y_test , y_pred , pos_label="yes")
print("F1_score" , f1)

y_prob = pipeline.predict_proba(x_test)[:, 1]
print(y_prob[:10])

threshold = 0.30
y_pred_30 = ["yes" if p >= threshold else "no" for p in y_prob]

precision_30 = precision_score(y_test, y_pred_30, pos_label="yes")
recall_30 = recall_score(y_test, y_pred_30, pos_label="yes")
f1_30 = f1_score(y_test, y_pred_30, pos_label="yes")

print("Threshold:", threshold)
print("Precision:", precision_30)
print("Recall:", recall_30)
print("F1-Score:", f1_30)

roc_auc = roc_auc_score(y_test , y_prob)
print("Roc_AUC" , roc_auc)


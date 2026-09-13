# Classical Machine Learning

A practical learning repository covering classical machine learning algorithms, model evaluation, model selection, and regularization using Python and Scikit-learn.

This repository documents my hands-on journey through classical machine learning, from regression and classification to cross-validation and hyperparameter optimization.

---

## Overview

This repository focuses on understanding and implementing the core concepts required for practical classical machine learning.

The goal is not only to train models, but to understand:

- How different machine learning algorithms work
- When to use regression vs classification
- How models learn from data
- How to evaluate model performance
- How to handle overfitting and underfitting
- How to validate models reliably
- How to perform hyperparameter tuning
- How to compare different models
- How to build reproducible machine learning workflows

---

## Topics Covered

### Regression

Regression is used when the target variable is continuous.

Implemented algorithms:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

### Classification

Classification is used when the target variable represents categories or classes.

Implemented algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes

### Model Evaluation

Performance evaluation is covered for both regression and classification problems.

#### Regression Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

#### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### Model Selection

Model validation and selection techniques covered in this repository include:

- K-Fold Cross-Validation
- Stratified K-Fold Cross-Validation
- GridSearchCV
- RandomizedSearchCV

These techniques are used to compare models and tune hyperparameters while reducing the risk of selecting a model based on a single data split.

### Overfitting and Underfitting

Core concepts covered:

- Overfitting
- Underfitting
- Generalization
- Bias
- Variance
- Bias-Variance Tradeoff

### Feature Selection

Feature selection techniques covered include:

- Correlation
- SelectKBest
- Recursive Feature Elimination (RFE)
- Feature Importance

---

## Repository Structure

```text
classical-machine-learning/
│
├── classification/
│   ├── 01_logistic_regression.py
│   ├── 02_knn.py
│   ├── 03_decision_tree.py
│   ├── 04_random_forest.py
│   ├── 05_svm.py
│   └── 06_naive_bayes.py
│
├── regression/
│   ├── 01_linear_regression.py
│   ├── 02_ridge.py
│   ├── 03_lasso.py
│   └── 04_elastic_net.py
│
├── model_selection/
│   ├── 01_kfold.py
│   ├── 02_stratified_kfold.py
│   ├── 03_grid_search.py
│   └── 04_randomized_search.py

└── README.md

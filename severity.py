#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install xgboost')

from google.colab import drive
drive.mount("/content/drive")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import pearsonr
import itertools

plt.style.use("ggplot")

# Load the dataset
df = pd.read_csv("/content/drive/MyDrive/PD3.csv")

# Correlational Analysis
corr = np.corrcoef(df.values.T)
sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 8}, yticklabels=df.columns, xticklabels=df.columns)
plt.tight_layout()
plt.show()

# Pearson correlation for each pair of features
for i, j in itertools.combinations(df, 2):
    xc = df[i].values
    yc = df[j].values
    a, b = pearsonr(xc, yc)
    if 0 != b.round(15) < 0.10:
        print("=" * 45)
        print(f"{i}----{j}")
        print(f"correlation coefficient: {a.round(15)}")
        print(f"p-value: {b.round(15)}")

# Feature selection and scaling
x = df[["FIM-M","FIM-C","SI"]]
y = df["FILS-4"].values.ravel()  # Flatten y to 1D array
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# Define a function to perform cross-validation and hyperparameter tuning
def cross_validate_and_tune(model, params, x, y):
    accuracy_list = []
    f1_list = []
    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tuned_model = GridSearchCV(model, params, cv=cv)
        tuned_model.fit(X_train, y_train)
        best = tuned_model.best_estimator_
        best_y = best.predict(X_test)
        accuracy_y = accuracy_score(y_test, best_y)
        f1_y = f1_score(y_test, best_y, average='weighted')
        accuracy_list.append(accuracy_y)
        f1_list.append(f1_y)
    return accuracy_list, f1_list

# Logistic Regression
log_params = {'C': [0.1, 1, 10, 100]}
log = LogisticRegression(multi_class='multinomial', max_iter=500)
log_accuracy, log_f1 = cross_validate_and_tune(log, log_params, x, y)

# Decision Tree Classifier
clf_params = {'max_depth': [3, 4, 5, 10, 15], 'min_samples_leaf': [5, 10, 15], 'min_samples_split': [2, 3, 4, 5]}
clf = DecisionTreeClassifier()
tree_accuracy, tree_f1 = cross_validate_and_tune(clf, clf_params, x, y)

# Support Vector Machine
svc_params = {'C': [1, 2, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
svc = SVC(kernel='rbf', max_iter=500, decision_function_shape='ovo')
svc_accuracy, svc_f1 = cross_validate_and_tune(svc, svc_params, x, y)

# Random Forest Classifier
forest_params = {'n_estimators': [900], 'criterion': ['entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'bootstrap': [False, True]}
forest = RandomForestClassifier()
forest_accuracy, forest_f1 = cross_validate_and_tune(forest, forest_params, x, y)

# XGBoost Classifier
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.5],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]
}
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_accuracy, xgb_f1 = cross_validate_and_tune(xgb, xgb_params, x, y)

# Print results
print("Logistic Regression Accuracy:", log_accuracy)
print("Logistic Regression F1:", log_f1)
print("Decision Tree Accuracy:", tree_accuracy)
print("Decision Tree F1:", tree_f1)
print("SVM Accuracy:", svc_accuracy)
print("SVM F1:", svc_f1)
print("Random Forest Accuracy:", forest_accuracy)
print("Random Forest F1:", forest_f1)
print("XGBoost Accuracy:", xgb_accuracy)
print("XGBoost F1:", xgb_f1)


# In[ ]:


print("Logistic Regression Accuracy:", {np.mean(log_accuracy)})
print("Logistic Regression f1:", {np.mean(log_f1)})
print("Decision Tree Accuracy:", {np.mean(tree_accuracy)})
print("Decision Tree f1:", {np.mean(tree_f1)})
print("SVM Accuracy:", {np.mean(svc_accuracy)})
print("SVM f1:", {np.mean(svc_f1)})
print("Random Forest Accuracy:", {np.mean(forest_accuracy)})
print("Random Forest f1:", {np.mean(forest_f1)})
print("XGBoost Accuracy:", {np.mean(xgb_accuracy)})
print("XGBoost f1:", {np.mean(xgb_f1)})


# In[ ]:


get_ipython().system('python --version')


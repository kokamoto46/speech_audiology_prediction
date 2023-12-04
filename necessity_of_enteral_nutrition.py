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
import itertools
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from scipy import interp
from numpy import mean, std
import joblib

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
x = df[["FIM-M", "FIM-C", "SI"]]
y = df["Tube"].values.ravel()  # Flatten y to 1D array
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
joblib.dump(scaler, '/content/drive/MyDrive/scaler.pkl')  # Save the scaler

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

# Function to perform cross-validation and hyperparameter tuning
def cross_validate_and_tune(model, params, x, y, model_name):
    accuracy_list = []
    roc_list = []
    recall_list = []
    specificity_list = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    best_model = None
    best_score = 0

    for train_index, test_index in cv.split(x, y):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tuned_model = GridSearchCV(model, params, cv=cv)
        tuned_model.fit(X_train, y_train)
        best = tuned_model.best_estimator_
        best_y = best.predict(X_test)
        accuracy_y = accuracy_score(y_test, best_y)
        accuracy_list.append(accuracy_y)
        roc_auc = roc_auc_score(y_test, best_y)
        roc_list.append(roc_auc)
        recall_y = recall_score(y_test, best_y)
        recall_list.append(recall_y)

        # Calculate specificity
        tn, fp, _, _ = confusion_matrix(y_test, best_y).ravel()
        specificity = tn / (tn + fp)
        specificity_list.append(specificity)

        # Save the best model
        if accuracy_y > best_score:
            best_score = accuracy_y
            best_model = best

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, best.predict_proba(X_test)[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Save the best model to a file
    joblib.dump(best_model, f"/content/drive/MyDrive/{model_name}_best_model.pkl")

    # Plot ROC curve
    plt.figure()
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=.8)

    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='black', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    return accuracy_list, roc_list, recall_list, specificity_list, mean_auc

# Define models and their parameters
log = LogisticRegression(multi_class='auto', max_iter=500)
log_params = {'C': [0.1, 1, 10, 100]}

clf = DecisionTreeClassifier()
clf_params = {'max_depth': [3, 4, 5, 10, 15], 'min_samples_leaf': [5, 10, 15], 'min_samples_split': [2, 3, 4, 5]}

svc = SVC(kernel='rbf', probability=True, max_iter=500)
svc_params = {'C': [1, 2, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}

forest = RandomForestClassifier()
forest_params = {'n_estimators': [900], 'criterion': ['entropy'], 'max_depth': [2, 3, 4, 5, 6, 7, 8], 'bootstrap': [False, True]}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.8, 0.9, 1]}

# Perform cross-validation and tuning for each model
log_results = cross_validate_and_tune(log, log_params, x_scaled, y, "logistic_regression")
tree_results = cross_validate_and_tune(clf, clf_params, x_scaled, y, "decision_tree")
svc_results = cross_validate_and_tune(svc, svc_params, x_scaled, y, "svm")
forest_results = cross_validate_and_tune(forest, forest_params, x_scaled, y, "random_forest")
xgb_results = cross_validate_and_tune(xgb, xgb_params, x_scaled, y, "xgboost")


# In[ ]:


# Extracting results for each model
log_accuracy, log_roc, log_recall, log_specificity, _ = log_results
tree_accuracy, tree_roc, tree_recall, tree_specificity, _ = tree_results
svc_accuracy, svc_roc, svc_recall, svc_specificity, _ = svc_results
forest_accuracy, forest_roc, forest_recall, forest_specificity, _ = forest_results
xgb_accuracy, xgb_roc, xgb_recall, xgb_specificity, _ = xgb_results

# Printing the results
print("Logistic Regression Accuracy:", np.mean(log_accuracy))
print("Logistic Regression AUC:", np.mean(log_roc))
print("Logistic Regression Recall:", np.mean(log_recall))
print("Logistic Regression Specificity:", np.mean(log_specificity))
print("Decision Tree Accuracy:", np.mean(tree_accuracy))
print("Decision Tree AUC:", np.mean(tree_roc))
print("Decision Tree Recall:", np.mean(tree_recall))
print("Decision Tree Specificity:", np.mean(tree_specificity))
print("SVM Accuracy:", np.mean(svc_accuracy))
print("SVM AUC:", np.mean(svc_roc))
print("SVM Recall:", np.mean(svc_recall))
print("SVM Specificity:", np.mean(svc_specificity))
print("Random Forest Accuracy:", np.mean(forest_accuracy))
print("Random Forest AUC:", np.mean(forest_roc))
print("Random Forest Recall:", np.mean(forest_recall))
print("Random Forest Specificity:", np.mean(forest_specificity))
print("XGBoost Accuracy:", np.mean(xgb_accuracy))
print("XGBoost AUC:", np.mean(xgb_roc))
print("XGBoost Recall:", np.mean(xgb_recall))
print("XGBoost Specificity:", np.mean(xgb_specificity))


# In[ ]:


print("Logistic Regression Accuracy:",log_accuracy)
print("Logistic Regression AUC:",log_roc)
print("Logistic Regression Recall:", log_recall)
print("Logistic Regression Specificity:",log_specificity)
print("Decision Tree Accuracy:",tree_accuracy)
print("Decision Tree AUC:",tree_roc)
print("Decision Tree Recall:",tree_recall)
print("Decision Tree Specificity:",tree_specificity)
print("SVM Accuracy:",svc_accuracy)
print("SVM AUC:",svc_roc)
print("SVM Recall:",svc_recall)
print("SVM Specificity:",svc_specificity)
print("Random Forest Accuracy:",forest_accuracy)
print("Random Forest AUC:",forest_roc)
print("Random Forest Recall:",forest_recall)
print("Random Forest Specificity:",forest_specificity)
print("XGBoost Accuracy:",xgb_accuracy)
print("XGBoost AUC:",xgb_roc)
print("XGBoost Recall:",xgb_recall)
print("XGBoost Specificity:",xgb_specificity)


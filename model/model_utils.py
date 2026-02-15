
## This module contains helper functions to create machine learning models used in the 
## Breast Cancer Classification project. Training is performed in train_all_models.py.
## This file exists to organize model-related logic and satisfy repository structure requirements.

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_logistic_regression():
    """Create Logistic Regression model"""
    return LogisticRegression(max_iter=5000)


def get_decision_tree():
    """Create Decision Tree model"""
    return DecisionTreeClassifier(random_state=42)


def get_knn():
    """Create K-Nearest Neighbors model"""
    return KNeighborsClassifier(n_neighbors=5)


def get_naive_bayes():
    """Create Gaussian Naive Bayes model"""
    return GaussianNB()


def get_random_forest():
    """Create Random Forest model"""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )


def get_xgboost():
    """Create XGBoost classifier"""
    return XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

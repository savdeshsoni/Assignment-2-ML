# Training all models

# Import libraries

import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef

from data_preprocessing import load_data, split_data, scale_features

# Create model folder if doesn't exist
if not os.path.exists("model"):
    os.makedirs("model")


# Load and Split the data

X, y = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)

# Create test CSV for Streamlit upload

test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv("test_data.csv", index=False)
print("Test file created successfully.")

# Scale for Logistic Regression and K Nearest Neighbor

X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Save the scaler

joblib.dump(scaler, "model/scaler.pkl")

results = []

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 1. Logistic Regression                                                             #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]

results.append(["Logistic Regression",
                accuracy_score(y_test, y_pred_lr),
                roc_auc_score(y_test, y_pred_lr),
                precision_score(y_test, y_pred_lr),
                recall_score(y_test, y_pred_lr),
                f1_score(y_test, y_pred_lr),
                matthews_corrcoef(y_test, y_pred_lr)
                ])

joblib.dump(lr, "model/logistic.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 2. Decision Tree                                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

results.append(["Decision Tree",
                accuracy_score(y_test, y_pred_dt),
                roc_auc_score(y_test, y_pred_dt),
                precision_score(y_test, y_pred_dt),
                recall_score(y_test, y_pred_dt),
                f1_score(y_test, y_pred_dt),
                matthews_corrcoef(y_test, y_pred_dt)
                ])

joblib.dump(dt, "model/decision_tree.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 3. KNN                                                                          #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

results.append(["KNN",
                accuracy_score(y_test, y_pred_knn),
                roc_auc_score(y_test, y_prob_knn),
                precision_score(y_test, y_pred_knn),
                recall_score(y_test, y_pred_knn),
                f1_score(y_test, y_pred_knn),
                matthews_corrcoef(y_test, y_pred_knn)
                ])

joblib.dump(knn, "model/knn.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 4. Naive Bayes                                                                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)[:, 1]

results.append(["Naive Bayes",
                accuracy_score(y_test, y_pred_nb),
                roc_auc_score(y_test, y_prob_nb),
                precision_score(y_test, y_pred_nb),
                recall_score(y_test, y_pred_nb),
                f1_score(y_test, y_pred_nb),
                matthews_corrcoef(y_test, y_pred_nb)
                ])

joblib.dump(nb, "model/naive_bayes.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 5. Random Forest                                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

results.append(["Random Forest",
                accuracy_score(y_test, y_pred_rf),
                roc_auc_score(y_test, y_prob_rf),
                precision_score(y_test, y_pred_rf),
                recall_score(y_test, y_pred_rf),
                f1_score(y_test, y_pred_rf),
                matthews_corrcoef(y_test, y_pred_rf)
                ])

joblib.dump(rf, "model/random_forest.pkl")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# 6. XGBoost                                                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    use_label_encoder=False
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

results.append(["XGBoost",
                accuracy_score(y_test, y_pred_xgb),
                roc_auc_score(y_test, y_prob_xgb),
                precision_score(y_test, y_pred_xgb),
                recall_score(y_test, y_pred_xgb),
                f1_score(y_test, y_pred_xgb),
                matthews_corrcoef(y_test, y_pred_xgb)
                ])

joblib.dump(xgb, "model/xgboost.pkl")

# Print comparison Table

results_df = pd.DataFrame(results, columns=[
    "Model",
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC"
])

print("\nMODEL COMPARISON TABLE\n")
print(results_df)
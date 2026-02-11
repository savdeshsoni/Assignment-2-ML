import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Breast Cancer Classification", layout="wide")

st.title("Breast Cancer Classification - ML Deployment")

# Load models

logistic = load("model/logistic.pkl")
decision_tree = load("model/decision_tree.pkl")
knn = load("model/knn.pkl")
naive_bayes = load("model/naive_bayes.pkl")
random_forest = load("model/random_forest.pkl")
xgboost = load("model/xgboost.pkl")
scaler = load("model/scaler.pkl")

models = {
    "Logistic Regression": logistic,
    "Decision Tree": decision_tree,
    "KNN": knn,
    "Naive Bayes": naive_bayes,
    "Random Forest": random_forest,
    "XGBoost": xgboost
}

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
model_choice = st.selectbox("Select Model", list(models.keys()))

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "target" not in data.columns:
        st.error("CSV must contain a 'target' column.")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        model = models[model_choice]

        if model_choice in ["Logistic Regression", "KNN"]:
            X = scaler.transform(X)

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
        else:
            y_prob = None

        st.subheader("Evaluation Metrics")

        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("Precision:", precision_score(y, y_pred))
        st.write("Recall:", recall_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred))
        st.write("MCC:", matthews_corrcoef(y, y_pred))

        if y_prob is not None:
            st.write("AUC Score:", roc_auc_score(y, y_prob))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve
)
from sklearn.model_selection import train_test_split

from train import (
    load_data,
    preprocess_data,
    train_autoencoder,
    train_lstm_autoencoder,
    train_isolation_forest,
    train_oneclass_svm,
    train_lof,
    train_hbos
)

# Streamlit setup
st.set_page_config(page_title="Network Intrusion Detection", layout="wide")
st.title("🔐 Network Intrusion Detection System (KDD Dataset)")

# Sidebar
st.sidebar.title("🧠 Model & Data Selection")

model_choices = [
    "Autoencoder",
    "LSTM Autoencoder",
    "Isolation Forest",
    "One-Class SVM",
    "Local Outlier Factor (LOF)",
    "Histogram-based Outlier Score (HBOS)"
]

selected_models = st.sidebar.multiselect(
    "Select one or more anomaly detection models",
    model_choices,
    default=["Isolation Forest"]
)

data_source = st.sidebar.radio("Select Input Source", ["Demo KDD Data", "Upload Your CSV"])

uploaded_file = None
if data_source == "Upload Your CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def get_data():
    if data_source == "Demo KDD Data":
        return load_data()
    elif uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        return None

# Load data
with st.spinner("🔄 Loading and preprocessing data..."):
    df = get_data()
    if df is None:
        st.warning("⚠️ Please upload a CSV file.")
        st.stop()

    # Fast mode
    if st.sidebar.checkbox("🧪 Fast Mode (20K rows)") and len(df) > 20000:
        df = df.sample(20000, random_state=42)

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train_norm = X_train[y_train == 0]

    st.success(f"✅ Data loaded! Rows: {len(df)}, Features: {X.shape[1]}")

# Run models
for model_name in selected_models:
    st.subheader(f"🔍 Results for {model_name}")
    try:
        if model_name == "Autoencoder":
            preds, scores = train_autoencoder(X_train_norm, X_test, y_test)
        elif model_name == "LSTM Autoencoder":
            preds, scores = train_lstm_autoencoder(X_train_norm, X_test, y_test)
        elif model_name == "Isolation Forest":
            preds, scores = train_isolation_forest(X_train_norm, X_test, y_test)
        elif model_name == "One-Class SVM":
            preds, scores = train_oneclass_svm(X_train_norm, X_test, y_test)
        elif model_name == "Local Outlier Factor (LOF)":
            preds, scores = train_lof(X_train_norm, X_test, y_test)
        elif model_name == "Histogram-based Outlier Score (HBOS)":
            preds, scores = train_hbos(X_train_norm, X_test, y_test)
        else:
            st.error(f"Unknown model: {model_name}")
            continue

        y_test_np = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test

        roc_auc = roc_auc_score(y_test_np, scores)
        accuracy = accuracy_score(y_test_np, preds)
        report = classification_report(y_test_np, preds, output_dict=True)
        cm = confusion_matrix(y_test_np, preds)

        # Metrics display
        col1, col2 = st.columns(2)
        col1.metric("ROC AUC", f"{roc_auc:.4f}")
        col2.metric("Accuracy", f"{accuracy:.4f}")

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

        # ROC Curve
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test_np, scores)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend()
        st.pyplot(fig_roc)

        # Score Distribution
        st.markdown("#### Anomaly Score Distribution")
        fig_score, ax_score = plt.subplots()
        sns.histplot(scores, bins=50, kde=True, color='green', ax=ax_score)
        ax_score.set_xlabel("Anomaly Score")
        st.pyplot(fig_score)

        # Prediction Breakdown
        st.markdown("#### Prediction Breakdown")
        pie_df = pd.Series(preds).value_counts().rename({0: "Normal", 1: "Anomaly"})
        st.bar_chart(pie_df)

        # Classification Report
        st.markdown("#### Classification Report")
        st.dataframe(pd.DataFrame(report).transpose().round(3))

        # Class-wise metrics
        st.markdown("#### Class-wise Metrics (Precision, Recall, F1)")
        class_metrics = pd.DataFrame(report).transpose().loc[["0", "1"], ["precision", "recall", "f1-score"]]
        st.bar_chart(class_metrics)

    except Exception as e:
        st.error(f"❌ Failed to run {model_name}: {str(e)}")

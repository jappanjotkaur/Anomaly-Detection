import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve

from pyod.models.hbos import HBOS

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)

results = []  # Store results for summary comparison

# --- Load KDD Dataset ---
def load_data():
    data_path = "data/kddcup.data_10_percent_corrected"
    col_names = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
                 "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                 "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
                 "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
                 "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
                 "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                 "dst_host_rerror_rate","dst_host_srv_rerror_rate"]
    df = pd.read_csv(data_path, header=None)
    df.columns = col_names + ["target"]
    return df

# --- Preprocessing ---
def preprocess_data(df):
    df['target'] = df['target'].apply(lambda x: 0 if x == 'normal.' else 1)
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = LabelEncoder().fit_transform(df[col])
    X = df.drop('target', axis=1)
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# --- Visualization Functions ---
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, scores, title):
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, scores):.4f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_score_histogram(scores, title):
    plt.figure()
    sns.histplot(scores, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.tight_layout()
    plt.show()

def show_class_breakdown(y_pred, title):
    unique, counts = np.unique(y_pred, return_counts=True)
    plt.figure()
    plt.bar(["Normal", "Anomaly"], counts)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_classification_table(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

# --- Wrapper for all models ---
def evaluate_model(name, y_true, y_pred, scores):
    print(f"\n--- {name} ---")
    print_classification_table(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, scores)
    print("Accuracy:", acc)
    print("ROC AUC:", roc)
    plot_conf_matrix(y_true, y_pred, f"Confusion Matrix - {name}")
    plot_roc(y_true, scores, f"ROC Curve - {name}")
    plot_score_histogram(scores, f"Anomaly Score Histogram - {name}")
    show_class_breakdown(y_pred, f"Prediction Breakdown - {name}")

    # Save metrics
    results.append({'Model': name, 'Accuracy': acc, 'ROC AUC': roc})

# --- Models ---
def train_isolation_forest(X_train_norm, X_test, y_test):
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    model.fit(X_train_norm)
    scores = -model.decision_function(X_test)
    preds = np.where(model.predict(X_test) == 1, 0, 1)
    return preds, scores

def train_oneclass_svm(X_train_norm, X_test, y_test):
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    model.fit(X_train_norm)
    scores = -model.decision_function(X_test)
    preds = np.where(model.predict(X_test) == 1, 0, 1)
    return preds, scores

def train_lof(X_train_norm, X_test, y_test):
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    model.fit(X_train_norm)
    scores = -model.decision_function(X_test)
    preds = np.where(model.predict(X_test) == 1, 0, 1)
    return preds, scores

def train_hbos(X_train_norm, X_test, y_test):
    model = HBOS(contamination=0.1)
    model.fit(X_train_norm)
    scores = model.decision_function(X_test)
    preds = model.predict(X_test)
    return preds, scores

def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    bottleneck = Dense(32, activation='relu')(x)
    x = Dense(64, activation='relu')(bottleneck)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(input_dim, activation='linear')(x)
    autoencoder = Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder(X_train_norm, X_test, y_test):
    model = build_autoencoder(X_train_norm.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_norm, X_train_norm, epochs=100, batch_size=256, shuffle=True, 
              validation_split=0.1, callbacks=[early_stop], verbose=0)

    recon_train = model.predict(X_train_norm)
    train_mse = np.mean(np.square(X_train_norm - recon_train), axis=1)
    threshold = np.percentile(train_mse, 90)

    recon_test = model.predict(X_test)
    test_mse = np.mean(np.square(X_test - recon_test), axis=1)
    preds = (test_mse > threshold).astype(int)

    return preds, test_mse

def build_lstm_autoencoder(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))
    encoded = LSTM(64, activation='relu', return_sequences=False)(inputs)
    repeated = RepeatVector(timesteps)(encoded)
    decoded = LSTM(64, activation='relu', return_sequences=True)(repeated)
    outputs = TimeDistributed(Dense(n_features))(decoded)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm_autoencoder(X_train_norm, X_test, y_test):
    timesteps = 1
    X_train_reshaped = X_train_norm.reshape((X_train_norm.shape[0], timesteps, X_train_norm.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, X_test.shape[1]))
    model = build_lstm_autoencoder(timesteps, X_train_norm.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train_reshaped, X_train_reshaped, epochs=50, batch_size=256,
              validation_split=0.1, callbacks=[early_stop], verbose=0)
    recon_train = model.predict(X_train_reshaped)
    train_mse = np.mean(np.square(X_train_reshaped - recon_train), axis=(1, 2))
    threshold = np.percentile(train_mse, 90)
    recon_test = model.predict(X_test_reshaped)
    test_mse = np.mean(np.square(X_test_reshaped - recon_test), axis=(1, 2))
    preds = (test_mse > threshold).astype(int)
    return preds, test_mse

# --- Final Comparison ---
def show_final_comparison():
    if not results:
        print("⚠️ No results to display. Ensure all models were evaluated.")
        return
    df = pd.DataFrame(results)
    print("\n===== Final Comparison Table =====")
    print(df)
    try:
        melted = df.melt(id_vars="Model")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=melted, x="Model", y="value", hue="variable")
        plt.title("Model Performance Comparison")
        plt.ylabel("Score")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", label_type="edge")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error during plotting: {e}")

# --- Main ---
def main():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_train_norm = X_train[y_train == 0]

    models = [
        ("Isolation Forest", train_isolation_forest),
        ("One-Class SVM", train_oneclass_svm),
        ("Local Outlier Factor", train_lof),
        ("HBOS", train_hbos),
        ("Autoencoder", train_autoencoder),
        ("LSTM Autoencoder", train_lstm_autoencoder)
    ]

    for name, func in models:
        preds, scores = func(X_train_norm, X_test, y_test)
        evaluate_model(name, y_test, preds, scores)

    show_final_comparison()

if __name__ == "__main__":
    main()

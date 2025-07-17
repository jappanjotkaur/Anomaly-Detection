# Anomaly Detection in Network Traffic

An interactive Streamlit-based anomaly detection dashboard for identifying **network intrusions** using the **KDD Cup 99** dataset. This app enables users to explore, compare, and visualize results from multiple anomaly detection models including deep learning and traditional ML techniques.

---

## 🚀 Features

- 📁 **Data Sources**:
  - Use preloaded KDD dataset (10% subset)
  - Upload your own custom CSV files

- 🧠 **Models Supported**:
  - Autoencoder (Feedforward Neural Network)
  - LSTM Autoencoder
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor (LOF)
  - Histogram-based Outlier Score (HBOS)

- 📊 **Visual Outputs**:
  - Confusion matrix
  - ROC curve
  - Anomaly score distributions
  - Class-wise metrics and bar charts
  - Classification report (precision, recall, F1-score)

- ⚡ Fast Mode: Quickly analyze a 20K-row subset for speed

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/network-intrusion-detection.git
cd network-intrusion-detection
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure

```
.
├── app.py              # Streamlit UI code
├── train.py            # Contains model definitions and training logic
├── requirements.txt    # Dependencies
├── data/
│   └── kddcup.data_10_percent_corrected  # Preloaded KDD dataset
```

---

## 📊 Model Evaluation

For each selected model, the app displays:

- ROC AUC & Accuracy
- Confusion Matrix
- ROC Curve
- Anomaly Score Histogram
- Prediction Breakdown (Normal vs Anomaly)
- Precision, Recall, F1-score table
- Bar chart of class-wise metrics

---

## 📁 Data Format

Expected columns:
- If using your own CSV, make sure:
  - Target column is binary (`0` for normal, `1` for anomaly)
  - Categorical fields are encoded (or match KDD format: `protocol_type`, `service`, `flag`)
  - No missing values

---

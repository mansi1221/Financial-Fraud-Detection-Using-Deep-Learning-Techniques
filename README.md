# 🛡️ Financial Fraud Detection System
### Using Deep Learning & Anomaly Detection

A deep learning–based system that analyzes credit card transaction data to detect fraudulent activities using **ANN, CNN, LSTM, and Autoencoder** models.

---

## 📌 Project Summary

This system combines **supervised deep learning** (ANN, CNN, LSTM) with **unsupervised anomaly detection** (Autoencoder) to detect both known fraud patterns and novel/unseen fraud behaviors. It is trained on the Kaggle Credit Card Fraud Detection dataset (284,807 transactions, 0.17% fraud rate).

**Key Highlights:**
- Handles extreme class imbalance using **SMOTE**
- **Autoencoder** detects unknown fraud via reconstruction error
- Compares 4 model architectures with ROC curves and metrics
- Generates publication-ready visualizations

---

## 🧠 Models Used

| Model | Type | Purpose |
|-------|------|---------|
| **ANN** | Supervised | Baseline non-linear classifier |
| **CNN (1D)** | Supervised | Local pattern extraction |
| **LSTM** | Supervised | Temporal/sequential pattern detection |
| **Autoencoder** | Unsupervised | Anomaly detection (novel fraud) |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` folder.

### 3. Run the System
```bash
python main.py
```

---

## 📁 Project Structure
```
financial_fraud_detection/
├── data/                        # Dataset (creditcard.csv)
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Loading, EDA, SMOTE, scaling
│   ├── model_ann.py             # ANN architecture & training
│   ├── model_cnn.py             # 1D-CNN architecture & training
│   ├── model_lstm.py            # LSTM architecture & training
│   ├── model_autoencoder.py     # Autoencoder anomaly detection
│   ├── evaluation.py            # Metrics, plots, comparison
│   └── utils.py                 # Seed, GPU check
├── results/                     # Output plots & metrics
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 📊 Output

The system generates the following in the `results/` folder:
- EDA visualizations (class distribution, correlation heatmap)
- Training history plots (loss & accuracy per model)
- Confusion matrices for each model
- ROC curve comparison across all models
- Autoencoder reconstruction error distribution
- Printed comparison table with Accuracy, Precision, Recall, F1, AUC-ROC

---

## 🔧 Technologies

Python 3.x | TensorFlow/Keras | Pandas | NumPy | Scikit-learn | imbalanced-learn | Matplotlib | Seaborn

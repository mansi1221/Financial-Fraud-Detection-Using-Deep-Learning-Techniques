"""
=============================================================
Financial Fraud Detection System Using Deep Learning
& Anomaly Detection
=============================================================
Main script — trains ANN, CNN, LSTM, and Autoencoder models,
evaluates them, and compares results.

Usage:
    python main.py
=============================================================
"""

import os
import sys
import warnings
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import set_seed, check_gpu
from src.data_preprocessing import load_data, explore_data, preprocess_data, prepare_sequential_data
from src.model_ann import build_ann, train_ann
from src.model_cnn import build_cnn, train_cnn
from src.model_lstm import build_lstm, train_lstm
from src.model_autoencoder import (build_autoencoder, train_autoencoder,
                                    find_optimal_threshold, evaluate_autoencoder)
from src.evaluation import (evaluate_model, plot_training_history,
                             plot_roc_comparison, print_comparison_table)


# =================== CONFIGURATION ===================
DATA_PATH = os.path.join("data", "creditcard.csv")
RESULTS_DIR = "results"
EPOCHS = 30  # Reduced for faster demonstration
BATCH_SIZE = 256
RANDOM_STATE = 42
# ======================================================


def main():
    print("=" * 60)
    print("  FINANCIAL FRAUD DETECTION SYSTEM")
    print("  Deep Learning & Anomaly Detection")
    print("=" * 60)

    # --- Setup ---
    set_seed(RANDOM_STATE)
    check_gpu()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Step 1: Load & Explore Data ---
    print("\n" + "=" * 60)
    print("  STEP 1: DATA LOADING & EXPLORATION")
    print("=" * 60)
    df = load_data(DATA_PATH)
    explore_data(df, save_dir=RESULTS_DIR)

    # --- Step 2: Preprocess Data ---
    print("\n" + "=" * 60)
    print("  STEP 2: DATA PREPROCESSING")
    print("=" * 60)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, apply_smote=True, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Save scaler for Gradio app
    joblib.dump(scaler, 'scaler.pkl')
    print("[INFO] Scaler saved as 'scaler.pkl'")

    # Create validation split from training data
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"[INFO] Final train: {X_train_final.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    # Prepare sequential data for CNN & LSTM
    X_train_seq, X_test_seq = prepare_sequential_data(X_train_final, X_test)
    X_val_seq = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    # Store all metrics for comparison
    all_metrics = {}

    # --- Step 3: Train ANN ---
    print("\n" + "=" * 60)
    print("  STEP 3: TRAINING ANN MODEL")
    print("=" * 60)
    ann_model = build_ann(input_dim=X_train_final.shape[1])
    ann_history = train_ann(ann_model, X_train_final, y_train_final, X_val, y_val,
                            epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_history(ann_history, model_name="ANN", save_dir=RESULTS_DIR)
    all_metrics["ANN"] = evaluate_model(ann_model, X_test, y_test, model_name="ANN", save_dir=RESULTS_DIR)

    # --- Step 4: Train CNN ---
    print("\n" + "=" * 60)
    print("  STEP 4: TRAINING CNN MODEL")
    print("=" * 60)
    cnn_model = build_cnn(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    cnn_history = train_cnn(cnn_model, X_train_seq, y_train_final, X_val_seq, y_val,
                            epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_history(cnn_history, model_name="CNN", save_dir=RESULTS_DIR)
    all_metrics["CNN"] = evaluate_model(cnn_model, X_test_seq, y_test, model_name="CNN", save_dir=RESULTS_DIR)

    # --- Step 5: Train LSTM ---
    print("\n" + "=" * 60)
    print("  STEP 5: TRAINING LSTM MODEL")
    print("=" * 60)
    lstm_model = build_lstm(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_history = train_lstm(lstm_model, X_train_seq, y_train_final, X_val_seq, y_val,
                              epochs=EPOCHS, batch_size=BATCH_SIZE)
    plot_training_history(lstm_history, model_name="LSTM", save_dir=RESULTS_DIR)
    all_metrics["LSTM"] = evaluate_model(lstm_model, X_test_seq, y_test, model_name="LSTM", save_dir=RESULTS_DIR)

    # --- Step 6: Train Autoencoder (Anomaly Detection) ---
    print("\n" + "=" * 60)
    print("  STEP 6: TRAINING AUTOENCODER (ANOMALY DETECTION)")
    print("=" * 60)
    # Train autoencoder on NORMAL transactions only
    X_train_normal = X_train_final[y_train_final == 0]
    X_val_normal = X_val[y_val == 0]
    print(f"[INFO] Normal training samples: {X_train_normal.shape[0]}")
    print(f"[INFO] Normal validation samples: {X_val_normal.shape[0]}")

    ae_model = build_autoencoder(input_dim=X_train_final.shape[1])
    ae_history = train_autoencoder(ae_model, X_train_normal, X_val_normal,
                                   epochs=100, batch_size=BATCH_SIZE)
    
    # Save autoencoder model for Gradio app
    ae_model.save('fraud_autoencoder_model.h5')
    print("[INFO] Autoencoder model saved as 'fraud_autoencoder_model.h5'")
    
    plot_training_history(ae_history, model_name="Autoencoder", save_dir=RESULTS_DIR)

    # Find optimal threshold on validation set
    threshold = find_optimal_threshold(ae_model, X_val, y_val)

    # Evaluate autoencoder
    ae_results = evaluate_autoencoder(ae_model, X_test, y_test, threshold, save_dir=RESULTS_DIR)
    all_metrics["Autoencoder"] = ae_results

    # --- Step 7: Compare All Models ---
    print("\n" + "=" * 60)
    print("  STEP 7: MODEL COMPARISON")
    print("=" * 60)
    plot_roc_comparison(y_test, all_metrics, save_dir=RESULTS_DIR)
    print_comparison_table(all_metrics)

    print("\n" + "=" * 60)
    print("  ALL DONE! Results saved to 'results/' folder.")
    print("=" * 60)

    # --- Step 8: Launch Web Dashboard ---
    print("\n" + "=" * 60)
    print("  STEP 8: LAUNCHING WEB DASHBOARD")
    print("=" * 60)
    print("[INFO] Starting local web server on port 8765...")
    
    import threading
    import http.server
    import socketserver
    import webbrowser
    import time

    PORT = 8765
    DIRECTORY = "web"

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)
            
    # Function to run the server
    def start_server():
        try:
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"[INFO] Dashboard running at http://localhost:{PORT}/index.html")
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 10048:
                print(f"[INFO] Server is already running on port {PORT}. Opening browser...")
            else:
                print(f"[ERROR] Failed to start server: {e}")

    # Start the server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Give the server a second to start, then open the browser
    time.sleep(1)
    url = f"http://localhost:{PORT}/index.html"
    print(f"[INFO] Opening {url} in your default web browser...")
    webbrowser.open(url)
    
    print("\n[INFO] The web server is running in the background.")
    print("[INFO] Press Ctrl+C to stop the script and shut down the server.")
    
    # Keep the main thread alive so the server doesn't die immediately
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")


if __name__ == "__main__":
    main()

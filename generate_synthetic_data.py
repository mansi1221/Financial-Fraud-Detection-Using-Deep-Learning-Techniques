"""
Synthetic Dataset Generator
============================
Generates a realistic synthetic credit card fraud dataset that mimics
the Kaggle Credit Card Fraud Detection dataset structure.

Features: Time, V1-V28 (PCA-transformed), Amount, Class
- Class 0 = Legitimate, Class 1 = Fraudulent
- Fraud ratio ~0.17% (matching real dataset)
"""

import numpy as np
import pandas as pd
import os


def generate_synthetic_creditcard_data(
    n_samples=50000,
    fraud_ratio=0.0017,
    random_state=42
):
    """
    Generate synthetic data mimicking the creditcard.csv format.
    
    Real dataset has 284,807 samples. We use 50,000 for faster training
    while maintaining the same structure and class imbalance.
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    print(f"Generating {n_samples} transactions...")
    print(f"  Legitimate: {n_legit}")
    print(f"  Fraudulent: {n_fraud}")
    print(f"  Fraud ratio: {fraud_ratio * 100:.2f}%")
    
    # --- Generate legitimate transactions ---
    # V1-V28 are PCA components, so they follow roughly normal distributions
    legit_features = np.random.randn(n_legit, 28)
    # Scale some features to have different variances (like real PCA components)
    for i in range(28):
        scale = max(0.5, 5.0 - i * 0.15)  # Earlier components have larger variance
        legit_features[:, i] *= scale
    
    legit_time = np.sort(np.random.uniform(0, 172800, n_legit))  # 2 days in seconds
    legit_amount = np.abs(np.random.exponential(scale=88.0, size=n_legit))
    legit_amount = np.clip(legit_amount, 0, 25691)  # Max amount
    
    # --- Generate fraudulent transactions ---
    # Fraud transactions have different distributions (shifted means, different spreads)
    fraud_features = np.random.randn(n_fraud, 28)
    
    # Key discriminative features (V1, V2, V3, V4, V10, V12, V14, V17 differ most)
    fraud_shifts = {
        0: -3.0,   # V1 tends to be more negative for fraud
        1: 2.5,    # V2 tends to be more positive
        2: -4.0,   # V3 strongly negative for fraud
        3: 3.0,    # V4 positive for fraud
        9: -4.5,   # V10 strongly negative
        11: -3.5,  # V12 negative
        13: -5.0,  # V14 strongly negative (most discriminative)
        16: -4.0,  # V17 negative
    }
    
    for i in range(28):
        scale = max(0.5, 5.0 - i * 0.15)
        fraud_features[:, i] *= scale
        if i in fraud_shifts:
            fraud_features[:, i] += fraud_shifts[i]
    
    fraud_time = np.sort(np.random.uniform(0, 172800, n_fraud))
    fraud_amount = np.abs(np.random.exponential(scale=122.0, size=n_fraud))
    fraud_amount = np.clip(fraud_amount, 0, 2125)
    
    # --- Combine into DataFrame ---
    v_columns = [f"V{i}" for i in range(1, 29)]
    
    legit_df = pd.DataFrame(legit_features, columns=v_columns)
    legit_df.insert(0, "Time", legit_time)
    legit_df["Amount"] = legit_amount
    legit_df["Class"] = 0
    
    fraud_df = pd.DataFrame(fraud_features, columns=v_columns)
    fraud_df.insert(0, "Time", fraud_time)
    fraud_df["Amount"] = fraud_amount
    fraud_df["Class"] = 1
    
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    df = generate_synthetic_creditcard_data(n_samples=50000, fraud_ratio=0.005)
    
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "creditcard.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to '{output_path}'")
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["Class"].value_counts())
    print(f"\nFraud ratio: {df['Class'].mean() * 100:.2f}%")

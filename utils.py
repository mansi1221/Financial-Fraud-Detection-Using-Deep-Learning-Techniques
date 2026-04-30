"""
Utility Functions
==================
Helper functions used across the project.
"""

import os
import numpy as np
import tensorflow as tf
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[INFO] Random seed set to {seed}")


def check_gpu():
    """Check and print available GPU devices."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("[INFO] No GPU found. Running on CPU.")
    return len(gpus) > 0

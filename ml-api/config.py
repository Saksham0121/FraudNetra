"""
Configuration file for Fraud Detection ML Pipeline
"""

TRAIN_PATH = "../dataset/fraudTrain.csv"
TEST_PATH = "../dataset/fraudTest.csv"

MODEL_PATH = "fraud_autoencoder.h5"

SCALER_PATH = "scaler.pkl"

# anomaly detection threshold
ANOMALY_THRESHOLD = 1.2
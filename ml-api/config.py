"""
Configuration file for Fraud Detection ML Pipeline
"""

import os

TRAIN_PATH = "../dataset/fraudTrain.csv"
TEST_PATH = "../dataset/fraudTest.csv"

MODEL_PATH = "fraud_autoencoder.h5"

SCALER_PATH = "scaler.pkl"

# anomaly detection threshold
ANOMALY_THRESHOLD = 1.2

THRESHOLD_PATH = "threshold.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "fraudnetra")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "transaction_logs")

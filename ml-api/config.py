"""
Configuration file for Fraud Detection ML Pipeline
All dataset paths and output paths defined here
"""

# Raw dataset paths

TRAIN_PATH = "../dataset/fraudTrain.csv"
TEST_PATH = "../dataset/fraudTest.csv"


# Processed dataset paths
PROCESSED_TRAIN_PATH = "processed_train.csv"
PROCESSED_TEST_PATH = "processed_test.csv"


# Encoded dataset paths
ENCODED_TRAIN_PATH = "train_encoded.csv"
ENCODED_TEST_PATH = "test_encoded.csv"


# Scaled dataset paths
SCALED_TRAIN_PATH = "train_scaled.csv"
SCALED_TEST_PATH = "test_scaled.csv"


# Model artifacts
SCALER_PATH = "scaler.pkl"
MODEL_PATH = "autoencoder.h5"
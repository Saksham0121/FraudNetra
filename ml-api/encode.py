"""
Encoding pipeline for Fraud Detection dataset

This script performs:

- Load processed datasets
- One-hot encode categorical features
- Ensure train/test consistency
- Save encoded datasets
"""

import pandas as pd

from config import (
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH,
    ENCODED_TRAIN_PATH,
    ENCODED_TEST_PATH
)


def load_processed_data():
    print("Loading processed datasets")
    train = pd.read_csv(PROCESSED_TRAIN_PATH)
    test = pd.read_csv(PROCESSED_TEST_PATH)
    return train, test

def encode_categorical_features(train, test):
    print("Encoding categorical features using one-hot encoding")
    combined = pd.concat([train, test], axis=0)
    combined_encoded = pd.get_dummies(combined)
    train_encoded = combined_encoded.iloc[:len(train)]
    test_encoded = combined_encoded.iloc[len(train):]
    return train_encoded, test_encoded

def save_encoded_data(train_encoded, test_encoded):
    print("Saving encoded datasets")
    train_encoded.to_csv(ENCODED_TRAIN_PATH, index=False)
    test_encoded.to_csv(ENCODED_TEST_PATH, index=False)


if __name__ == "__main__":
    train, test = load_processed_data()
    train_encoded, test_encoded = encode_categorical_features(train, test)
    save_encoded_data(train_encoded, test_encoded)
    print("Encoding completed successfully")
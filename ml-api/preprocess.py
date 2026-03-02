"""
Fraud Detection Preprocessing Pipeline

This module handles:

- Loading dataset
- Cleaning dataset
- Feature engineering

"""

import pandas as pd
import numpy as np
from config import TRAIN_PATH, TEST_PATH


# Columns to remove
DROP_COLUMNS = [
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num"
]

# Load dataset

def load_dataset(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df


# Clean dataset
def clean_dataset(df):
    print("Cleaning dataset")
    df = df.drop(columns=DROP_COLUMNS)
    return df


# Feature engineering

def feature_engineering(df):
    print("Creating new features")

    # Convert time column
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    # Extract time features

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month


    # Distance feature

    df["distance"] = np.sqrt(

        (df["lat"] - df["merch_lat"])**2 +

        (df["long"] - df["merch_long"])**2

    )


    return df


# Test pipeline
if __name__ == "__main__":
    train_df = load_dataset(TRAIN_PATH)
    train_df = clean_dataset(train_df)
    train_df = feature_engineering(train_df)

    print("Final shape:", train_df.shape)

    print(train_df.head())
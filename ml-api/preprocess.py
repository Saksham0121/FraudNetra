"""
Fraud Detection Prepocessing Pipeline

This script performs:
- Data loading
- Data cleaning
- Feature engineering
- Saving processed datasets
"""
import pandas as pd
import numpy as np

from config import (
    TRAIN_PATH,
    TEST_PATH,
    PROCESSED_TRAIN_PATH,
    PROCESSED_TEST_PATH
)


# Columns to drop

DROP_COLUMNS = [
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num"
]


# Load dataset

def load_dataset(path):

    print(f"\nLoading dataset : {path}")
    df = pd.read_csv(path)
    print("Shape :", df.shape)
    return df

# Clean datast

def clean_dataset(df):
    print("Cleaning dataset")

    df = df.drop(columns=DROP_COLUMNS)
    return df


# Feature engineering
def feature_engineering(df):

    print("Performing feature engineering")
    # Convert transaction time

    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"]
    )

    # Create time features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month

    # Create distance feature
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"])**2 +
        (df["long"] - df["merch_long"])**2
    )

    # Drop original time colum

    df = df.drop(columns=["trans_date_trans_time"])
    return df


# Save dataset
def save_dataset(df, path):
    print(f"Saving dataset: {path}")
    df.to_csv(path, index=False)


# Full pipeline
def process_dataset(input_path, output_path):
    df = load_dataset(input_path)
    df = clean_dataset(df)
    df = feature_engineering(df)
    save_dataset(df, output_path)


# Run pipeline

if __name__ == "__main__":
    print("\nProcessing TRAIN dataset")

    process_dataset(

        TRAIN_PATH,
        PROCESSED_TRAIN_PATH
    )

    print("\nProcessing TEST dataset")
    process_dataset(
        TEST_PATH,
        PROCESSED_TEST_PATH
    )
    print("\nPreprocessing completed successfully")
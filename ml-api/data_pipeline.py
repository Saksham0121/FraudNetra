import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from config import SCALER_PATH


# columns to remove (high cardinality or useless)
DROP_COLUMNS = [
    "Unnamed: 0",
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num",
    "merchant",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "unix_time"
]


def load_data(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return df


def feature_engineering(df):
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour"] = df["trans_date_trans_time"].dt.hour
        df["day"] = df["trans_date_trans_time"].dt.day
        df["month"] = df["trans_date_trans_time"].dt.month
        df = df.drop(columns=["trans_date_trans_time"])
    else:
        df["hour"] = 0
        df["day"] = 0
        df["month"] = 0

    # distance feature
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )

    return df


def encode_data(df):
    print("Encoding categorical features...")
    df = pd.get_dummies(df)
    return df


def prepare_dataset(path):
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    feature_columns = X.columns.tolist()
    return X, y, feature_columns


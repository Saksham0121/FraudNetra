import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# columns to remove
DROP_COLUMNS = [
    "first",
    "last",
    "street",
    "trans_num",
    "cc_num"
]


def load_data(path):
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop(columns=DROP_COLUMNS)
    return df


def feature_engineering(df):

    # convert datetime
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # extract time features
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month

    # distance feature
    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 +
        (df["long"] - df["merch_long"]) ** 2
    )
    df = df.drop(columns=["trans_date_trans_time"])
    return df

def encode_data(df):
    print("Encoding categorical features...")
    df = pd.get_dummies(df)
    return df

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def prepare_dataset(path):
    df = load_data(path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    X_scaled, scaler = scale_data(X)
    return X_scaled, y, scaler
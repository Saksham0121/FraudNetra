import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from config import MODEL_PATH, SCALER_PATH, THRESHOLD_PATH


print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

print("Loading scaler...")
scaler = joblib.load(SCALER_PATH)

print("Loading threshold...")
threshold = joblib.load(THRESHOLD_PATH)


def compute_anomaly_score(X):
    """
    Computes reconstruction error using the autoencoder
    """

    reconstructed = model.predict(X)

    error = np.mean((X - reconstructed) ** 2, axis=1)

    return error


def predict_transaction(df):
    """
    Predict fraud for a dataframe of transactions
    """

    X = scaler.transform(df)

    error = compute_anomaly_score(X)

    fraud_prediction = error > threshold

    return error, fraud_prediction


if __name__ == "__main__":

    print("Running test prediction...")

    # get model input dimension automatically
    input_dim = model.input_shape[1]

    # generate dummy transaction
    sample = np.random.rand(1, input_dim)

    error = compute_anomaly_score(sample)

    print("Anomaly Score:", error)

    if error > threshold:
        print("Fraud detected")
    else:
        print("Transaction appears normal")
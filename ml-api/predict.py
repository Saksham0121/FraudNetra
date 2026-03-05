import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from config import MODEL_PATH, SCALER_PATH, ANOMALY_THRESHOLD


print("Loading model...")

model = load_model(MODEL_PATH, compile=False)

print("Loading scaler...")

scaler = joblib.load(SCALER_PATH)


def compute_anomaly_score(X):

    reconstructed = model.predict(X)

    error = np.mean((X - reconstructed) ** 2, axis=1)

    return error


def predict_transaction(df):

    X = scaler.transform(df)

    error = compute_anomaly_score(X)

    fraud_prediction = error > ANOMALY_THRESHOLD

    return error, fraud_prediction


if __name__ == "__main__":

    print("Running test prediction...")

    input_dim = model.input_shape[1]

    sample = np.random.rand(1, input_dim)

    error = compute_anomaly_score(sample)

    print("Anomaly Score:", error)

    if error > ANOMALY_THRESHOLD:
        print("Fraud detected")
    else:
        print("Transaction appears normal")
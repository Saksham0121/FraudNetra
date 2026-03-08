from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from config import MODEL_PATH, SCALER_PATH, ANOMALY_THRESHOLD


app = FastAPI()

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)


class Transaction(BaseModel):
    transaction: list[float]


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}


@app.post("/predict")
def predict(data: Transaction):

    input_dim = model.input_shape[1]

    X = np.zeros((1, input_dim))

    for i, value in enumerate(data.transaction):
        if i < input_dim:
            X[0][i] = value

    X = scaler.transform(X)

    reconstructed = model.predict(X)

    error = np.mean((X - reconstructed) ** 2)

    fraud = error > ANOMALY_THRESHOLD

    return {
        "anomaly_score": float(error),
        "is_fraud": bool(fraud)
    }
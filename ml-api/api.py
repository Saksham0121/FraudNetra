from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from config import MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, FEATURE_COLUMNS_PATH
from data_pipeline import clean_data, feature_engineering, encode_data


app = FastAPI()

fraud_stats = {
    "total_transactions": 0,
    "fraud_detected": 0
}

# load model artifacts
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)


class Transaction(BaseModel):
    amt: float
    category: str
    gender: str
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float


@app.get("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.post("/predict")
def predict(transaction: Transaction):

    df = pd.DataFrame([transaction.dict()])

    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    df = df.reindex(columns=feature_columns, fill_value=0)

    X = df.values
    X = scaler.transform(X)

    reconstructed = model.predict(X)

    error = np.mean((X - reconstructed) ** 2)

    fraud = error > threshold

    # update monitoring stats
    fraud_stats["total_transactions"] += 1

    if fraud:
        fraud_stats["fraud_detected"] += 1

    return {
        "anomaly_score": float(error),
        "is_fraud": bool(fraud)
    }

@app.get("/fraud-stats")
def fraud_stats_endpoint():

    total = fraud_stats["total_transactions"]
    fraud = fraud_stats["fraud_detected"]

    fraud_rate = 0

    if total > 0:
        fraud_rate = (fraud / total) * 100

    return {
        "total_transactions": total,
        "fraud_detected": fraud,
        "fraud_rate_percent": fraud_rate
    }

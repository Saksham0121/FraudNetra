from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from pymongo import MongoClient

from tensorflow.keras.models import load_model

from config import (
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    FEATURE_COLUMNS_PATH,
    MONGO_URI,
    MONGO_DB_NAME,
    MONGO_COLLECTION_NAME,
)
from data_pipeline import clean_data, feature_engineering, encode_data


app = FastAPI()

fraud_stats = {
    "total_transactions": 0,
    "fraud_detected": 0
}

transaction_logs = []

# load model artifacts
model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)
threshold = joblib.load(THRESHOLD_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

mongo_client = None
mongo_collection = None

try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    mongo_client.admin.command("ping")
    mongo_collection = mongo_client[MONGO_DB_NAME][MONGO_COLLECTION_NAME]
    print(f"Connected to MongoDB: {MONGO_DB_NAME}.{MONGO_COLLECTION_NAME}")
except Exception as exc:
    print(f"MongoDB connection unavailable, using in-memory logs only: {exc}")
    mongo_client = None
    mongo_collection = None


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
    return {
        "message": "Fraud Detection API running",
        "mongodb_connected": mongo_collection is not None
    }


def log_transaction(record):
    transaction_logs.append(record)

    if mongo_collection is not None:
        mongo_collection.insert_one(record)


def get_recent_frauds(limit):
    if mongo_collection is not None:
        recent_frauds = list(
            mongo_collection.find(
                {"is_fraud": True},
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit)
        )
        return recent_frauds

    recent_frauds = [record for record in transaction_logs if record["is_fraud"]]
    return recent_frauds[-limit:][::-1]

@app.post("/predict")
def predict(transaction: Transaction):
    transaction_data = transaction.dict()
    df = pd.DataFrame([transaction_data])

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

    log_transaction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction": transaction_data,
            "anomaly_score": float(error),
            "is_fraud": bool(fraud)
        }
    )

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


@app.get("/recent-frauds")
def recent_frauds_endpoint(limit: int = 10):
    recent_frauds = get_recent_frauds(limit)
    return {
        "count": len(recent_frauds),
        "recent_frauds": recent_frauds
    }

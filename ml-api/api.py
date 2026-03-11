from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model

from config import MODEL_PATH, SCALER_PATH, ANOMALY_THRESHOLD
from data_pipeline import clean_data, feature_engineering, encode_data

app = FastAPI()

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)


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

    # convert input to dataframe
    df = pd.DataFrame([transaction.dict()])

    # run preprocessing pipeline
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_data(df)

    X = df.values

    X = scaler.transform(X)

    reconstructed = model.predict(X)

    error = np.mean((X - reconstructed) ** 2)

    fraud = error > ANOMALY_THRESHOLD

    return {
        "anomaly_score": float(error),
        "is_fraud": bool(fraud)
    }
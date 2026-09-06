"""
src/fastapi_app.py

FastAPI microservice for fraud detection.

Endpoints:
- GET /health
- POST /predict
- POST /predict/batch
- POST /train
"""

from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model import load_model, save_model, TrainedFraudModel, MODEL_PATH
from src.pipeline import train_best_model


class Transaction(BaseModel):
    Time: float = Field(..., description="Time (seconds) since first transaction in dataset")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount")


class TransactionBatch(BaseModel):
    transactions: List[Transaction]


class Prediction(BaseModel):
    is_fraud: int = Field(..., description="1 = fraud, 0 = legitimate")
    fraud_probability: float = Field(..., ge=0, le=1)
    threshold_used: float


class BatchPrediction(BaseModel):
    predictions: List[Prediction]


class TrainResponse(BaseModel):
    message: str
    metrics: dict
    threshold: float


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description="CSH-XGB based fraud detection service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_model: TrainedFraudModel | None = None


def get_model() -> TrainedFraudModel:
    global _model
    if _model is None:
        _model = load_model(MODEL_PATH)
    return _model


@app.on_event("startup")
def startup_event():
    try:
        m = get_model()
        print(f"[startup] Loaded model with threshold={m.best_threshold:.4f}")
    except FileNotFoundError as ex:
        print(f"[startup] {ex}")


@app.get("/health", tags=["system"])
def health_check():
    try:
        _ = get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as exc:
        return {"status": "degraded", "model_loaded": False, "detail": str(exc)}


def _txn_to_row(txn: Transaction) -> dict:
    return {
        "Time": txn.Time,
        "V1": txn.V1,
        "V2": txn.V2,
        "V3": txn.V3,
        "V4": txn.V4,
        "V5": txn.V5,
        "V6": txn.V6,
        "V7": txn.V7,
        "V8": txn.V8,
        "V9": txn.V9,
        "V10": txn.V10,
        "V11": txn.V11,
        "V12": txn.V12,
        "V13": txn.V13,
        "V14": txn.V14,
        "V15": txn.V15,
        "V16": txn.V16,
        "V17": txn.V17,
        "V18": txn.V18,
        "V19": txn.V19,
        "V20": txn.V20,
        "V21": txn.V21,
        "V22": txn.V22,
        "V23": txn.V23,
        "V24": txn.V24,
        "V25": txn.V25,
        "V26": txn.V26,
        "V27": txn.V27,
        "V28": txn.V28,
        "Amount": txn.Amount,
    }


@app.post("/predict", response_model=Prediction, tags=["inference"])
def predict_one(transaction: Transaction):
    try:
        model = get_model()
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex))

    X = pd.DataFrame([_txn_to_row(transaction)])
    proba = model.pipeline.predict_proba(X)[:, 1][0]
    label = int(proba >= model.best_threshold)

    return Prediction(
        is_fraud=label,
        fraud_probability=float(proba),
        threshold_used=float(model.best_threshold),
    )


@app.post("/predict/batch", response_model=BatchPrediction, tags=["inference"])
def predict_batch(batch: TransactionBatch):
    if not batch.transactions:
        raise HTTPException(status_code=400, detail="transactions list cannot be empty")
    try:
        model = get_model()
    except FileNotFoundError as ex:
        raise HTTPException(status_code=503, detail=str(ex))

    rows = [_txn_to_row(t) for t in batch.transactions]
    X = pd.DataFrame(rows)
    probas = model.pipeline.predict_proba(X)[:, 1]
    labels = (probas >= model.best_threshold).astype(int)

    preds = [
        Prediction(
            is_fraud=int(lab),
            fraud_probability=float(prob),
            threshold_used=float(model.best_threshold),
        )
        for lab, prob in zip(labels, probas)
    ]
    return BatchPrediction(predictions=preds)


@app.post("/train", response_model=TrainResponse, tags=["training"])
def train_model_endpoint():
    """
    Trigger model retraining via API.
    """
    global _model
    try:
        trained = train_best_model()
        save_model(trained, MODEL_PATH)
        _model = trained
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TrainResponse(
        message="Model trained and reloaded successfully.",
        metrics=trained.metrics,
        threshold=trained.best_threshold,
    )

"""
src/model.py

Model wrapper class and utility to load/save the trained fraud model.
"""

import os
from dataclasses import dataclass
from typing import Dict

import joblib
from imblearn.pipeline import Pipeline as ImbPipeline


@dataclass
class TrainedFraudModel:
    """
    Mirror of the dataclass in pipeline.py, duplicated here so that
    loading via joblib from different modules is safe.
    """
    pipeline: ImbPipeline
    best_threshold: float
    metrics: Dict[str, float]


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.joblib")


def save_model(model: TrainedFraudModel, path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str = MODEL_PATH) -> TrainedFraudModel:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first."
        )
    model = joblib.load(path)
    return model

"""
src/pipeline.py

Core ML pipeline:
- Load dataset from csv_files/creditcard.csv
- Build preprocessing
- Apply hybrid resampling (under + SMOTE)
- Train candidate models (LR, RF, XGB)
- Select best by F2-score
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42
DATA_PATH = os.path.join("csv_files", "creditcard.csv")


@dataclass
class TrainedFraudModel:
    """Bundle of pipeline + best threshold + simple metrics."""
    pipeline: ImbPipeline
    best_threshold: float
    metrics: Dict[str, float]


def load_data(path: str = DATA_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the credit card fraud dataset.
    Expects a `creditcard.csv` file with a `Class` column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Please download creditcard.csv from Kaggle "
            "and place it in the csv_files/ directory."
        )
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the dataset.")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing pipeline that scales all numeric features.
    """
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)]
    )
    return preprocessor


def hybrid_resampler_steps():
    """
    Return ordered steps for imblearn pipeline:
    - Random undersampling
    - SMOTE oversampling
    """
    under = RandomUnderSampler(
        sampling_strategy=0.1,
        random_state=RANDOM_STATE,
    )
    smote = SMOTE(
        sampling_strategy=1.0,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    return [("under", under), ("smote", smote)]


def compute_class_weights(y: pd.Series) -> Dict[int, float]:
    """
    Compute balanced class weights for the binary labels.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y,
    )
    return {int(cls): float(w) for cls, w in zip(classes, weights)}


def _build_xgb_classifier(class_weights: Dict[int, float]) -> XGBClassifier:
    """
    Build an XGBoost classifier for imbalanced fraud detection.
    """
    scale_pos_weight = class_weights[1] / class_weights[0]
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_best_model(test_size: float = 0.2) -> TrainedFraudModel:
    """
    Train several models and return the best one (by F2-score)
    wrapped in a TrainedFraudModel object.
    """
    X, y = load_data()
    preprocessor = build_preprocessor(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    class_weights = compute_class_weights(y_train)

    # Candidate models
    lr = LogisticRegression(
        max_iter=1000,
        class_weight=class_weights,
        solver="lbfgs",
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight=class_weights,
        random_state=RANDOM_STATE,
    )

    xgb = _build_xgb_classifier(class_weights)

    candidates = {
        "LogisticRegression": lr,
        "RandomForest": rf,
        "XGBoost": xgb,
    }

    best_pipeline = None
    best_threshold = 0.5
    best_metrics = {"f2": -1.0}
    best_name = None

    for name, estimator in candidates.items():
        steps = [("pre", preprocessor)]
        steps.extend(hybrid_resampler_steps())
        steps.append(("clf", estimator))

        pipe = ImbPipeline(steps=steps)
        pipe.fit(X_train, y_train)

        y_scores = pipe.predict_proba(X_valid)[:, 1]

        precisions, recalls, thresholds = precision_recall_curve(y_valid, y_scores)

        # F2-score emphasises recall
        f2_scores = []
        for p, r in zip(precisions, recalls):
            if (p + 4 * r) == 0:
                f2_scores.append(0.0)
            else:
                f2 = (5 * p * r) / (4 * p + r)
                f2_scores.append(f2)

        f2_scores = np.array(f2_scores)
        best_idx = int(np.argmax(f2_scores))

        local_best_threshold = 0.5
        if best_idx < len(thresholds):
            local_best_threshold = float(thresholds[best_idx])

        y_pred = (y_scores >= local_best_threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_valid, y_scores)),
            "avg_precision": float(average_precision_score(y_valid, y_scores)),
            "f1": float(f1_score(y_valid, y_pred)),
            "f2": float(f2_scores[best_idx]),
        }

        if metrics["f2"] > best_metrics.get("f2", -1.0):
            best_metrics = metrics
            best_threshold = local_best_threshold
            best_pipeline = pipe
            best_name = name

    print(f"[train_best_model] Best model: {best_name} with F2={best_metrics['f2']:.4f}")

    return TrainedFraudModel(
        pipeline=best_pipeline,
        best_threshold=float(best_threshold),
        metrics=best_metrics,
    )

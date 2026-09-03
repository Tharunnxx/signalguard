"""
risk_model.py
-------------
The underlying AI/ML payment risk model that SignalGuard evaluates.

IMPORTANT: This model is intentionally simple (Logistic Regression). It is
NOT the product - SignalGuard is. This model's only job is to produce a
realistic "APPROVE / REVIEW / DECLINE" recommendation + confidence, so that
SignalGuard has something real to build trust checks on top of.

Trained ONLY on numeric risk/behavior features. Never trained on `scenario`
or `actual_risk` - those are ground-truth columns reserved for evaluating
SignalGuard later, not real-world model inputs.

Run:
    python risk_model.py

Outputs:
    Prints train/test metrics (Precision, Recall, F1, ROC-AUC).
    Saves the trained model + imputer to risk_model.joblib in this folder.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import joblib

DATA_PATH = "../../data/transactions.csv"
MODEL_OUT_PATH = "risk_model.joblib"

# The 12 numeric features the model is allowed to see.
# Deliberately excludes 'scenario' and 'actual_risk' (ground truth / answer key).
FEATURE_COLUMNS = [
    "amount",
    "device_age_days",
    "account_age_days",
    "previous_txn_count",
    "txn_velocity_1h",
    "avg_amount_7d",
    "amount_deviation",
    "latency_ms",
    "retry_count",
    "ip_risk_score",
    "device_risk_score",
    "location_risk_score",
    "bank_risk_score",
    "amount_risk_score",
]

TARGET_COLUMN = "outcome"  # LEGIT / FRAUD


def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLUMNS].copy()
    y = (df[TARGET_COLUMN] == "FRAUD").astype(int)  # FRAUD=1, LEGIT=0
    return df, X, y


def build_and_train():
    print("Loading data...")
    df, X, y = load_data(DATA_PATH)

    print(f"Features used ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN} -> FRAUD=1, LEGIT=0")
    print(f"Class balance:\n{y.value_counts()}\n")

    # Impute missing values (from MISSING_SIGNAL scenario) with column median.
    # Fit imputer BEFORE the split is avoided on purpose - we split first,
    # then fit imputer/scaler on train only, to prevent data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Scale features - Logistic Regression coefficients are only meaningful
    # for comparison (needed later for counterfactual testing) if features
    # are on the same scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Evaluation on held-out test set ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix (rows=actual, cols=predicted) [LEGIT, FRAUD]:\n{cm}")

    # Show feature coefficients - this is what counterfactual.py will use
    # later to decide which features matter most for a given decision.
    coef_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "coefficient": model.coef_[0]
    }).sort_values("coefficient", key=abs, ascending=False)
    print("\n--- Feature coefficients (higher |value| = more influence) ---")
    print(coef_df.to_string(index=False))

    # Save model + preprocessing objects together, so risk_model.joblib is
    # a single self-contained artifact the FastAPI backend can load later.
    joblib.dump({
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
    }, MODEL_OUT_PATH)
    print(f"\nSaved trained model bundle to {MODEL_OUT_PATH}")

    return model, imputer, scaler


def predict_with_confidence(model, imputer, scaler, transaction_features: dict):
    """
    Given a single transaction's feature dict, returns:
        recommendation: 'APPROVE' | 'REVIEW' | 'DECLINE'
        confidence: float 0-1 (model's probability of FRAUD)

    Thresholds are simple and transparent on purpose - SignalGuard is what
    adds nuance on top of this, not this model.
    """
    row = pd.DataFrame([transaction_features])[FEATURE_COLUMNS]
    row_imputed = imputer.transform(row)
    row_scaled = scaler.transform(row_imputed)

    fraud_proba = model.predict_proba(row_scaled)[0, 1]

    if fraud_proba < 0.3:
        recommendation = "APPROVE"
    elif fraud_proba < 0.7:
        recommendation = "REVIEW"
    else:
        recommendation = "DECLINE"

    confidence = fraud_proba if recommendation == "DECLINE" else 1 - fraud_proba
    return {
        "recommendation": recommendation,
        "fraud_probability": round(float(fraud_proba), 4),
        "confidence": round(float(confidence), 4),
    }


if __name__ == "__main__":
    build_and_train()

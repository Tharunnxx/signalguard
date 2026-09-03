"""
model_confidence.py
--------------------
Model Confidence checker for SignalGuard.

Answers: "how sure is the underlying risk model actually, about this
specific decision?" This is NOT about whether the model is right - it's
about whether the model itself is close to a coin flip or genuinely
decisive. A model that outputs 0.51 fraud probability and rounds that to
"APPROVE" is not making a confident call, even if the recommendation looks
clean on the surface.

Approach: distance from the decision boundary (0.5), scaled to 0-100.
    - fraud_probability = 0.5  -> confidence = 0   (pure coin flip)
    - fraud_probability = 0.05 or 0.95 -> confidence ~= 90
    - fraud_probability = 0.0 or 1.0  -> confidence = 100 (maximally decisive)

Verified against the real trained model + real data: most everyday traffic
(NORMAL, BANK_DRIFT, SIGNAL_CONFLICT, MISSING_SIGNAL, LATENCY_ANOMALY) sits
in the 16-33 confidence range - the model, trained with class_weight=
"balanced" on a rare-fraud dataset, is naturally cautious and rarely near
0 or 1 except on the most clear-cut fraud patterns (HIGH_VELOCITY ~94,
COMBINED_INCIDENT ~81). This is an honest reflection of the real model,
not a target to "fix" - it's exactly the kind of thing SignalGuard exists
to surface: high-looking recommendations can still ride on a model that
isn't very sure of itself.

Run:
    python model_confidence.py
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../models/risk_model.joblib"
DATA_PATH = "../../data/transactions.csv"


def load_model_bundle(path: str = MODEL_PATH) -> dict:
    """Loads the self-contained model bundle (model + imputer + scaler + features)."""
    return joblib.load(path)


def get_fraud_probabilities(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Runs the trained model on a dataframe, returns fraud probability per row."""
    X = df[bundle["feature_columns"]]
    X_imputed = bundle["imputer"].transform(X)
    X_scaled = bundle["scaler"].transform(X_imputed)
    return bundle["model"].predict_proba(X_scaled)[:, 1]


def score_from_proba(fraud_proba) -> float | np.ndarray:
    """
    Converts a fraud probability (0-1) into a 0-100 confidence score based
    on distance from the 0.5 decision boundary.
    """
    return np.clip(np.abs(fraud_proba - 0.5) / 0.5 * 100, 0, 100)


def score_row(row: pd.Series, bundle: dict) -> tuple[float, list[str]]:
    """
    Scores a single transaction's model confidence.
    Returns (score, issues) - issues is non-empty when the model is close
    to a coin flip, for the dashboard's evidence trail.
    """
    fraud_proba = get_fraud_probabilities(pd.DataFrame([row]), bundle)[0]
    score = round(float(score_from_proba(fraud_proba)), 2)

    issues = []
    if score < 40:
        issues.append(
            f"Model Confidence: fraud probability is {fraud_proba:.2f} - "
            f"close to a coin flip (0.5), the model isn't very sure of this call"
        )
    return score, issues


def score_dataframe(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Scores every row in df. Adds 'fraud_proba', 'confidence_score',
    and 'confidence_issues' columns.
    """
    df = df.copy()
    fraud_proba = get_fraud_probabilities(df, bundle)
    df["fraud_proba"] = fraud_proba
    df["confidence_score"] = np.round(score_from_proba(fraud_proba), 2)
    df["confidence_issues"] = [
        [f"Model Confidence: fraud probability is {p:.2f} - close to a coin flip (0.5), "
         f"the model isn't very sure of this call"] if s < 40 else []
        for p, s in zip(df["fraud_proba"], df["confidence_score"])
    ]
    return df


if __name__ == "__main__":
    print("Loading model bundle...")
    bundle = load_model_bundle()

    print("Loading transactions...")
    df = pd.read_csv(DATA_PATH)

    print("Scoring model confidence for all transactions...")
    scored = score_dataframe(df, bundle)

    print("\nAverage Model Confidence Score by scenario:")
    print(scored.groupby("scenario")["confidence_score"].mean().round(2))

    print("\nSample issues found (first 3 LATENCY_ANOMALY rows - lowest confidence scenario):")
    low_conf_rows = scored[scored["scenario"] == "LATENCY_ANOMALY"].head(3)
    for _, r in low_conf_rows.iterrows():
        print(f"  {r['transaction_id']}: score={r['confidence_score']}, issues={r['confidence_issues']}")

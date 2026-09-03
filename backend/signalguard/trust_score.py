"""
trust_score.py
--------------
The Decision Trust Score combiner - takes the individual check scores
(Data Quality, Drift, Signal Consistency, Model Confidence, Decision
Robustness) and combines them into one 0-100 Trust Score with a
HIGH / MEDIUM / LOW label.

This file is a PURE COMBINER: it does not run any checker itself, does not
touch a dataframe, and does not know how any score was computed. It just
takes numbers in and produces a trust verdict out. This keeps it reusable
from anywhere - a batch evaluation script over the whole CSV, or the
FastAPI backend scoring one live transaction - without duplicating logic.

--- TEMPORARY WEIGHTING NOTICE ---
The full design uses 5 weighted components:
    Data Quality        20%
    Drift Health        20%
    Signal Consistency  20%
    Model Confidence    15%
    Decision Robustness 25%   (needs counterfactual.py - Day 2, not built yet)

Until Decision Robustness exists, this combiner re-normalizes the other
four weights so they still sum to 100% (each divided by 0.75, since they
summed to 75% originally):
    Data Quality        20% -> 26.7%
    Drift Health         20% -> 26.7%
    Signal Consistency  20% -> 26.7%
    Model Confidence     15% -> 20.0%

The moment decision_robustness is passed in, compute_trust_score
automatically switches back to the full 5-weight formula. No other code
needs to change.

Run:
    python trust_score.py
(runs a self-test combining all four real checkers built so far against
the real dataset)
"""

from typing import Optional

# ---- Full design weights (used once Decision Robustness exists) ----
FULL_WEIGHTS = {
    "data_quality": 0.20,
    "drift": 0.20,
    "signal_consistency": 0.20,
    "model_confidence": 0.15,
    "decision_robustness": 0.25,
}

# ---- Temporary weights (Decision Robustness not built yet) ----
# = FULL_WEIGHTS for the 4 available components, each divided by 0.75
_TEMP_BASE = 1.0 - FULL_WEIGHTS["decision_robustness"]  # 0.75
TEMP_WEIGHTS = {
    "data_quality": FULL_WEIGHTS["data_quality"] / _TEMP_BASE,
    "drift": FULL_WEIGHTS["drift"] / _TEMP_BASE,
    "signal_consistency": FULL_WEIGHTS["signal_consistency"] / _TEMP_BASE,
    "model_confidence": FULL_WEIGHTS["model_confidence"] / _TEMP_BASE,
}

HIGH_TRUST_THRESHOLD = 80
MEDIUM_TRUST_THRESHOLD = 50


def _classify(score: float) -> str:
    if score >= HIGH_TRUST_THRESHOLD:
        return "HIGH_TRUST"
    elif score >= MEDIUM_TRUST_THRESHOLD:
        return "MEDIUM_TRUST"
    else:
        return "LOW_TRUST"


def compute_trust_score(
    data_quality: float,
    drift: float,
    signal_consistency: float,
    model_confidence: float,
    decision_robustness: Optional[float] = None,
) -> dict:
    """
    Combines individual check scores (each 0-100) into a single Decision
    Trust Score.

    If decision_robustness is None, uses the TEMPORARY re-normalized
    4-component weighting. Once decision_robustness is provided (once
    counterfactual.py exists), automatically uses the full 5-component
    weighting instead.

    Returns:
        {
            "trust_score": float 0-100,
            "trust_level": "HIGH_TRUST" | "MEDIUM_TRUST" | "LOW_TRUST",
            "weights_used": "full" | "temporary_4_component",
            "components": {...the raw inputs, for the audit trail...},
        }
    """
    components = {
        "data_quality": data_quality,
        "drift": drift,
        "signal_consistency": signal_consistency,
        "model_confidence": model_confidence,
        "decision_robustness": decision_robustness,
    }

    if decision_robustness is None:
        score = (
            data_quality * TEMP_WEIGHTS["data_quality"]
            + drift * TEMP_WEIGHTS["drift"]
            + signal_consistency * TEMP_WEIGHTS["signal_consistency"]
            + model_confidence * TEMP_WEIGHTS["model_confidence"]
        )
        weights_used = "temporary_4_component"
    else:
        score = (
            data_quality * FULL_WEIGHTS["data_quality"]
            + drift * FULL_WEIGHTS["drift"]
            + signal_consistency * FULL_WEIGHTS["signal_consistency"]
            + model_confidence * FULL_WEIGHTS["model_confidence"]
            + decision_robustness * FULL_WEIGHTS["decision_robustness"]
        )
        weights_used = "full"

    score = round(max(0.0, min(100.0, score)), 2)

    return {
        "trust_score": score,
        "trust_level": _classify(score),
        "weights_used": weights_used,
        "components": components,
    }


if __name__ == "__main__":
    # Self-test: run all four real checkers built so far against the real
    # dataset, combine them, and see the actual Trust Score by scenario.
    import pandas as pd
    import joblib

    from data_quality import assess_data_quality_batch
    from drift import compute_baseline, score_dataframe as score_drift
    from signal_conflict import score_dataframe as score_conflict
    from model_confidence import load_model_bundle, score_dataframe as score_confidence
    from counterfactual import compute_feature_stds, score_dataframe as score_robustness

    print("Loading transactions...")
    df = pd.read_csv("../../data/transactions.csv")

    print("Running Data Quality checker...")
    df = assess_data_quality_batch(df)

    print("Running Drift checker...")
    baseline = compute_baseline(df)
    df = score_drift(df, baseline)

    print("Running Signal Conflict checker...")
    df = score_conflict(df)

    print("Running Model Confidence checker...")
    bundle = load_model_bundle()
    df = score_confidence(df, bundle)

    print("Running Decision Robustness (Counterfactual) checker...")
    feature_stds = compute_feature_stds(df)
    df = score_robustness(df, bundle, feature_stds)

    print("Combining into Trust Score (full 5-component weighting)...")
    results = [
        compute_trust_score(
            data_quality=row["data_quality_score"],
            drift=row["drift_score"],
            signal_consistency=row["conflict_score"],
            model_confidence=row["confidence_score"],
            decision_robustness=row["robustness_score"],
        )
        for _, row in df.iterrows()
    ]
    df["trust_score"] = [r["trust_score"] for r in results]
    df["trust_level"] = [r["trust_level"] for r in results]

    print("\nAverage Trust Score by scenario:")
    print(df.groupby("scenario")["trust_score"].mean().round(2).sort_values())

    print("\nTrust Level distribution by scenario:")
    print(pd.crosstab(df["scenario"], df["trust_level"]))

    print("\nOverall Trust Level distribution:")
    print(df["trust_level"].value_counts())

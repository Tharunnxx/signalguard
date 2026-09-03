"""
pipeline.py
-----------
The shared SignalGuard scoring pipeline - used by BOTH:
  1. Batch scoring (scoring the whole historical CSV for the dashboard)
  2. Live scoring (the FastAPI endpoint scoring one incoming transaction)

This exists so there is exactly ONE place that defines "how SignalGuard
scores a transaction" - the API and any batch/evaluation script both call
into this, instead of duplicating the same 5-checker + trust-score +
incident logic in two places that could quietly drift apart.

Two things must be computed from HISTORICAL data before live scoring can
happen at all:
    - the Drift baseline (NORMAL-row mean/std for bank_risk_score, latency_ms)
    - the Counterfactual nudge sizes (std-dev of 8 key features)
These are computed ONCE (e.g. at FastAPI startup) via load_resources(),
then reused for every request - this mirrors how a real fraud system
refreshes baselines periodically, not on every single transaction, and
keeps live scoring fast.

Run:
    python pipeline.py
(self-test: loads resources once, then scores a handful of individual
transactions through the SAME code path the live API will use, to prove
single-transaction scoring works correctly before FastAPI is built on it)
"""

import pandas as pd

from data_quality import assess_data_quality
from drift import compute_baseline as compute_drift_baseline, score_row as score_drift_row
from signal_conflict import score_row as score_conflict_row
from model_confidence import load_model_bundle, score_row as score_confidence_row
from counterfactual import compute_feature_stds, score_dataframe as score_robustness_batch
from trust_score import compute_trust_score
from incident_detection import get_action, build_incident

DATA_PATH = "../../data/transactions.csv"
MODEL_PATH = "../models/risk_model.joblib"


def load_resources(data_path: str = DATA_PATH, model_path: str = MODEL_PATH) -> dict:
    """
    Loads/computes everything needed to score transactions, ONCE. Call this
    at FastAPI startup and hang onto the result for the lifetime of the
    server (or until a periodic baseline refresh, in a real system).

    Returns a dict with:
        - model_bundle: the trained model + imputer + scaler + feature list
        - drift_baseline: NORMAL-row mean/std for drift-monitored features
        - feature_stds: std-dev of counterfactual-perturbed features
        - historical_df: the raw historical dataset (kept for reference /
          re-deriving baselines later if needed)
    """
    historical_df = pd.read_csv(data_path)
    model_bundle = load_model_bundle(model_path)
    drift_baseline = compute_drift_baseline(historical_df)
    feature_stds = compute_feature_stds(historical_df)

    return {
        "model_bundle": model_bundle,
        "drift_baseline": drift_baseline,
        "feature_stds": feature_stds,
        "historical_df": historical_df,
    }


def score_transaction(txn: dict, resources: dict) -> dict:
    """
    Scores ONE transaction through the full SignalGuard pipeline. This is
    what the live FastAPI endpoint will call for every incoming request.

    Args:
        txn: dict of the transaction's fields (same columns as the CSV -
             risk scores, amount, latency_ms, etc.)
        resources: the dict returned by load_resources().

    Returns:
        A full result dict: component scores, trust score/level, action,
        and (if not PROCEED) a complete incident record with evidence.
    """
    model_bundle = resources["model_bundle"]

    # --- Data Quality ---
    dq_result = assess_data_quality(txn)

    # --- Drift ---
    row_series = pd.Series(txn)
    drift_score, drift_issues = score_drift_row(row_series, resources["drift_baseline"])

    # --- Signal Conflict ---
    conflict_score, conflict_issues = score_conflict_row(row_series)

    # --- Model Confidence ---
    confidence_score, confidence_issues = score_confidence_row(row_series, model_bundle)

    # --- Decision Robustness (Counterfactual) ---
    # counterfactual.score_dataframe is vectorized for batch speed, but
    # works correctly on a single-row dataframe too - reused here rather
    # than writing a separate single-row implementation.
    single_row_df = pd.DataFrame([txn])
    robustness_result = score_robustness_batch(single_row_df, model_bundle, resources["feature_stds"])
    robustness_score = float(robustness_result.iloc[0]["robustness_score"])
    robustness_issues = robustness_result.iloc[0]["robustness_issues"]
    base_recommendation = robustness_result.iloc[0]["base_recommendation"]

    # --- Combine into Trust Score ---
    trust_result = compute_trust_score(
        data_quality=dq_result["data_quality_score"],
        drift=drift_score,
        signal_consistency=conflict_score,
        model_confidence=confidence_score,
        decision_robustness=robustness_score,
    )

    action = get_action(trust_result["trust_level"])

    all_issues = {
        "data_quality": dq_result["all_issues"],
        "drift": drift_issues,
        "signal_conflict": conflict_issues,
        "model_confidence": confidence_issues,
        "decision_robustness": robustness_issues,
    }

    result = {
        "transaction_id": txn.get("transaction_id", "UNKNOWN"),
        "model_recommendation": base_recommendation,
        "trust_score": trust_result["trust_score"],
        "trust_level": trust_result["trust_level"],
        "action": action,
        "component_scores": trust_result["components"],
    }

    # Only attach a full incident record if action isn't a smooth PROCEED -
    # mirrors the same "don't clutter the audit trail" choice from
    # incident_detection.py.
    if action != "PROCEED":
        result["incident"] = build_incident(result["transaction_id"], trust_result, all_issues)
    else:
        result["incident"] = None

    return result


if __name__ == "__main__":
    print("Loading resources (model, drift baseline, feature stds)...")
    resources = load_resources()
    print("Resources loaded.\n")

    print("Scoring a few individual transactions through the live-scoring code path...")
    sample_txns = resources["historical_df"].sample(3, random_state=7).to_dict("records")

    for txn in sample_txns:
        result = score_transaction(txn, resources)
        print(f"\nTransaction: {result['transaction_id']} (true scenario: {txn.get('scenario')})")
        print(f"  Model recommendation: {result['model_recommendation']}")
        print(f"  Trust Score: {result['trust_score']} ({result['trust_level']})")
        print(f"  Action: {result['action']}")
        print(f"  Component scores: {result['component_scores']}")
        if result["incident"]:
            print(f"  Evidence ({result['incident']['n_evidence_items']} items):")
            for item in result["incident"]["evidence"]:
                print(f"    [{item['checker']}] {item['issue']}")

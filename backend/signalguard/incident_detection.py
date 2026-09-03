"""
incident_detection.py
----------------------
Turns a Decision Trust Score result into an action, and packages a full
incident record (with evidence from every checker) whenever a transaction
needs a second look. This is the last step of SignalGuard's core workflow:

    Trust Score -> HIGH/MEDIUM/LOW -> Proceed/Review/Fallback

Design choice: incidents are ONLY created for REVIEW and FALLBACK actions,
not for smooth PROCEED (HIGH_TRUST) transactions. A real system doesn't
clutter its audit trail with routine approvals - only cases that actually
need a human, or an investigation agent, to look at. This also keeps
Day 3's LLM investigation agent's job clear: it only ever sees genuine
incidents, not 50,000 rows of "nothing to see here".

Action mapping:
    HIGH_TRUST   -> PROCEED   (no incident created)
    MEDIUM_TRUST -> REVIEW    (incident created)
    LOW_TRUST    -> FALLBACK  (incident created)

Each incident bundles every checker's evidence (Data Quality, Drift,
Signal Conflict, Model Confidence, Decision Robustness issues) into one
record, so the investigation agent and dashboard have real, specific
reasons to show - not just a bare number.

Run:
    python incident_detection.py
(runs a self-test over the real dataset using the full SignalGuard
pipeline, and reports how many incidents were created and why)
"""

import pandas as pd

TRUST_LEVEL_TO_ACTION = {
    "HIGH_TRUST": "PROCEED",
    "MEDIUM_TRUST": "REVIEW",
    "LOW_TRUST": "FALLBACK",
}

# Trust levels that generate an incident record. HIGH_TRUST (PROCEED) never does.
INCIDENT_TRUST_LEVELS = {"MEDIUM_TRUST", "LOW_TRUST"}


def get_action(trust_level: str) -> str:
    """Maps a trust level to the action SignalGuard recommends."""
    return TRUST_LEVEL_TO_ACTION.get(trust_level, "REVIEW")


def build_incident(
    transaction_id: str,
    trust_result: dict,
    all_issues: dict,
    counterfactual: dict = None,
) -> dict:
    """
    Packages a single incident record.

    Args:
        transaction_id: the transaction's ID.
        trust_result: the dict returned by trust_score.compute_trust_score().
        all_issues: dict of {checker_name: [issue strings]} from every
            checker that ran on this transaction, e.g.:
            {
                "data_quality": [...],
                "drift": [...],
                "signal_conflict": [...],
                "model_confidence": [...],
                "decision_robustness": [...],
            }

    Returns:
        A single incident dict, ready to store / return via the API /
        hand to the Day 3 investigation agent.
    """
    action = get_action(trust_result["trust_level"])

    # Flatten all issues into one evidence list, tagged by which checker
    # raised them, so the audit trail is self-explanatory.
    evidence = []
    for checker_name, issues in all_issues.items():
        for issue in issues:
            evidence.append({"checker": checker_name, "issue": issue})

    return {
        "transaction_id": transaction_id,
        "trust_score": trust_result["trust_score"],
        "trust_level": trust_result["trust_level"],
        "action": action,
        "component_scores": trust_result["components"],
        "evidence": evidence,
        "n_evidence_items": len(evidence),
        "counterfactual": counterfactual,
    }


def detect_incidents(df: pd.DataFrame) -> list:
    """
    Scans a fully-scored dataframe (must already have trust_score,
    trust_level, and every checker's score/issues columns - i.e. run
    AFTER the full trust_score.py pipeline) and returns a list of
    incident records for every MEDIUM_TRUST / LOW_TRUST row.

    HIGH_TRUST (PROCEED) rows are skipped entirely - no incident created.
    """
    incidents = []

    incident_rows = df[df["trust_level"].isin(INCIDENT_TRUST_LEVELS)]

    for _, row in incident_rows.iterrows():
        trust_result = {
            "trust_score": row["trust_score"],
            "trust_level": row["trust_level"],
            "components": {
                "data_quality": row["data_quality_score"],
                "drift": row["drift_score"],
                "signal_consistency": row["conflict_score"],
                "model_confidence": row["confidence_score"],
                "decision_robustness": row.get("robustness_score"),
            },
        }
        all_issues = {
            "data_quality": row.get("all_issues", []),
            "drift": row.get("drift_issues", []),
            "signal_conflict": row.get("conflict_issues", []),
            "model_confidence": row.get("confidence_issues", []),
            "decision_robustness": row.get("robustness_issues", []),
        }
        counterfactual = {
            "original_decision": row.get("base_recommendation"),
            "original_fraud_probability": row.get("base_fraud_probability"),
            "counterfactual_decision": row.get("counterfactual_recommendation"),
            "counterfactual_fraud_probability": row.get("counterfactual_fraud_probability"),
            "flipped": bool(row.get("counterfactual_flipped", False)),
            "changed_feature": row.get("counterfactual_feature") or None,
            "changed_direction": row.get("counterfactual_direction") or None,
        }
        incidents.append(build_incident(row["transaction_id"], trust_result, all_issues, counterfactual))
    return incidents


if __name__ == "__main__":
    import joblib
    from data_quality import assess_data_quality_batch
    from drift import compute_baseline, score_dataframe as score_drift
    from signal_conflict import score_dataframe as score_conflict
    from model_confidence import load_model_bundle, score_dataframe as score_confidence
    from counterfactual import compute_feature_stds, score_dataframe as score_robustness
    from trust_score import compute_trust_score

    print("Loading transactions...")
    df = pd.read_csv("../../data/transactions.csv")

    print("Running full SignalGuard pipeline (all 5 checkers)...")
    df = assess_data_quality_batch(df)
    baseline = compute_baseline(df)
    df = score_drift(df, baseline)
    df = score_conflict(df)
    bundle = load_model_bundle()
    df = score_confidence(df, bundle)
    feature_stds = compute_feature_stds(df)
    df = score_robustness(df, bundle, feature_stds)

    print("Combining into Trust Score...")
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

    print("Detecting incidents (MEDIUM_TRUST / LOW_TRUST only)...")
    incidents = detect_incidents(df)

    print(f"\nTotal transactions: {len(df)}")
    print(f"Total incidents created: {len(incidents)}")
    print(f"  (i.e. {len(df) - len(incidents)} transactions PROCEEDED with no incident logged)")

    action_counts = pd.Series([i["action"] for i in incidents]).value_counts()
    print(f"\nIncident actions breakdown:\n{action_counts}")

    print("\nSample incident (a FALLBACK case if any exist, else first incident):")
    fallback_incidents = [i for i in incidents if i["action"] == "FALLBACK"]
    sample = fallback_incidents[0] if fallback_incidents else incidents[0]
    print(f"  Transaction: {sample['transaction_id']}")
    print(f"  Trust Score: {sample['trust_score']} ({sample['trust_level']})")
    print(f"  Action: {sample['action']}")
    print(f"  Component scores: {sample['component_scores']}")
    print(f"  Evidence ({sample['n_evidence_items']} items):")
    for item in sample["evidence"]:
        print(f"    [{item['checker']}] {item['issue']}")
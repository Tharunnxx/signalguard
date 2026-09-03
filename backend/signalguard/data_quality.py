"""
data_quality.py
----------------
SignalGuard's Data Quality check - the first of five trust checks that feed
into the Decision Trust Score.

Question this answers: "Is the data behind this AI decision actually
complete and sane, or is something missing/broken?"

This does NOT look at whether the transaction is risky or fraudulent - that
is the underlying ML model's job. This only checks whether the DATA feeding
that model can be trusted at all. A model can be 95% confident on garbage
data, and that confidence would be meaningless - this check is what catches
that.

Two kinds of problems are checked for:
1. MISSING SIGNALS - a risk-score field is null (e.g. a signal provider like
   device fingerprinting failed to respond).
2. SANITY VIOLATIONS - a field has a value that isn't just missing, it's
   physically/logically invalid (e.g. a negative amount, a risk score
   outside 0-1, impossible latency).

Score: starts at 100, deductions applied per issue found, floored at 0.
"""

import pandas as pd
import numpy as np

# The 5 risk-score signals every transaction should have.
# These correspond directly to the MISSING_SIGNAL injection scenario.
RISK_SCORE_COLUMNS = [
    "ip_risk_score",
    "device_risk_score",
    "location_risk_score",
    "bank_risk_score",
    "amount_risk_score",
]

# Point penalty per missing risk signal. 5 columns x 15 = 75 max deduction,
# leaving room for sanity-check penalties to still matter even when several
# signals are missing at once.
MISSING_SIGNAL_PENALTY = 15

# Point penalty per sanity violation found (kept lower than a missing
# signal - bad data is bad, but a null is the more common, "expected"
# failure mode we're primarily guarding against).
SANITY_VIOLATION_PENALTY = 10

# Reasonable real-world bounds used for sanity checks. Latency above 10s
# almost certainly indicates a broken reading, not real processing time.
MAX_PLAUSIBLE_LATENCY_MS = 10_000


def _check_missing_signals(txn: dict) -> list:
    """Returns a list of human-readable issue strings for any missing risk score."""
    issues = []
    for col in RISK_SCORE_COLUMNS:
        value = txn.get(col)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            issues.append(f"Missing signal: '{col}' is null")
    return issues


def _check_sanity_violations(txn: dict) -> list:
    """Returns a list of human-readable issue strings for physically invalid values."""
    issues = []

    amount = txn.get("amount")
    if amount is not None and not pd.isna(amount) and amount <= 0:
        issues.append(f"Invalid amount: {amount} (must be positive)")

    latency = txn.get("latency_ms")
    if latency is not None and not pd.isna(latency):
        if latency < 0:
            issues.append(f"Invalid latency: {latency}ms (cannot be negative)")
        elif latency > MAX_PLAUSIBLE_LATENCY_MS:
            issues.append(f"Implausible latency: {latency}ms (exceeds {MAX_PLAUSIBLE_LATENCY_MS}ms)")

    retry_count = txn.get("retry_count")
    if retry_count is not None and not pd.isna(retry_count) and retry_count < 0:
        issues.append(f"Invalid retry_count: {retry_count} (cannot be negative)")

    txn_velocity = txn.get("txn_velocity_1h")
    if txn_velocity is not None and not pd.isna(txn_velocity) and txn_velocity < 0:
        issues.append(f"Invalid txn_velocity_1h: {txn_velocity} (cannot be negative)")

    for col in RISK_SCORE_COLUMNS:
        value = txn.get(col)
        if value is not None and not pd.isna(value):
            if value < 0 or value > 1:
                issues.append(f"Out-of-range risk score: '{col}' = {value} (expected 0-1)")

    return issues


def assess_data_quality(txn: dict) -> dict:
    """
    Assesses a single transaction's data quality.

    Args:
        txn: dict (or dict-like row) with the transaction's fields.

    Returns:
        {
            "data_quality_score": float 0-100,
            "missing_signals": [...],
            "sanity_violations": [...],
            "all_issues": [...],   # combined, for the audit trail / evidence panel
        }
    """
    missing_signals = _check_missing_signals(txn)
    sanity_violations = _check_sanity_violations(txn)

    score = 100
    score -= len(missing_signals) * MISSING_SIGNAL_PENALTY
    score -= len(sanity_violations) * SANITY_VIOLATION_PENALTY
    score = max(0, min(100, score))

    return {
        "data_quality_score": float(score),
        "missing_signals": missing_signals,
        "sanity_violations": sanity_violations,
        "all_issues": missing_signals + sanity_violations,
    }


def assess_data_quality_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs assess_data_quality on every row of a DataFrame.
    Returns the original DataFrame with new columns:
        data_quality_score, n_issues
    Useful for evaluation - e.g. checking that MISSING_SIGNAL-scenario rows
    actually score lower than NORMAL rows.
    """
    results = df.to_dict("records")
    scored = [assess_data_quality(row) for row in results]

    df = df.copy()
    df["data_quality_score"] = [r["data_quality_score"] for r in scored]
    df["n_issues"] = [len(r["all_issues"]) for r in scored]
    return df


if __name__ == "__main__":
    # Quick self-test against the real dataset: confirms MISSING_SIGNAL rows
    # score lower than NORMAL rows, which is the whole point of this check.
    DATA_PATH = "../../data/transactions.csv"
    print("Loading transactions...")
    df = pd.read_csv(DATA_PATH)

    print("Scoring data quality for all transactions...")
    scored_df = assess_data_quality_batch(df)

    print("\nAverage Data Quality Score by scenario:")
    print(scored_df.groupby("scenario")["data_quality_score"].mean().sort_values())

    print("\nSample issues found (first 3 MISSING_SIGNAL rows):")
    missing_rows = scored_df[scored_df["scenario"] == "MISSING_SIGNAL"].head(3)
    for _, row in missing_rows.iterrows():
        result = assess_data_quality(row.to_dict())
        print(f"  {row['transaction_id']}: score={result['data_quality_score']}, issues={result['all_issues']}")

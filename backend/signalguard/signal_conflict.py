"""
signal_conflict.py
-------------------
Signal Conflict checker for SignalGuard.

Answers: "are the risk signals on this transaction contradicting each
other?" A model that averages two signals which flatly disagree (one says
very safe, another says very risky) is making a fragile decision - that's
exactly what this catches. Targets SIGNAL_CONFLICT (and partially
COMBINED_INCIDENT, BANK_DRIFT, since those also involve one signal
disagreeing with the others).

Approach: generic all-pairs comparison (not hardcoded to the demo's
specific device-vs-ip pattern - this checks every pair among the 5 risk
score columns, so it stays robust to conflicts SignalGuard wasn't
specifically told to expect).

    1. For every pair of the 5 risk-score columns, compute |difference|.
    2. If a pair differs by more than CONFLICT_THRESHOLD, that's one
       "conflict" (threshold tuned against real data: 0.6 keeps false
       positives on NORMAL traffic to ~2%, while still catching injected
       conflicts, which typically differ by 0.6-0.9).
    3. Each conflicting pair costs PENALTY_PER_PAIR points, capped at
       MAX_TOTAL_PENALTY so no single row can be zeroed by conflicts alone.

Run:
    python signal_conflict.py
"""

from itertools import combinations

import pandas as pd

RISK_COLUMNS = [
    "ip_risk_score",
    "device_risk_score",
    "location_risk_score",
    "bank_risk_score",
    "amount_risk_score",
]

CONFLICT_THRESHOLD = 0.6   # tuned against real data: ~2% false-positive rate on NORMAL
PENALTY_PER_PAIR = 20.0
MAX_TOTAL_PENALTY = 80.0

RISK_PAIRS = list(combinations(RISK_COLUMNS, 2))


def score_row(row: pd.Series) -> tuple[float, list[str]]:
    """
    Scores a single transaction's signal conflict.
    Returns (score, issues) where issues is a list of human-readable
    explanations for any conflicts found (for the dashboard's evidence trail).
    """
    issues = []

    for c1, c2 in RISK_PAIRS:
        v1, v2 = row[c1], row[c2]

        if pd.isna(v1) or pd.isna(v2):
            continue  # missing values are Data Quality's job, not Conflict's

        diff = abs(v1 - v2)
        if diff > CONFLICT_THRESHOLD:
            safe_col, risky_col = (c1, c2) if v1 < v2 else (c2, c1)
            issues.append(
                f"Signal conflict: '{safe_col}' says safe ({row[safe_col]:.2f}) "
                f"but '{risky_col}' says risky ({row[risky_col]:.2f})"
            )

    penalty = min(len(issues) * PENALTY_PER_PAIR, MAX_TOTAL_PENALTY)
    score = round(max(0.0, 100.0 - penalty), 2)
    return score, issues


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores every row in df. Adds 'conflict_score' and 'conflict_issues' columns.
    """
    scores = []
    issues_list = []
    for _, row in df.iterrows():
        score, issues = score_row(row)
        scores.append(score)
        issues_list.append(issues)

    df = df.copy()
    df["conflict_score"] = scores
    df["conflict_issues"] = issues_list
    return df


if __name__ == "__main__":
    print("Loading transactions...")
    df = pd.read_csv("../../data/transactions.csv")

    print("Scoring signal conflict for all transactions...")
    scored = score_dataframe(df)

    print("\nAverage Signal Conflict Score by scenario:")
    print(scored.groupby("scenario")["conflict_score"].mean().round(2))

    print("\nSample issues found (first 3 SIGNAL_CONFLICT rows):")
    conflict_rows = scored[scored["scenario"] == "SIGNAL_CONFLICT"].head(3)
    for _, r in conflict_rows.iterrows():
        print(f"  {r['transaction_id']}: score={r['conflict_score']}, issues={r['conflict_issues']}")

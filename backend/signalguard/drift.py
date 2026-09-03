"""
drift.py
--------
Drift checker for SignalGuard.

Answers: "has this transaction's data pattern shifted away from what's
normal / what the model expects?" Targets the BANK_DRIFT scenario (and
naturally also flags LATENCY_ANOMALY, since latency is one of the two
features monitored).

Approach: z-score against a NORMAL-only baseline.
    1. Compute mean/std of key features using ONLY rows labeled NORMAL
       (this is the "healthy" reference distribution).
    2. For every transaction, compute how many standard deviations (z)
       it is from that baseline for each monitored feature.
    3. Below z=2: no penalty (ordinary variation).
       Above z=2: penalty scales with how extreme z is, capped at 70
       so one huge outlier can't be the only thing that matters.
    4. Final score = 100 - (penalty from the worst-offending feature).

Monitored features: bank_risk_score, latency_ms
(the two features that BANK_DRIFT actually manipulates; keeping scope
tight avoids diluting the signal with unrelated features).

Run:
    python drift.py
"""

import numpy as np
import pandas as pd

MONITORED_FEATURES = ["bank_risk_score", "latency_ms"]

Z_THRESHOLD = 2.0      # below this z-score, no penalty at all
PENALTY_SLOPE = 15.0   # penalty points per unit of z beyond the threshold
MAX_PENALTY = 70.0     # cap so one extreme feature can't fully zero the score


def compute_baseline(df: pd.DataFrame) -> dict:
    """
    Computes mean/std for each monitored feature using only NORMAL rows.
    Returns a dict like {"bank_risk_score": (mean, std), ...}.
    This baseline is what "normal" looks like, and should be computed once
    (e.g. at startup / from training data) and reused for scoring live
    transactions later.
    """
    normal_rows = df[df["scenario"] == "NORMAL"] if "scenario" in df.columns else df
    baseline = {}
    for col in MONITORED_FEATURES:
        mean = normal_rows[col].mean()
        std = normal_rows[col].std()
        baseline[col] = (mean, std)
    return baseline


def _penalty_from_z(z: float) -> float:
    """Converts a z-score into a 0-70 penalty. No penalty below Z_THRESHOLD."""
    return float(np.clip((abs(z) - Z_THRESHOLD) * PENALTY_SLOPE, 0, MAX_PENALTY))


def score_row(row: pd.Series, baseline: dict) -> tuple[float, list[str]]:
    """
    Scores a single transaction's drift.
    Returns (score, issues) where issues is a list of human-readable
    explanations for any drift found (for the dashboard's evidence trail).
    """
    penalties = {}
    issues = []

    for col in MONITORED_FEATURES:
        mean, std = baseline[col]
        value = row[col]

        if pd.isna(value) or std == 0 or pd.isna(std):
            continue  # missing values are Data Quality's job, not Drift's

        z = (value - mean) / std
        penalty = _penalty_from_z(z)
        penalties[col] = penalty

        if penalty > 0:
            direction = "above" if z > 0 else "below"
            issues.append(
                f"Drift: '{col}' is {abs(z):.1f} std devs {direction} normal "
                f"(value={value:.2f}, expected ~{mean:.2f})"
            )

    worst_penalty = max(penalties.values()) if penalties else 0.0
    score = round(max(0.0, 100.0 - worst_penalty), 2)
    return score, issues


def score_dataframe(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """
    Scores every row in df. Adds 'drift_score' and 'drift_issues' columns.
    """
    scores = []
    issues_list = []
    for _, row in df.iterrows():
        score, issues = score_row(row, baseline)
        scores.append(score)
        issues_list.append(issues)

    df = df.copy()
    df["drift_score"] = scores
    df["drift_issues"] = issues_list
    return df


if __name__ == "__main__":
    print("Loading transactions...")
    df = pd.read_csv("../../data/transactions.csv")

    print("Computing NORMAL baseline...")
    baseline = compute_baseline(df)
    for col, (mean, std) in baseline.items():
        print(f"  {col}: mean={mean:.3f}, std={std:.3f}")

    print("\nScoring drift for all transactions...")
    scored = score_dataframe(df, baseline)

    print("\nAverage Drift Score by scenario:")
    print(scored.groupby("scenario")["drift_score"].mean().round(2))

    print("\nSample issues found (first 3 BANK_DRIFT rows):")
    bank_drift_rows = scored[scored["scenario"] == "BANK_DRIFT"].head(3)
    for _, r in bank_drift_rows.iterrows():
        print(f"  {r['transaction_id']}: score={r['drift_score']}, issues={r['drift_issues']}")

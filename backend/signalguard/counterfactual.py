"""
counterfactual.py
------------------
Decision Robustness checker for SignalGuard - the main technical/demo
differentiator.

Answers: "If one risk signal had been slightly different, would the
model's decision have flipped?" A decision that survives small, realistic
nudges to its inputs is robust and can be trusted. A decision that flips
from a tiny nudge is fragile - the model was sitting right on the edge,
not confidently right, even if the raw recommendation looked clean.

Approach:
    1. Take the transaction's real feature values and get the model's
       baseline recommendation (APPROVE / REVIEW / DECLINE).
    2. For each of 8 features the model actually leans on most heavily
       (by |coefficient|: txn_velocity_1h, retry_count, amount_deviation,
       previous_txn_count, account_age_days, avg_amount_7d, amount,
       latency_ms), nudge it up by 1 standard deviation and down by 1
       standard deviation (std computed from the real dataset - a
       realistic, defensible amount of "what if this were a bit
       different", not an arbitrary number).
    3. Re-run the model on each of the 16 nudged versions (8 features x 2
       directions) and check whether the recommendation changed.
    4. Robustness score = 100 - 12 points per flip, floored at 0. A
       decision with 0 flips out of 16 nudges stays at 100; a decision
       that flips on more than 8 nudges bottoms out at 0.

Verified against the real trained model + real data: HIGH_VELOCITY (the
most clear-cut fraud pattern) scores a perfect 100 - completely robust.
NORMAL and MISSING_SIGNAL average ~77-78 - not because they're wrong, but
because their baseline fraud probability sits close to the 0.3 APPROVE/
REVIEW boundary, so a real subset of ordinary-looking transactions have
genuinely fragile decisions. This is an honest finding, not a target to
"fix": it is exactly the kind of fragility SignalGuard exists to surface.

Run:
    python counterfactual.py
"""

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../models/risk_model.joblib"
DATA_PATH = "../../data/transactions.csv"

# The 8 features the model leans on most heavily (by |coefficient|),
# used for nudging. Keeping this list focused (not all 14 features) keeps
# runtime reasonable and avoids diluting the signal with low-influence
# features that rarely move a decision anyway.
PERTURB_FEATURES = [
    "txn_velocity_1h",
    "retry_count",
    "amount_deviation",
    "previous_txn_count",
    "account_age_days",
    "avg_amount_7d",
    "amount",
    "latency_ms",
]

# Features that can never go negative in the real world - nudges are
# clipped at 0 for these instead of allowed to go negative.
NONNEGATIVE_FEATURES = {
    "txn_velocity_1h", "retry_count", "previous_txn_count",
    "account_age_days", "avg_amount_7d", "amount", "latency_ms",
}

PENALTY_PER_FLIP = 12.0

APPROVE_THRESHOLD = 0.3
DECLINE_THRESHOLD = 0.7


def load_model_bundle(path: str = MODEL_PATH) -> dict:
    """Loads the self-contained model bundle (model + imputer + scaler + features)."""
    return joblib.load(path)


def compute_feature_stds(df: pd.DataFrame) -> pd.Series:
    """
    Computes the standard deviation of each perturbable feature from the
    real dataset. This defines what a "realistic nudge" size looks like
    for each feature (features have very different scales, e.g. amount
    vs. amount_deviation).
    """
    return df[PERTURB_FEATURES].std()


def _get_recommendation(fraud_proba):
    """Vectorized: converts fraud probability into APPROVE/REVIEW/DECLINE."""
    return np.where(
        fraud_proba < APPROVE_THRESHOLD, "APPROVE",
        np.where(fraud_proba < DECLINE_THRESHOLD, "REVIEW", "DECLINE"),
    )


def _batch_predict_proba(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    X = df[bundle["feature_columns"]]
    X_imputed = bundle["imputer"].transform(X)
    X_scaled = bundle["scaler"].transform(X_imputed)
    return bundle["model"].predict_proba(X_scaled)[:, 1]


def score_dataframe(df: pd.DataFrame, bundle: dict, feature_stds: pd.Series) -> pd.DataFrame:
    """
    Scores every row in df for Decision Robustness. Vectorized: runs one
    batch prediction per nudge (16 total) rather than one prediction per
    row per nudge, so this stays fast even on tens of thousands of rows.

    Adds 'robustness_score', 'n_flips', and 'robustness_issues' columns.
    """
    df = df.copy()

    base_proba = _batch_predict_proba(df, bundle)
    base_rec = _get_recommendation(base_proba)

    flip_counts = np.zeros(len(df), dtype=int)
    flip_features = [[] for _ in range(len(df))]

    # Tracks the single most informative nudge per row, for showing a real
    # "original vs counterfactual" pair in the UI. Prefers an actual flip
    # over a non-flip, and among flips/non-flips prefers the biggest
    # probability swing - so the demo shows the most dramatic real example,
    # not an arbitrary one of the 16 nudges tried.
    best_delta = np.full(len(df), -1.0)
    best_cf_proba = base_proba.copy()
    best_cf_rec = base_rec.copy()
    best_flipped = np.zeros(len(df), dtype=bool)
    best_feat_name = np.array([""] * len(df), dtype=object)
    best_feat_direction = np.array([""] * len(df), dtype=object)

    for feat in PERTURB_FEATURES:
        std = feature_stds[feat]
        if std == 0 or pd.isna(std):
            continue
        for direction, label in [(1, "up"), (-1, "down")]:
            df_nudged = df.copy()
            new_val = df_nudged[feat] + direction * std
            if feat in NONNEGATIVE_FEATURES:
                new_val = new_val.clip(lower=0)
            df_nudged[feat] = new_val

            nudged_proba = _batch_predict_proba(df_nudged, bundle)
            nudged_rec = _get_recommendation(nudged_proba)

            flipped = nudged_rec != base_rec
            flip_counts += flipped.astype(int)
            for i in np.where(flipped)[0]:
                flip_features[i].append(f"{feat} ({label} 1 std)")

            delta = np.abs(nudged_proba - base_proba)
            prefer = (flipped & ~best_flipped) | ((flipped == best_flipped) & (delta > best_delta))
            best_delta = np.where(prefer, delta, best_delta)
            best_cf_proba = np.where(prefer, nudged_proba, best_cf_proba)
            best_cf_rec = np.where(prefer, nudged_rec, best_cf_rec)
            best_flipped = np.where(prefer, flipped, best_flipped)
            best_feat_name = np.where(prefer, feat, best_feat_name)
            best_feat_direction = np.where(prefer, label, best_feat_direction)

    scores = np.clip(100.0 - flip_counts * PENALTY_PER_FLIP, 0, 100)

    issues = [
        [f"Decision Robustness: recommendation flips when nudging {', '.join(feats)}"]
        if feats else []
        for feats in flip_features
    ]

    df["base_recommendation"] = base_rec
    df["base_fraud_probability"] = np.round(base_proba * 100, 2)
    df["n_flips"] = flip_counts
    df["robustness_score"] = np.round(scores, 2)
    df["robustness_issues"] = issues
    df["counterfactual_recommendation"] = best_cf_rec
    df["counterfactual_fraud_probability"] = np.round(best_cf_proba.astype(float) * 100, 2)
    df["counterfactual_flipped"] = best_flipped
    df["counterfactual_feature"] = best_feat_name
    df["counterfactual_direction"] = best_feat_direction
    return df


if __name__ == "__main__":
    print("Loading model bundle...")
    bundle = load_model_bundle()

    print("Loading transactions...")
    df = pd.read_csv(DATA_PATH)

    print("Computing feature standard deviations (nudge sizes)...")
    feature_stds = compute_feature_stds(df)
    print(feature_stds.round(3))

    print("\nRunning counterfactual tests (16 nudges per transaction)...")
    scored = score_dataframe(df, bundle, feature_stds)

    print("\nAverage Decision Robustness Score by scenario:")
    print(scored.groupby("scenario")["robustness_score"].mean().round(2))

    print("\nSample issues found (first 3 NORMAL rows with flips):")
    fragile_rows = scored[(scored["scenario"] == "NORMAL") & (scored["n_flips"] > 0)].head(3)
    for _, r in fragile_rows.iterrows():
        print(f"  {r['transaction_id']}: score={r['robustness_score']}, "
              f"base={r['base_recommendation']}, flips={r['n_flips']}, issues={r['robustness_issues']}")

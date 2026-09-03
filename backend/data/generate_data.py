"""
generate_data.py
-----------------
Generates the SignalGuard synthetic payment dataset.

IMPORTANT: This data is 100% synthetic. It does NOT represent real Razorpay
transactions, users, or fraud patterns. It exists purely to give SignalGuard
a controlled environment with known ground truth to demonstrate and evaluate
against.

Run:
    python generate_data.py

Outputs:
    ../../data/transactions.csv        (the full dataset, model-facing columns
                                         + ground truth columns together)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from scenarios import SCENARIO_REGISTRY

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
SEED = 42
N_TRANSACTIONS = 50_000
N_USERS = 1_500
DAYS_WINDOW = 30  # transactions spread over the last 30 days

# FRAUD threshold on actual_risk. NOTE: this was originally 0.55, but given
# how the risk-score components are distributed (Beta(2,8), mean ~0.2),
# actual_risk almost never exceeds ~0.55 in practice - only ~16/50000 rows
# did. That meant ~98% of "FRAUD" labels were coming from the 2% random
# label-noise flip instead of the real risk formula, making the label
# essentially unlearnable. Lowered to align with the real distribution
# (roughly the 97-98th percentile of actual_risk), so FRAUD labels are
# actually driven by genuine risk signal.
FRAUD_THRESHOLD = 0.35

# Fraction of transactions that get an abnormal scenario injected.
# Kept deliberately small - incidents should be the exception, not the norm.
SCENARIO_FRACTIONS = {
    "BANK_DRIFT": 0.03,
    "LATENCY_ANOMALY": 0.02,
    "MISSING_SIGNAL": 0.03,
    "SIGNAL_CONFLICT": 0.02,
    "HIGH_VELOCITY": 0.02,
    "COMBINED_INCIDENT": 0.01,
}  # remaining ~87% stays NORMAL

BANKS = [
    "HDFC Bank", "ICICI Bank", "State Bank of India", "Axis Bank",
    "Kotak Mahindra Bank", "Yes Bank", "Punjab National Bank",
    "IndusInd Bank", "IDFC First Bank", "Bank of Baroda",
]
CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]
DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.72, 0.22, 0.06]
PAYMENT_METHODS = ["UPI", "Card", "Netbanking", "Wallet"]
PAYMENT_WEIGHTS = [0.45, 0.30, 0.15, 0.10]
CARD_NETWORKS = ["Visa", "Mastercard", "RuPay", "Amex"]


def make_users(rng, n_users):
    """
    Each synthetic user has a stable 'personality' - a spend tier, a home
    bank, a home city, a device, and an account age. Transactions are then
    sampled per-user so amounts and behavior stay internally consistent
    (e.g. a low-spend user doesn't suddenly show up with wildly random
    amounts every time).
    """
    account_age_days = rng.integers(1, 3000, size=n_users)
    # more established accounts naturally have more transaction history
    previous_txn_count = np.clip(
        (account_age_days / 6) + rng.normal(0, 15, size=n_users), 0, None
    ).astype(int)

    # spend tier drives avg_amount_7d: lognormal gives realistic
    # "many small/medium spenders, few big spenders" shape
    spend_tier = rng.lognormal(mean=6.5, sigma=0.9, size=n_users)  # ~ INR 300-15000
    spend_tier = np.clip(spend_tier, 50, 200_000)

    users = pd.DataFrame({
        "user_id": [f"U{100000+i}" for i in range(n_users)],
        "account_age_days": account_age_days,
        "previous_txn_count": previous_txn_count,
        "avg_amount_7d": spend_tier,
        "home_bank": rng.choice(BANKS, size=n_users),
        "home_city": rng.choice(CITIES, size=n_users),
        "home_device": rng.choice(DEVICE_TYPES, size=n_users, p=DEVICE_WEIGHTS),
        "device_age_days": rng.integers(1, 1500, size=n_users),
    })
    return users


def make_transactions(rng, users, n_txn):
    """
    Samples transactions from users (with repeats - active users transact
    more often), and generates per-transaction fields correlated to each
    user's baseline.
    """
    # active users transact more: weight sampling by previous_txn_count
    weights = users["previous_txn_count"] + 1
    weights = weights / weights.sum()
    sampled_idx = rng.choice(users.index, size=n_txn, p=weights)
    u = users.loc[sampled_idx].reset_index(drop=True)

    now = datetime(2026, 8, 31)
    start = now - timedelta(days=DAYS_WINDOW)
    timestamps = [
        start + timedelta(seconds=int(s))
        for s in rng.integers(0, DAYS_WINDOW * 24 * 3600, size=n_txn)
    ]

    payment_method = rng.choice(PAYMENT_METHODS, size=n_txn, p=PAYMENT_WEIGHTS)
    card_network = np.where(
        payment_method == "Card",
        rng.choice(CARD_NETWORKS, size=n_txn),
        "NOT_CARD",
    )

    # amount = user's typical spend * a multiplier close to 1 (normal variation)
    multiplier = rng.lognormal(mean=0.0, sigma=0.35, size=n_txn)
    amount = np.round(u["avg_amount_7d"].values * multiplier, 2)

    amount_deviation = (amount - u["avg_amount_7d"].values) / (u["avg_amount_7d"].values + 1e-6)

    txn_velocity_1h = rng.poisson(0.3, size=n_txn)
    latency_ms = np.round(rng.lognormal(mean=4.9, sigma=0.4, size=n_txn), 1)  # ~ 80-400ms typical
    retry_count = rng.poisson(0.1, size=n_txn)

    # risk scores: normal transactions skew low (Beta(2,8) has mean ~0.2, long right tail)
    ip_risk_score = rng.beta(2, 8, size=n_txn)
    device_risk_score = rng.beta(2, 8, size=n_txn)
    location_risk_score = rng.beta(2, 10, size=n_txn)
    bank_risk_score = rng.beta(2, 8, size=n_txn)
    amount_risk_score = np.clip(np.abs(amount_deviation) * 0.4 + rng.beta(2, 10, size=n_txn), 0, 1)

    df = pd.DataFrame({
        "transaction_id": [f"T{500000+i}" for i in range(n_txn)],
        "timestamp": timestamps,
        "user_id": u["user_id"].values,
        "amount": amount,
        "payment_method": payment_method,
        "bank": u["home_bank"].values,
        "card_network": card_network,
        "location": u["home_city"].values,
        "device_type": u["home_device"].values,
        "device_age_days": u["device_age_days"].values,
        "account_age_days": u["account_age_days"].values,
        "previous_txn_count": u["previous_txn_count"].values,
        "txn_velocity_1h": txn_velocity_1h,
        "avg_amount_7d": np.round(u["avg_amount_7d"].values, 2),
        "amount_deviation": np.round(amount_deviation, 4),
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "ip_risk_score": np.round(ip_risk_score, 4),
        "device_risk_score": np.round(device_risk_score, 4),
        "location_risk_score": np.round(location_risk_score, 4),
        "bank_risk_score": np.round(bank_risk_score, 4),
        "amount_risk_score": np.round(amount_risk_score, 4),
        "scenario": "NORMAL",  # default; overwritten for injected rows
    })
    return df


def compute_ground_truth(df, rng):
    """
    Computes 'actual_risk' (continuous 0-1) and 'outcome' (LEGIT/FRAUD) from
    the FINAL feature values - i.e. this runs AFTER scenario injection, so
    ground truth is causally consistent with whatever the data actually looks
    like, not arbitrarily assigned.

    This is a hand-built (not learned) risk function - intentionally simple
    and transparent, since it defines the "true" answer the ML model will
    later try to approximate.
    """
    risk_cols = ["ip_risk_score", "device_risk_score", "location_risk_score",
                 "bank_risk_score", "amount_risk_score"]

    # missing risk signals are treated as "unknown -> moderately risky" (0.5)
    # for the purposes of computing ground truth, mirroring how a real risk
    # engine would fall back on missing data.
    filled = df[risk_cols].fillna(0.5)
    mean_risk_score = filled.mean(axis=1)

    velocity_risk = np.clip(df["txn_velocity_1h"] / 15, 0, 1)
    retry_risk = np.clip(df["retry_count"] / 8, 0, 1)
    deviation_risk = np.clip(np.abs(df["amount_deviation"]) / 3, 0, 1)
    new_account_risk = np.clip(1 - (df["account_age_days"] / 400), 0, 1)

    combined = (
        0.40 * mean_risk_score
        + 0.20 * velocity_risk
        + 0.15 * retry_risk
        + 0.15 * deviation_risk
        + 0.10 * new_account_risk
    )

    noise = rng.normal(0, 0.05, size=len(df))
    actual_risk = np.clip(combined + noise, 0, 1)

    label_noise = rng.random(len(df)) < 0.02  # 2% intentional label noise, like real data
    outcome = np.where(actual_risk > FRAUD_THRESHOLD, "FRAUD", "LEGIT")
    outcome = np.where(label_noise, np.where(outcome == "FRAUD", "LEGIT", "FRAUD"), outcome)

    df["actual_risk"] = np.round(actual_risk, 4)
    df["outcome"] = outcome
    return df


def inject_all_scenarios(df, rng):
    """
    Splits off non-overlapping row subsets for each abnormal scenario and
    applies the corresponding injector. Rows not selected for any scenario
    stay NORMAL.
    """
    n = len(df)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)

    cursor = 0
    for scenario_name, fraction in SCENARIO_FRACTIONS.items():
        count = int(n * fraction)
        idx = all_idx[cursor: cursor + count]
        cursor += count
        injector = SCENARIO_REGISTRY[scenario_name]
        df = injector(df, idx, rng)

    return df


def main():
    rng = np.random.default_rng(SEED)

    print(f"Generating {N_USERS} synthetic users...")
    users = make_users(rng, N_USERS)

    print(f"Generating {N_TRANSACTIONS} synthetic transactions...")
    df = make_transactions(rng, users, N_TRANSACTIONS)

    print("Injecting labeled abnormal scenarios...")
    df = inject_all_scenarios(df, rng)

    print("Computing ground truth (actual_risk, outcome) from final feature values...")
    df = compute_ground_truth(df, rng)

    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = "../../data/transactions.csv"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df)} rows to {out_path}")
    print("\nScenario distribution:")
    print(df["scenario"].value_counts())
    print("\nOutcome distribution:")
    print(df["outcome"].value_counts())
    print(f"\nMissing values per column (from MISSING_SIGNAL injection):")
    print(df[["ip_risk_score", "device_risk_score", "location_risk_score", "amount_risk_score"]].isna().sum())


if __name__ == "__main__":
    main()
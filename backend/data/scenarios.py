"""
scenarios.py
------------
Each function here takes the full transactions DataFrame and a list of row
indices, and deliberately corrupts those rows in one specific, controlled way.

Rules every scenario function follows:
1. It ONLY touches the rows given to it (via `idx`).
2. It sets df.loc[idx, 'scenario'] to its own name (the ground-truth label).
3. It does NOT touch 'outcome' or 'actual_risk' directly - those are
   recomputed centrally in generate_data.py AFTER all injections, from the
   final feature values. This keeps ground truth causally tied to the data.
"""

import numpy as np


def inject_bank_drift(df, idx, rng, target_bank=None):
    """
    Simulates one bank's traffic pattern suddenly shifting - e.g. a bank
    changed something on their end (new fraud wave, processing change).
    Raises that bank's risk baseline and latency for affected rows.
    """
    bank = target_bank or rng.choice(df["bank"].unique())
    df.loc[idx, "bank"] = bank
    df.loc[idx, "bank_risk_score"] = np.clip(
        rng.beta(6, 3, size=len(idx)), 0, 1
    )  # normally beta(2,8) -> low; drifted -> skewed high
    df.loc[idx, "latency_ms"] = df.loc[idx, "latency_ms"] * rng.uniform(1.5, 2.5, size=len(idx))
    df.loc[idx, "scenario"] = "BANK_DRIFT"
    return df


def inject_latency_anomaly(df, idx, rng):
    """
    Simulates infra stress / a degraded processing path.
    Latency spikes hard; nothing else about the transaction is unusual.
    """
    df.loc[idx, "latency_ms"] = df.loc[idx, "latency_ms"] * rng.uniform(5, 15, size=len(idx))
    df.loc[idx, "scenario"] = "LATENCY_ANOMALY"
    return df


def inject_missing_signal(df, idx, rng):
    """
    Simulates a signal provider (device fingerprinting, IP intel, etc.)
    failing to return data. Nulls out one or two risk-score columns.
    This is what the Data Quality checker is supposed to catch.
    """
    risk_cols = ["ip_risk_score", "device_risk_score", "location_risk_score", "amount_risk_score"]
    for i in idx:
        n_missing = rng.choice([1, 2], p=[0.7, 0.3])
        cols_to_null = rng.choice(risk_cols, size=n_missing, replace=False)
        df.loc[i, cols_to_null] = np.nan
    df.loc[idx, "scenario"] = "MISSING_SIGNAL"
    return df


def inject_signal_conflict(df, idx, rng):
    """
    Simulates contradictory evidence: one signal says 'very safe',
    another says 'very risky', on the SAME transaction. Real fraud
    signals rarely agree perfectly, but this exaggerates it deliberately
    so the Signal Conflict checker has something clear to detect.
    """
    n = len(idx)
    # device says very safe, IP says very risky (classic conflict pattern)
    df.loc[idx, "device_risk_score"] = rng.uniform(0.0, 0.1, size=n)
    df.loc[idx, "ip_risk_score"] = rng.uniform(0.85, 1.0, size=n)
    # also: long-standing account (trusted) but odd location (untrusted)
    df.loc[idx, "account_age_days"] = rng.integers(800, 2500, size=n)
    df.loc[idx, "location_risk_score"] = rng.uniform(0.8, 1.0, size=n)
    df.loc[idx, "scenario"] = "SIGNAL_CONFLICT"
    return df


def inject_high_velocity(df, idx, rng):
    """
    Simulates rapid repeated attempts - card testing / automated abuse.
    """
    n = len(idx)
    df.loc[idx, "txn_velocity_1h"] = rng.integers(8, 25, size=n)
    df.loc[idx, "retry_count"] = rng.integers(3, 10, size=n)
    df.loc[idx, "scenario"] = "HIGH_VELOCITY"
    return df


def inject_combined_incident(df, idx, rng):
    """
    Worst case: several problems stack on the same transaction at once.
    This is the scenario that should produce the lowest trust scores.
    """
    n = len(idx)
    df.loc[idx, "bank_risk_score"] = np.clip(rng.beta(6, 3, size=n), 0, 1)
    df.loc[idx, "latency_ms"] = df.loc[idx, "latency_ms"] * rng.uniform(3, 8, size=n)
    df.loc[idx, "txn_velocity_1h"] = rng.integers(6, 20, size=n)
    df.loc[idx, "device_risk_score"] = rng.uniform(0.0, 0.1, size=n)
    df.loc[idx, "ip_risk_score"] = rng.uniform(0.85, 1.0, size=n)
    df.loc[idx, "scenario"] = "COMBINED_INCIDENT"
    return df


SCENARIO_REGISTRY = {
    "BANK_DRIFT": inject_bank_drift,
    "LATENCY_ANOMALY": inject_latency_anomaly,
    "MISSING_SIGNAL": inject_missing_signal,
    "SIGNAL_CONFLICT": inject_signal_conflict,
    "HIGH_VELOCITY": inject_high_velocity,
    "COMBINED_INCIDENT": inject_combined_incident,
}

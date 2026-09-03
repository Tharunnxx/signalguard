"""
api/transactions.py
--------------------
Transaction-related endpoints:
    POST /api/transactions/score   - score ONE new transaction live
    GET  /api/transactions         - browse the batch-scored historical set
    GET  /api/transactions/{id}    - look up one historical transaction's result
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from signalguard.pipeline import score_transaction

router = APIRouter(prefix="/api/transactions", tags=["transactions"])


class TransactionInput(BaseModel):
    """
    A single transaction to score live. Mirrors the feature columns the
    risk model was trained on. The 5 risk-score fields are optional
    (None) so a caller can simulate a signal provider failing to respond -
    exactly what the Data Quality checker is built to catch.
    """
    transaction_id: Optional[str] = None
    amount: float
    device_age_days: float
    account_age_days: float
    previous_txn_count: float
    txn_velocity_1h: float
    avg_amount_7d: float
    amount_deviation: float
    latency_ms: float
    retry_count: float
    ip_risk_score: Optional[float] = None
    device_risk_score: Optional[float] = None
    location_risk_score: Optional[float] = None
    bank_risk_score: Optional[float] = None
    amount_risk_score: Optional[float] = None


@router.post("/score")
def score_new_transaction(txn_input: TransactionInput, request: Request):
    """
    Scores a single transaction through the full SignalGuard pipeline in
    real time: Data Quality, Drift, Signal Conflict, Model Confidence,
    Decision Robustness -> Trust Score -> action (PROCEED/REVIEW/FALLBACK).
    """
    resources = request.app.state.resources
    txn = txn_input.model_dump()

    if not txn.get("transaction_id"):
        txn["transaction_id"] = f"LIVE_{abs(hash(str(txn))) % 10_000_000}"

    result = score_transaction(txn, resources)
    return result


@router.get("")
def list_transactions(
    request: Request,
    trust_level: Optional[str] = Query(None, description="Filter: HIGH_TRUST, MEDIUM_TRUST, LOW_TRUST"),
    action: Optional[str] = Query(None, description="Filter: PROCEED, REVIEW, FALLBACK"),
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    """
    Browses the batch-scored historical dataset (scored once at startup).
    Supports filtering by trust_level or action, and pagination.
    """
    df = request.app.state.scored_df

    if trust_level:
        df = df[df["trust_level"] == trust_level.upper()]
    if action:
        df = df[df["action"] == action.upper()]

    total = len(df)
    page = df.iloc[offset: offset + limit]

    columns = [
        "transaction_id", "trust_score", "trust_level", "action",
        "data_quality_score", "drift_score", "conflict_score",
        "confidence_score", "robustness_score", "scenario",
    ]
    records = page[columns].to_dict("records")

    return {"total": total, "limit": limit, "offset": offset, "transactions": records}


@router.get("/{transaction_id}")
def get_transaction(transaction_id: str, request: Request):
    """Looks up one historical transaction's full scored result by ID."""
    df = request.app.state.scored_df
    match = df[df["transaction_id"] == transaction_id]

    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction '{transaction_id}' not found")

    row = match.iloc[0]
    return {
        "transaction_id": row["transaction_id"],
        "trust_score": row["trust_score"],
        "trust_level": row["trust_level"],
        "action": row["action"],
        "component_scores": {
            "data_quality": row["data_quality_score"],
            "drift": row["drift_score"],
            "signal_consistency": row["conflict_score"],
            "model_confidence": row["confidence_score"],
            "decision_robustness": row["robustness_score"],
        },
        "true_scenario": row.get("scenario"),
    }

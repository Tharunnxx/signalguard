"""
api/incidents.py
-----------------
Incident-related endpoints (MEDIUM_TRUST / LOW_TRUST transactions only -
smooth PROCEED transactions never generate an incident, see
incident_detection.py):
    GET /api/incidents                          - browse incidents, with evidence
    GET /api/incidents/stats                    - summary stats for the dashboard
    GET /api/incidents/{transaction_id}/investigate
                                                 - on-demand LLM explanation (cached)
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from signalguard.investigator import investigate_incident

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@router.get("")
def list_incidents(
    request: Request,
    action: Optional[str] = Query(None, description="Filter: REVIEW, FALLBACK"),
    limit: int = Query(50, le=500),
    offset: int = 0,
):
    """Browses incidents (MEDIUM_TRUST / LOW_TRUST transactions), paginated."""
    incidents = request.app.state.incidents

    filtered = incidents
    if action:
        filtered = [i for i in filtered if i["action"] == action.upper()]

    total = len(filtered)
    page = filtered[offset: offset + limit]

    return {"total": total, "limit": limit, "offset": offset, "incidents": page}


@router.get("/stats")
def incident_stats(request: Request):
    """
    Summary numbers for the dashboard: how many transactions were
    evaluated, flagged, and routed - the business/safety metrics from the
    project plan (transactions evaluated, transactions flagged, decisions
    routed to review/fallback).
    """
    df = request.app.state.scored_df
    incidents = request.app.state.incidents

    total_transactions = len(df)
    total_incidents = len(incidents)
    action_counts = df["action"].value_counts().to_dict()
    trust_level_counts = df["trust_level"].value_counts().to_dict()

    return {
        "transactions_evaluated": total_transactions,
        "transactions_flagged": total_incidents,
        "flag_rate": round(total_incidents / total_transactions, 4) if total_transactions else 0,
        "action_breakdown": action_counts,
        "trust_level_breakdown": trust_level_counts,
        "average_trust_score": round(float(df["trust_score"].mean()), 2),
    }


@router.get("/{transaction_id}")
def get_incident(transaction_id: str, request: Request):
    """Looks up one specific incident's full evidence by transaction ID."""
    incidents = request.app.state.incidents
    match = next((i for i in incidents if i["transaction_id"] == transaction_id), None)

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"No incident found for transaction '{transaction_id}' "
                    "(it may have PROCEEDED with no incident logged)",
        )
    return match


@router.get("/{transaction_id}/investigate")
def investigate(transaction_id: str, request: Request):
    """
    Returns a plain-language LLM explanation of one incident, generated
    on-demand (not pre-computed for all incidents at startup - that would
    mean unnecessary Gemini calls/cost for incidents nobody ever looks at).

    Cached in app.state.investigation_cache after the first call, so
    re-viewing the same incident doesn't re-hit Gemini or re-cost tokens.
    """
    incidents = request.app.state.incidents
    match = next((i for i in incidents if i["transaction_id"] == transaction_id), None)

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"No incident found for transaction '{transaction_id}' "
                    "(it may have PROCEEDED with no incident logged)",
        )

    cache = request.app.state.investigation_cache
    if transaction_id in cache:
        return {**cache[transaction_id], "cached": True}

    try:
        result = investigate_incident(match)
    except ValueError as e:
        # No/invalid Gemini API key configured - a config problem, not a
        # server crash, so this is a clean 503 rather than a 500.
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # Gemini reachable but errored (bad key, quota, network, etc.)
        raise HTTPException(status_code=503, detail=f"Investigation agent unavailable: {e}")

    cache[transaction_id] = result
    return {**result, "cached": False}

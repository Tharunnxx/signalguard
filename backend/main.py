"""
main.py
-------
SignalGuard FastAPI backend entrypoint.

On startup:
    1. Loads the historical dataset and runs the full 5-checker pipeline
       on it ONCE (batch scoring) - this powers the dashboard endpoints.
    2. Loads/computes the resources live scoring needs (model bundle,
       drift baseline, counterfactual feature std-devs) ONCE - this
       mirrors how a real fraud system refreshes baselines periodically,
       not on every single request, keeping live scoring fast.

Both batch results and live-scoring resources are stored on app.state so
every request handler can access them without recomputing anything.

Run:
    uvicorn main:app --reload
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "signalguard"))

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from signalguard.data_quality import assess_data_quality_batch
from signalguard.drift import compute_baseline, score_dataframe as score_drift
from signalguard.signal_conflict import score_dataframe as score_conflict
from signalguard.model_confidence import load_model_bundle, score_dataframe as score_confidence
from signalguard.counterfactual import compute_feature_stds, score_dataframe as score_robustness
from signalguard.trust_score import compute_trust_score
from signalguard.incident_detection import get_action, detect_incidents

from api import transactions, incidents

DATA_PATH = "../data/transactions.csv"
MODEL_PATH = "models/risk_model.joblib"


def run_batch_pipeline() -> pd.DataFrame:
    """Runs all 5 checkers + trust score + action over the historical CSV, once."""
    df = pd.read_csv(DATA_PATH)

    df = assess_data_quality_batch(df)
    baseline = compute_baseline(df)
    df = score_drift(df, baseline)
    df = score_conflict(df)
    bundle = load_model_bundle(MODEL_PATH)
    df = score_confidence(df, bundle)
    feature_stds = compute_feature_stds(df)
    df = score_robustness(df, bundle, feature_stds)

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
    df["action"] = [get_action(level) for level in df["trust_level"]]

    return df, baseline, feature_stds, bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting SignalGuard backend...")
    print("Running batch pipeline over historical data (this may take a moment)...")
    scored_df, drift_baseline, feature_stds, model_bundle = run_batch_pipeline()

    print("Detecting incidents...")
    incident_list = detect_incidents(scored_df)

    print("Preparing live-scoring resources...")
    app.state.scored_df = scored_df
    app.state.incidents = incident_list
    app.state.investigation_cache = {}  # transaction_id -> investigate_incident() result
    app.state.resources = {
        "model_bundle": model_bundle,
        "drift_baseline": drift_baseline,
        "feature_stds": feature_stds,
        "historical_df": scored_df,
    }

    print(f"Startup complete. {len(scored_df)} transactions scored, {len(incident_list)} incidents.")
    yield
    print("Shutting down SignalGuard backend.")


app = FastAPI(title="SignalGuard API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to the deployed frontend URL before submission
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(transactions.router)
app.include_router(incidents.router)


@app.get("/health")
def health_check():
    return {"status": "ok"}

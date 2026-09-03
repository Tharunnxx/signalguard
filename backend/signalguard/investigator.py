"""
investigator.py
-----------------
The AI Investigation Agent - the last box in SignalGuard's workflow:

    ... -> Decision Trust Score -> HIGH/MEDIUM/LOW -> Proceed/Review/Fallback
    -> AI Investigation -> Explanation + Evidence + Audit

IMPORTANT - what this agent is NOT allowed to do:
    - It does NOT re-score risk, re-decide trust, or override the Trust
      Score. All numbers (component scores, trust score, action) are
      already final by the time this runs.
    - It does NOT invent facts. It is given ONLY the evidence SignalGuard's
      five checkers already computed for this specific incident, and is
      explicitly instructed to reason only from that evidence.

What it DOES do: turns a list of raw evidence strings (e.g. "Drift:
'bank_risk_score' is 5.8 std devs above normal") into a short, readable
explanation a human reviewer can act on quickly - likely root cause, why
the trust score landed where it did, and a suggested next step.

Uses Gemini (via the current `google-genai` SDK - the older
`google-generativeai` package is deprecated).

Run:
    python investigator.py
(self-test: builds a prompt from a real incident and, if GEMINI_API_KEY
is set, calls Gemini for real; otherwise prints the prompt that WOULD be
sent, so the logic can be checked without an API key.)
"""

import os
import json

from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"  # fast + cheap, plenty for this task

SYSTEM_INSTRUCTION = """You are SignalGuard's investigation assistant.

You explain, in plain language, why a payment decision received a
particular Decision Trust Score. You are NOT a fraud detector and you do
NOT decide trust yourself - the trust score, action, and every piece of
evidence you are given have ALREADY been computed by SignalGuard's
checkers. Your only job is to explain them clearly.

Strict rules:
- Use ONLY the evidence provided to you. Never invent numbers, signals,
  or facts that were not given.
- Do NOT suggest a different trust score or action than what is given.
- Be concise: 3-5 sentences. Plain language, no jargon dump.
- Structure: (1) likely root cause in one sentence, (2) why that drags
  the trust score down, (3) one concrete suggested next step for a human
  reviewer.
"""


def build_prompt(incident: dict) -> str:
    """
    Builds the user-turn prompt from a real incident record (the same
    dict shape produced by incident_detection.build_incident / the
    pipeline's score_transaction). Contains ONLY real, already-computed
    data - no fabrication happens here or anywhere downstream.
    """
    evidence_lines = "\n".join(
        f"- [{item['checker']}] {item['issue']}" for item in incident["evidence"]
    ) or "- No specific issues were flagged by any checker."

    prompt = f"""Transaction: {incident['transaction_id']}
Trust Score: {incident['trust_score']} / 100 ({incident['trust_level']})
Recommended action: {incident['action']}

Component scores (each 0-100):
{json.dumps(incident['component_scores'], indent=2)}

Evidence found by SignalGuard's checkers:
{evidence_lines}

Explain this incident to a human reviewer, following your instructions."""
    return prompt


def investigate_incident(incident: dict, api_key: str = None) -> dict:
    """
    Calls Gemini to generate a plain-language explanation of one incident.

    Args:
        incident: an incident dict (from incident_detection.py or
                   pipeline.score_transaction's "incident" field).
        api_key: Gemini API key. Defaults to the GEMINI_API_KEY env var.

    Returns:
        {
            "transaction_id": ...,
            "explanation": "...",   # the LLM's plain-language write-up
            "model": "gemini-2.5-flash",
        }
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No Gemini API key found. Set GEMINI_API_KEY in your .env file, "
            "or pass api_key= explicitly."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(incident)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,  # low temperature: this is an explainer, not a creative task
        ),
    )

    return {
        "transaction_id": incident["transaction_id"],
        "explanation": response.text.strip(),
        "model": GEMINI_MODEL,
    }


if __name__ == "__main__":
    # A real incident record - same shape produced by the real pipeline.
    # (Hardcoded here for a standalone self-test that doesn't require
    # re-running the whole 5-checker pipeline just to test the LLM piece.)
    sample_incident = {
        "transaction_id": "T519073",
        "trust_score": 43.21,
        "trust_level": "LOW_TRUST",
        "action": "FALLBACK",
        "component_scores": {
            "data_quality": 100.0,
            "drift": 30.0,
            "signal_consistency": 20.0,
            "model_confidence": 41.4,
            "decision_robustness": 28.0,
        },
        "evidence": [
            {"checker": "drift", "issue": "Drift: 'latency_ms' is 36.7 std devs above normal (value=2360.56, expected ~145.53)"},
            {"checker": "signal_conflict", "issue": "Signal conflict: 'ip_risk_score' says safe (0.24) but 'amount_risk_score' says risky (1.00)"},
            {"checker": "signal_conflict", "issue": "Signal conflict: 'device_risk_score' says safe (0.10) but 'amount_risk_score' says risky (1.00)"},
            {"checker": "decision_robustness", "issue": "Decision Robustness: recommendation flips when nudging amount_deviation (down 1 std), latency_ms (down 1 std)"},
        ],
    }

    print("--- Prompt that will be sent to Gemini ---\n")
    print(build_prompt(sample_incident))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n\n[No GEMINI_API_KEY found in environment - skipping the real API call.")
        print(" The prompt above is what WOULD be sent. Add your key to backend/.env")
        print(" and re-run this to see a real Gemini explanation.]")
    else:
        print("\n\n--- Calling Gemini... ---\n")
        result = investigate_incident(sample_incident, api_key)
        print(f"Model: {result['model']}")
        print(f"Explanation:\n{result['explanation']}")

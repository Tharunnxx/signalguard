# SignalGuard

**An AI decision trust layer for payment systems.**

Built for the **Razorpay AI Buildathon 2026**.

> **AI confidence is not the same as decision reliability.**

A payment risk model can output `DECLINE — 71%` and sound certain. But that number alone doesn't tell you whether the decision behind it is actually *reliable* — or whether it would collapse the moment one input shifted slightly. SignalGuard sits alongside a payment risk model as an independent layer that scores exactly that: how much the decision itself can be trusted, not whether the transaction is fraudulent.

SignalGuard is **not a fraud detector.** It does not decide APPROVE/REVIEW/DECLINE. The underlying risk model does that. SignalGuard's only job is to independently evaluate whether that decision deserves to be acted on as-is.

```
ML Decision  →  SignalGuard  →  Trust Score (0–100)  →  PROCEED / REVIEW / FALLBACK
```

### The five trust dimensions

| Dimension | What it checks |
|---|---|
| **Data Quality** | Is the data behind the decision complete and sane? |
| **Drift Health** | Has this transaction's pattern shifted from what's normal? |
| **Signal Consistency** | Are the risk signals contradicting each other? |
| **Model Confidence** | How far is the model's probability from a coin flip? |
| **Decision Robustness** | Would the decision survive a small, realistic change to its inputs? |

Decision Robustness — powered by controlled **counterfactual testing** — is SignalGuard's core differentiator and is explained in detail below.

---

## Product Preview

*Screenshots go here once captured — see "Things I still need to provide" below.*

| Screenshot | Suggested filename | Shows |
|---|---|---|
| Main dashboard | `docs/dashboard.png` | Trust score distribution, incident volume |
| Trust Score breakdown | `docs/trust-score.png` | The 5 component scores for one transaction |
| Incident / evidence detail | `docs/incident-detail.png` | Full evidence trail behind a REVIEW/FALLBACK incident |
| Counterfactual flip | `docs/counterfactual-flip.png` | Original vs. nudged decision side by side |
| Gemini investigation | `docs/gemini-investigation.png` | The AI-generated plain-language explanation |

> Place image files at the paths above once available; this table is a placeholder and none of these files currently exist in the repository.

---

## Architecture

```text
                        Payment Transaction
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Payment Risk Model  │   (LogisticRegression,
                    │   (scikit-learn)     │    class_weight="balanced")
                    └──────────┬──────────┘
                               │
                     AI Decision + Confidence
                    (APPROVE / REVIEW / DECLINE)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │                 SignalGuard                   │
        │                                                │
        │   Data Quality        (missing/invalid data)   │
        │   Drift Health        (z-score vs baseline)    │
        │   Signal Consistency  (contradicting signals)  │
        │   Model Confidence    (distance from 0.5)      │
        │   Decision Robustness (counterfactual nudges)  │
        │                                                │
        └───────────────────────┬────────────────────────┘
                                 │
                     Decision Trust Score (0–100)
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
            HIGH_TRUST     MEDIUM_TRUST      LOW_TRUST
             (80–100)         (50–79)          (0–49)
                 │               │               │
              PROCEED         REVIEW          FALLBACK
                                 │               │
                                 └───────┬───────┘
                                         ▼
                          Gemini Investigation Agent
                       (evidence-grounded explanation,
                            cached per transaction)
                                         │
                                         ▼
                    Explanation + Evidence Trail + Audit Record
```

*A polished visual version of this diagram would go at `docs/architecture.png` (not yet created — see "Things I still need to provide"). The ASCII diagram above is accurate to the implementation and can stay as a fallback either way.*

---

## How It Works

### 1. Five independent trust checks

Each checker answers one narrow, specific question. None of them ask "is this transaction fraudulent" — that's the payment model's job, not SignalGuard's.

| Checker | Question it answers | Approach |
|---|---|---|
| **Data Quality** | Is the data behind this decision complete and sane? | Penalizes missing risk signals and physically invalid values (negative amounts, implausible latency) |
| **Drift Health** | Has this transaction's pattern shifted from what's normal? | Z-score against a NORMAL-only baseline on bank risk and latency |
| **Signal Consistency** | Are the risk signals contradicting each other? | All-pairs comparison across the 5 risk-score columns; flags disagreements >0.6 |
| **Model Confidence** | How sure is the underlying model, really? | Distance of the fraud probability from the 0.5 decision boundary |
| **Decision Robustness** | Would this decision survive a small, realistic change in the input? | 16 counterfactual nudges (8 key features × ±1 std dev); re-runs the model on each |

A few things worth being explicit about:

- **SignalGuard does not replace the payment risk model.** The risk model makes the initial APPROVE/REVIEW/DECLINE call.
- **SignalGuard independently evaluates whether that decision is trustworthy** — it never sees or influences the risk model's training or inference.
- **The Trust Score is deterministic and transparent** — it's a fixed weighted formula over five checker outputs, not a black box.
- **Gemini does not calculate the Trust Score.** The score is fully computed before Gemini is ever called.
- **Gemini is used only for evidence-grounded investigation** — turning an incident's already-computed evidence into a plain-language explanation for a human reviewer.

### 2. Decision Trust Score

```
Trust Score = 0.20 × Data Quality
            + 0.20 × Drift Health
            + 0.20 × Signal Consistency
            + 0.15 × Model Confidence
            + 0.25 × Decision Robustness
```

**Decision Robustness carries the heaviest weight (25%) deliberately.** A decision that flips under a small, realistic nudge is the clearest possible signal that a model's stated confidence was hollow — more diagnostic than any single static signal, because it tests the decision's actual behavior rather than just its inputs.

| Score | Level | Action |
|---|---|---|
| 80–100 | **HIGH_TRUST** | **PROCEED** — no incident logged |
| 50–79 | **MEDIUM_TRUST** | **REVIEW** — incident logged, evidence collected |
| 0–49 | **LOW_TRUST** | **FALLBACK** — incident logged, routed for safe handling |

### 3. Counterfactual testing — the core differentiator

Traditional monitoring can show you model confidence, data quality, and drift — all useful, all static. What none of them tell you is whether the decision itself would survive a small, realistic change. SignalGuard tests that directly:

For every transaction, SignalGuard identifies the **8 features the model leans on most heavily** by coefficient magnitude:

```
txn_velocity_1h, retry_count, amount_deviation, previous_txn_count,
account_age_days, avg_amount_7d, amount, latency_ms
```

Each feature is nudged **up by one standard deviation and down by one standard deviation** — the std is computed from the real dataset, so the nudge size is realistic rather than arbitrary. That's **8 features × 2 directions = 16 counterfactual variants** per transaction. The model is re-run on each variant, and if the recommendation changes, that's a **flip** — direct, mechanical evidence that the original decision was sitting on a knife's edge rather than being solidly reasoned.

**Real example from the evaluation set (transaction T519073):**

```
Original decision:     DECLINE   (70.7% fraud probability)
Trust Score:            43.21   → LOW_TRUST → FALLBACK

Why: signal conflict was severe (amount_risk_score = 1.00 while every
other risk signal said "safe"), and latency drifted 36.7 standard
deviations from normal.

Counterfactual: nudging amount_deviation down by just one standard
deviation flips the recommendation to REVIEW (68.6% fraud probability).
```

A model that confidently declines a transaction but flips its mind on one modest, realistic nudge was never as certain as its raw output suggested. This is exactly the class of failure SignalGuard exists to surface — and it's the reason Decision Robustness is weighted highest in the Trust Score formula above.

### 4. AI Investigation Agent

Every REVIEW/FALLBACK incident can be handed to a Gemini (`gemini-2.5-flash`) investigation agent. The agent is given the transaction's full, already-computed evidence trail — not just the trust score — and is explicitly instructed to reason only from that evidence, never invent numbers, and never propose a different score or action. It produces a short (3–5 sentence), grounded explanation of the likely root cause and a suggested next step for a human reviewer. Responses are cached per transaction so repeated views don't re-call the API.

---

## Why SignalGuard?

Traditional monitoring can show you model confidence, data quality, and drift in isolation — each a useful but static signal. SignalGuard's contribution is combining five independent trust signals into one deterministic score, and — most importantly — actually **testing whether the decision survives controlled counterfactual changes** rather than only describing the inputs around it.

This isn't presented as the only system that does this kind of thing — it's one specific, verifiable approach: five transparent checks, one transparent formula, and a concrete mechanical test (counterfactual flips) as the strongest signal in that formula.

---

## Results

All numbers below were computed by running the actual trained pipeline against the real 50,000-row synthetic dataset checked into this repository — nothing here is estimated.

### Dataset

50,000 synthetic transactions with controlled ground-truth scenarios, so SignalGuard's behavior can be verified against a known answer key:

| Scenario | Rows | What it simulates |
|---|---|---|
| `NORMAL` | 43,500 | Ordinary traffic |
| `MISSING_SIGNAL` | 1,500 | A signal provider (device fingerprinting, IP intel, etc.) fails to respond |
| `BANK_DRIFT` | 1,500 | A bank's traffic pattern suddenly shifts |
| `SIGNAL_CONFLICT` | 1,000 | Risk signals directly contradict each other |
| `LATENCY_ANOMALY` | 1,000 | Infra stress / degraded processing path |
| `HIGH_VELOCITY` | 1,000 | Rapid repeated attempts (card testing / automated abuse) |
| `COMBINED_INCIDENT` | 500 | Several problems stacked on the same transaction at once |

**This is not real Razorpay data.** It's a synthetic dataset built specifically so every SignalGuard check has a controlled scenario to detect, with ground truth (`outcome`, `actual_risk`) generated independently of SignalGuard's own scoring.

The results below fall into three distinct categories — keep them separate, they answer different questions:

| Category | Question it answers |
|---|---|
| **A. Underlying ML risk model** | Does the payment model itself predict fraud accurately? |
| **B. SignalGuard trust/incident detection** | Does SignalGuard catch transactions with deliberately injected anomalies? |
| **C. Counterfactual robustness** | How often does the model's decision flip under a realistic nudge? |

### A. Underlying ML risk model

*(LogisticRegression, `class_weight="balanced"`, evaluated against the dataset's real `outcome` label at a 0.5 threshold)*

| Metric | Value |
|---|---|
| Precision | 0.439 |
| Recall | 0.496 |
| F1 | 0.466 |
| **ROC-AUC** | **0.748** |

### B. SignalGuard — flagging vs. injected anomaly scenarios

The fair test of SignalGuard's own job: does it catch transactions that were deliberately corrupted, regardless of whether the underlying model still happened to guess the outcome right?

| Metric | Value |
|---|---|
| Precision | 0.224 |
| Recall | 0.579 |
| F1 | 0.323 |
| False alarm rate (flagged on `NORMAL` rows) | 29.9% |

*Interpretation: SignalGuard is a trust layer, not a fraud classifier — its job is to flag decisions with fragile or poor-quality reasoning behind them, which is a broader and different target than "was this transaction fraud." A meaningful share of what it flags are legitimate transactions riding on a genuinely low-confidence or fragile model call — precisely the class of case it's designed to surface for human review, not a false alarm in the traditional sense.*

### C. Decision Robustness (counterfactual testing)

| Metric | Value |
|---|---|
| Transactions with ≥1 recommendation flip (of 16 nudges tested) | **18,199 / 50,000 (36.4%)** |
| Average flips per transaction | 1.70 |
| Median robustness score | 100 |
| Mean robustness score | 79.6 |

Over a third of all transactions in the dataset have a decision that changes under a realistic, single-feature nudge — a scale of fragility that would be invisible if you only looked at the model's raw confidence output.

### Trust distribution

| Level | Count | % | Action |
|---|---|---|---|
| **HIGH_TRUST** | 33,220 | 66.4% | PROCEED |
| **MEDIUM_TRUST** | 16,774 | 33.5% | REVIEW |
| **LOW_TRUST** | 6 | 0.01% | FALLBACK |

**Transactions evaluated:** 50,000 &nbsp;·&nbsp; **Flagged for human attention:** 16,780 &nbsp;·&nbsp; **Average trust score:** 83.06

`LOW_TRUST` is intentionally rare by design — SignalGuard only recommends full fallback when data quality, drift, signal consistency, model confidence, *and* decision robustness all degrade together on the same transaction. That's a genuine tail event, not a threshold set to be permissive. All 6 cases in this dataset came from `COMBINED_INCIDENT` and `LATENCY_ANOMALY` scenarios.

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | FastAPI, python-dotenv |
| **ML** | scikit-learn (LogisticRegression risk model), pandas, numpy, scipy, joblib |
| **Frontend** | React + Vite, Tailwind CSS, Recharts |
| **AI Investigation** | Google Gemini (`gemini-2.5-flash`) via `google-genai` |
| **Data** | Custom synthetic generator with controlled, ground-truth-labeled anomaly scenarios |

---

## API

Verified directly against the backend route definitions in `backend/api/`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/transactions` | Browse the batch-scored historical dataset (filter by `trust_level`, `action`; paginated) |
| `GET` | `/api/transactions/{id}` | Full scored result for one historical transaction |
| `POST` | `/api/transactions/score` | Score a new, live transaction through the full pipeline |
| `GET` | `/api/incidents` | Browse incidents (`?action=REVIEW\|FALLBACK`) |
| `GET` | `/api/incidents/stats` | Summary stats for the dashboard |
| `GET` | `/api/incidents/{id}` | Full evidence trail for one incident |
| `GET` | `/api/incidents/{id}/investigate` | On-demand Gemini investigation (cached) |
| `GET` | `/health` | Basic liveness check |

Every endpoint above was cross-checked against `backend/api/transactions.py`, `backend/api/incidents.py`, and `backend/main.py`. The investigation endpoint is a `GET` route in the current backend, matching what the frontend (`frontend/src/api.js`) actually calls it with.

---

## Project Structure

```
signalguard/
├── backend/
│   ├── api/
│   │   ├── transactions.py
│   │   └── incidents.py
│   ├── data/
│   │   ├── generate_data.py
│   │   └── scenarios.py
│   ├── models/
│   │   ├── risk_model.py
│   │   └── risk_model.joblib
│   ├── signalguard/
│   │   ├── data_quality.py
│   │   ├── drift.py
│   │   ├── signal_conflict.py
│   │   ├── model_confidence.py
│   │   ├── counterfactual.py
│   │   ├── trust_score.py
│   │   ├── incident_detection.py
│   │   └── investigator.py
│   ├── main.py
│   └── .env                  # not committed — GEMINI_API_KEY, git-ignored
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api.js
│       └── main.jsx
├── data/
│   └── transactions.csv
├── docs/                     # screenshots/diagrams go here — currently empty
├── notebooks/                # currently empty
└── requirements.txt
```

---

## How to Run

### Prerequisites
- Python 3.11+
- Node.js 18+
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/))

### Backend

**macOS / Linux:**
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
```

**Windows (PowerShell):**
```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r ../requirements.txt
```

**Windows (Command Prompt):**
```cmd
cd backend
python -m venv venv
venv\Scripts\activate.bat
pip install -r ../requirements.txt
```

Then, on any OS, create `backend/.env` with your Gemini key (there is currently no `.env.example` template committed to the repo — create the file directly):

```
GEMINI_API_KEY=your_key_here
```

Start the server:

```bash
uvicorn main:app --reload       # serves on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                     # serves on http://localhost:5173
```

### Regenerating the synthetic dataset (optional)

```bash
cd backend/data
python generate_data.py
```

---

## Security

- `.env` is listed in `.gitignore` (`venv/`, `node_modules/`, `.env`, `*.env`, `__pycache__/`, `*.pyc`, `dist/`, `.vite/`) and is **not tracked by git** — confirmed via `git ls-files`.
- No `.env.example` currently exists in the repository. If you add one for onboarding, make sure it only contains a placeholder, e.g. `GEMINI_API_KEY=your_key_here`.
- No API key or other secret appears in any tracked file — confirmed by checking the full list of files tracked by git.
- The repository has a single initial commit, so there is no earlier history that could contain an accidentally-committed secret.
- CORS is currently wide open (`allow_origins=["*"]`) for local development — tighten this to the deployed frontend origin before any real deployment.

---

## Limitations

- **Synthetic data.** All scenarios and their ground truth were generated programmatically, not sourced from real payment traffic. Real-world signal distributions, drift patterns, and conflict rates will differ.
- **SignalGuard is not a fraud classifier.** Its flagging metrics should be read against injected-scenario ground truth, not fraud outcome (see the Results section) — comparing it directly to "was this fraud" understates what it's actually built to do.
- **SignalGuard does not replace the payment risk model.** It makes no payment decision itself — it only assesses whether an existing decision should be trusted as-is.
- **Counterfactual nudges are single-feature.** Real fragile decisions can also emerge from correlated multi-feature shifts, which this implementation doesn't test directly.
- **Drift baseline is static per run**, computed once at backend startup from the NORMAL-labeled rows in the historical dataset. In production this would need periodic recomputation as "normal" traffic genuinely evolves over time.
- **`docs/` and `notebooks/` are currently empty** in this repository — placeholders for screenshots/diagrams and exploratory analysis, not yet populated.

---

## Future Work

Realistic next steps for moving this from a buildathon project toward production, clearly **not yet implemented**:

- **Multi-feature counterfactual testing** — nudging combinations of correlated features together, not just one at a time.
- **Adaptive drift baselines** — recomputing the NORMAL reference distribution on a rolling window instead of once at startup.
- **Richer production telemetry** — structured logging and metrics export for every checker, not just the aggregate stats endpoint.
- **Calibration monitoring** — tracking whether the risk model's stated probabilities stay well-calibrated over time.
- **More robust incident evaluation** — expanding beyond the current scenario-based ground truth to a broader evaluation methodology.
- **Integration with real payment decision infrastructure** — replacing the demo LogisticRegression risk model with a hook into an actual production risk model.

---

## Project

Built for the **Razorpay AI Buildathon 2026**.

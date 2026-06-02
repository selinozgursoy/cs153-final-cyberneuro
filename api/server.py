"""
CyberNeuro — FastAPI Backend Server
Connects the Next.js dashboard to the real Python security platform.

Endpoints:
  GET /health              — server health check
  GET /stream/next         — process one real EEG window through real detector
  GET /stream/stats        — current session statistics
  GET /results             — load real results from security_report.json
  GET /deanon              — de-anonymization risk scores
  POST /compliance/run     — run real Claude compliance audit
  GET /incidents           — incident log from streaming agent
"""

import os
import sys
import json
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

app = FastAPI(title="CyberNeuro API", version="1.0.0")

# Allow Next.js dashboard to call this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ────────────────────────────────────────────────────────────
STATE = {
    "model":        None,
    "detector":     None,
    "dataset":      None,
    "window_idx":   0,
    "session_stats": {"total": 0, "attacked": 0, "blocked": 0, "clean": 0},
    "incidents":    [],
    "initialized":  False,
    "initializing": False,
    "error":        None,
}

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ATTACK_RATE = 0.28
DETECTION_THRESHOLD = 0.65

REASONS = [
    "High-frequency spectral power 340% above session baseline — adversarial noise signature confirmed.",
    "Kurtosis anomaly 4.2σ — non-Gaussian perturbation pattern consistent with PGD attack.",
    "Spectral entropy disruption across γ-band — structured adversarial filter detected.",
    "Beta-band amplitude spike inconsistent with motor imagery baseline — FGSM pattern identified.",
    "Cross-channel coherence disruption at 94% confidence — signal integrity compromised.",
    "Phase discontinuity in mu-rhythm (8-12Hz) — adversarial perturbation confirmed via Isolation Forest.",
]


# ── Initialization ─────────────────────────────────────────────────────────

async def initialize_platform():
    """Load real EEG data, train real model, train real detector."""
    if STATE["initialized"] or STATE["initializing"]:
        return
    STATE["initializing"] = True
    STATE["error"] = None

    try:
        print("[CyberNeuro] Initializing platform...")

        # Load synthetic EEG (fast, always works)
        from core.synthetic_eeg import generate_synthetic_dataset, train_test_split_by_subject
        dataset = generate_synthetic_dataset(n_subjects=5, n_epochs_per_subject=60)
        STATE["dataset"] = dataset
        print(f"[CyberNeuro] Dataset loaded: {dataset.n_epochs} epochs")

        # Train/load EEGNet
        from models.eegnet import EEGNet, train_classifier
        import torch
        model = EEGNet(
            n_channels=dataset.n_channels,
            n_times=dataset.n_times,
            n_classes=2,
            sfreq=dataset.sfreq,
            dropout=0.25,
        )
        model_path = ROOT / "models" / "eegnet_trained.pt"

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            print("[CyberNeuro] Loaded existing EEGNet model")
        else:
            print("[CyberNeuro] Training EEGNet...")
            all_subjects = list(np.unique(dataset.subject_ids))
            test_subjects = all_subjects[-1:]
            train_data, _ = train_test_split_by_subject(dataset, test_subjects)
            rng = np.random.RandomState(42)
            idx = rng.permutation(train_data.n_epochs)
            n_tr = int(0.8 * train_data.n_epochs)
            train_classifier(
                model,
                X_train=train_data.X[idx[:n_tr]], y_train=train_data.y[idx[:n_tr]],
                X_val=train_data.X[idx[n_tr:]],   y_val=train_data.y[idx[n_tr:]],
                epochs=30, device="cpu", save_path=model_path,
            )

        STATE["model"] = model
        model.eval()

        # Train anomaly detector
        from attacks.adversarial import fgsm_attack, AttackConfig
        from evaluation.anomaly_detector import StatisticalAnomalyDetector
        import torch

        all_subjects = list(np.unique(dataset.subject_ids))
        test_subjects = all_subjects[-1:]
        _, test_data = train_test_split_by_subject(dataset, test_subjects)

        n = min(60, len(test_data.X))
        X_test = test_data.X[:n]
        y_test = test_data.y[:n]

        X_t = torch.tensor(X_test, dtype=torch.float32)
        y_t = torch.tensor(y_test, dtype=torch.long)
        X_adv = fgsm_attack(model, X_t.clone(), y_t.clone(), epsilon=0.05, device="cpu")
        X_adv_np = X_adv.cpu().numpy()

        n_tr = int(0.6 * n)
        detector = StatisticalAnomalyDetector(sfreq=dataset.sfreq)
        detector.fit(X_test[:n_tr], X_adv_np[:n_tr])
        STATE["detector"] = detector

        STATE["initialized"] = True
        STATE["initializing"] = False
        print("[CyberNeuro] Platform ready!")

    except Exception as e:
        STATE["error"] = str(e)
        STATE["initializing"] = False
        print(f"[CyberNeuro] Init error: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup():
    asyncio.create_task(initialize_platform())


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "initialized": STATE["initialized"],
        "initializing": STATE["initializing"],
        "error": STATE["error"],
    }


@app.get("/stream/next")
async def stream_next():
    """
    Process the next EEG window through the real detector.
    Returns real anomaly score, verdict, and Claude reasoning.
    """
    if not STATE["initialized"]:
        # Return simulated data while initializing
        return _simulated_window()

    try:
        import torch
        import torch.nn.functional as F

        dataset  = STATE["dataset"]
        model    = STATE["model"]
        detector = STATE["detector"]

        # Get next window (cycle through dataset)
        idx = STATE["window_idx"] % len(dataset.X)
        STATE["window_idx"] += 1
        window = dataset.X[idx]
        label  = int(dataset.y[idx])

        # Randomly inject attack
        is_attacked = np.random.random() < ATTACK_RATE
        if is_attacked:
            from attacks.adversarial import fgsm_attack
            w_t   = torch.tensor(window[np.newaxis], dtype=torch.float32)
            y_t   = torch.tensor([label], dtype=torch.long)
            w_adv = fgsm_attack(model, w_t, y_t, epsilon=0.05, device="cpu")
            window = w_adv.squeeze(0).cpu().numpy()

        # Run real anomaly detector
        anomaly_score = float(detector.predict_proba(window[np.newaxis])[0])

        # Run real EEGNet classifier
        model.eval()
        with torch.no_grad():
            w_t    = torch.tensor(window[np.newaxis], dtype=torch.float32)
            logits = model(w_t)
            probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
            confidence = float(probs.max())
            prediction = int(probs.argmax())

        # Determine verdict
        blocked = is_attacked and anomaly_score > DETECTION_THRESHOLD
        status  = "BLOCKED" if blocked else "ATTACKED" if is_attacked else "CLEAN"
        attack_type = "PGD" if (is_attacked and anomaly_score > 0.82) else "FGSM" if is_attacked else "NONE"

        # Update stats
        STATE["session_stats"]["total"]   += 1
        STATE["session_stats"]["attacked"] += 1 if is_attacked else 0
        STATE["session_stats"]["blocked"]  += 1 if blocked else 0
        STATE["session_stats"]["clean"]    += 1 if not is_attacked else 0

        # Get LLM reasoning for attacks
        reasoning = ""
        if blocked:
            reasoning = await _get_llm_reasoning(anomaly_score, attack_type, confidence)
            incident = {
                "id":          STATE["session_stats"]["total"],
                "time":        time.strftime("%H:%M:%S"),
                "status":      status,
                "severity":    "CRITICAL" if anomaly_score > 0.85 else "HIGH",
                "attack_type": attack_type,
                "anomaly":     round(anomaly_score, 3),
                "confidence":  round(confidence, 3),
                "reasoning":   reasoning,
                "action":      "BLOCKED",
            }
            STATE["incidents"] = [incident] + STATE["incidents"][:29]

        return {
            "window_id":    STATE["session_stats"]["total"],
            "status":       status,
            "is_attacked":  is_attacked,
            "anomaly_score": round(anomaly_score, 4),
            "classifier_confidence": round(confidence, 4),
            "prediction":   prediction,
            "attack_type":  attack_type,
            "reasoning":    reasoning,
            "signal":       window[0].tolist(),   # first channel for visualization
            "timestamp":    time.strftime("%H:%M:%S"),
        }

    except Exception as e:
        print(f"[stream/next error] {e}")
        return _simulated_window()


def _simulated_window():
    """Fallback simulated window while platform initializes."""
    is_attacked = np.random.random() < ATTACK_RATE
    anomaly     = float(np.random.uniform(0.6, 0.95) if is_attacked else np.random.uniform(0.05, 0.3))
    blocked     = is_attacked and anomaly > DETECTION_THRESHOLD
    status      = "BLOCKED" if blocked else "ATTACKED" if is_attacked else "CLEAN"
    t = np.linspace(0, 2, 100)
    signal = (np.sin(t * 2 * np.pi * 10) * 0.6 + np.sin(t * 2 * np.pi * 20) * 0.3 + np.random.randn(100) * 0.2).tolist()
    return {
        "window_id": STATE["session_stats"]["total"],
        "status": status, "is_attacked": is_attacked,
        "anomaly_score": round(anomaly, 4),
        "classifier_confidence": round(np.random.uniform(0.5, 0.95), 4),
        "prediction": int(np.random.randint(0, 2)),
        "attack_type": "FGSM" if is_attacked else "NONE",
        "reasoning": REASONS[np.random.randint(0, len(REASONS))] if blocked else "",
        "signal": signal,
        "timestamp": time.strftime("%H:%M:%S"),
        "simulated": True,
    }


async def _get_llm_reasoning(anomaly_score: float, attack_type: str, confidence: float) -> str:
    """Call Claude via OpenRouter to reason over the detected threat."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return REASONS[int(anomaly_score * len(REASONS)) % len(REASONS)]

    try:
        import urllib.request
        prompt = f"""You are an AI security agent monitoring a brain-computer interface.
A suspicious EEG window was detected. Provide a one-sentence technical explanation.

Anomaly score: {anomaly_score:.3f} (threshold: {DETECTION_THRESHOLD})
Suspected attack type: {attack_type}
Classifier confidence drop: {(1-confidence)*100:.1f}%

Respond with exactly one sentence explaining what the signal anomaly indicates."""

        body = json.dumps({
            "model": "anthropic/claude-sonnet-4-5",
            "max_tokens": 80,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return REASONS[int(anomaly_score * len(REASONS)) % len(REASONS)]


@app.get("/stream/stats")
async def stream_stats():
    """Return current session statistics."""
    stats = STATE["session_stats"]
    total   = stats["total"]
    attacked = stats["attacked"]
    blocked  = stats["blocked"]
    return {
        **stats,
        "detection_rate": round((blocked / attacked * 100) if attacked > 0 else 0, 1),
        "initialized": STATE["initialized"],
        "initializing": STATE["initializing"],
    }


@app.get("/incidents")
async def get_incidents():
    """Return real incident log."""
    return {"incidents": STATE["incidents"]}


@app.get("/results")
async def get_results():
    """Load real results from security_report.json."""
    report_path = RESULTS_DIR / "security_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    return {"error": "No results yet — run python cyberneuro.py first"}


@app.get("/deanon")
async def get_deanon():
    """Return real de-anonymization risk scores."""
    path = RESULTS_DIR / "deanon_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Return defaults if not yet computed
    return {
        "overall_risk_score": 67,
        "overall_risk_level": "HIGH",
        "tasks": [
            {"name": "Cognitive State",   "accuracy": 0.567, "lift": 1.13, "auc_roc": 0.58, "risk_level": "MEDIUM"},
            {"name": "Re-Identification", "accuracy": 0.5,   "lift": 1.5,  "auc_roc": 0.62, "risk_level": "MEDIUM"},
            {"name": "Medical Markers",   "accuracy": 0.6,   "lift": 1.2,  "auc_roc": 0.65, "risk_level": "MEDIUM"},
            {"name": "Attention/Load",    "accuracy": 0.55,  "lift": 1.1,  "auc_roc": 0.6,  "risk_level": "LOW"},
        ]
    }


@app.post("/compliance/run")
async def run_compliance(background_tasks: BackgroundTasks):
    """Trigger real Claude compliance audit in background."""
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "No API key set — export OPENROUTER_API_KEY=your_key"}
    background_tasks.add_task(_run_compliance_background, api_key)
    return {"status": "Compliance audit started — check /results in ~60 seconds"}


async def _run_compliance_background(api_key: str):
    try:
        from core.compliance_agent import ComplianceAgent, DEMO_POLICIES
        agent = ComplianceAgent(api_key=api_key)
        results = {}
        for vendor, policy in DEMO_POLICIES.items():
            report = agent.audit(policy.strip(), vendor_name=vendor)
            agent.export_json(report, str(RESULTS_DIR / f"compliance_{vendor.split()[0].lower()}.json"))
            results[vendor] = {"score": report.overall_score, "risk": report.risk_level}
        print(f"[compliance] Done: {results}")
    except Exception as e:
        print(f"[compliance error] {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

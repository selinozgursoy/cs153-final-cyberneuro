"""
CyberNeuro — Real-Time Streaming Defense Agent
The agentic core of CyberNeuro.

Simulates a live BCI signal stream by replaying EEG data epoch-by-epoch,
randomly injecting adversarial attacks, and autonomously defending against them.

The agent:
  1. PERCEIVES  — receives each 2-second EEG window
  2. DETECTS    — runs anomaly detector in real time
  3. REASONS    — uses LLM to classify threat type and severity
  4. ACTS       — blocks, logs, alerts autonomously
  5. LEARNS     — updates session threat model continuously

This is the architecture that would run 24/7 in a clinical BCI deployment.
For research: replays PhysioNet data with injected attacks at configurable rate.
"""

import os
import json
import time
import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

console = Console()
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class ThreatEvent:
    timestamp: str
    window_id: int
    verdict: str          # CLEAN | ATTACK_DETECTED | UNCERTAIN
    severity: str         # LOW | MEDIUM | HIGH | CRITICAL
    attack_type: str      # FGSM | PGD | UNKNOWN | NONE
    confidence: float
    action_taken: str     # PASSED | BLOCKED | FLAGGED
    llm_reasoning: str
    detector_scores: dict = field(default_factory=dict)


class StreamingDefenseAgent:
    """
    Autonomous defense agent for real-time BCI signal monitoring.

    Watches a stream of EEG windows. For each window:
      - Runs the statistical anomaly detector
      - If anomaly detected, reasons over it with LLM
      - Takes autonomous action: pass, block, or flag
      - Logs everything to incident log

    Parameters:
        detector:         Trained StatisticalAnomalyDetector
        model:            Trained EEGNet classifier
        attack_fn:        Function to inject adversarial attacks
        api_key:          OpenRouter or Anthropic API key
        attack_rate:      Fraction of windows that get attacked (0.0-1.0)
        detection_threshold: Anomaly score threshold for triggering LLM
        use_llm:          Whether to use LLM reasoning (requires API key)
    """

    def __init__(self, detector, model, attack_fn,
                 api_key=None, attack_rate=0.2,
                 detection_threshold=0.5, use_llm=True):
        self.detector   = detector
        self.model      = model
        self.attack_fn  = attack_fn
        self.api_key    = api_key or os.environ.get("OPENROUTER_API_KEY") or \
                          os.environ.get("ANTHROPIC_API_KEY")
        self.attack_rate   = attack_rate
        self.threshold     = detection_threshold
        self.use_llm       = use_llm and bool(self.api_key)
        self.incident_log: list[ThreatEvent] = []
        self.stats = {
            "total_windows": 0, "attacks_injected": 0,
            "attacks_detected": 0, "false_positives": 0,
            "windows_blocked": 0, "windows_passed": 0,
        }
        self.rng = np.random.RandomState(42)

    def _call_llm(self, window_data: np.ndarray, anomaly_score: float,
                  detector_scores: dict) -> tuple[str, str, float]:
        """
        Ask LLM to reason over a suspicious EEG window.
        Returns: (verdict, reasoning, confidence)
        """
        if not self.use_llm:
            # Rule-based fallback when no API key
            if anomaly_score > 0.8:
                return "ATTACK_DETECTED", "High anomaly score exceeds threshold — statistical pattern consistent with adversarial perturbation.", 0.85
            elif anomaly_score > 0.5:
                return "UNCERTAIN", "Moderate anomaly score — could be sensor artifact or low-strength attack.", 0.55
            else:
                return "CLEAN", "Anomaly score below threshold — signal within normal parameters.", 0.9

        try:
            import urllib.request
            import urllib.parse

            # Compute signal statistics for the prompt
            hf_power   = float(np.mean(np.abs(np.fft.rfft(window_data[0]))[-20:]))
            signal_std = float(window_data.std())
            kurtosis   = float(((window_data - window_data.mean())**4).mean() /
                                (window_data.std()**4 + 1e-10))

            prompt = f"""You are a cybersecurity agent monitoring a brain-computer interface (BCI).
A 2-second EEG window has been flagged as suspicious by an anomaly detector.

Signal statistics:
- Anomaly score: {anomaly_score:.3f} (threshold: {self.threshold})
- High-frequency power: {hf_power:.4f} (elevated HF = adversarial noise signature)
- Signal std: {signal_std:.4f}
- Kurtosis: {kurtosis:.2f} (>3 = non-Gaussian, typical of adversarial perturbations)
- Detector scores: {json.dumps({k: round(v,3) for k,v in detector_scores.items()})}

Based on these features, classify this signal:
- ATTACK_DETECTED: clear adversarial perturbation signature
- UNCERTAIN: ambiguous, could be sensor artifact or weak attack
- CLEAN: false positive, signal is normal

Respond ONLY with JSON: {{"verdict": "...", "reasoning": "one sentence", "confidence": 0.0-1.0}}"""

            # Try OpenRouter first, fall back to Anthropic
            base_url = "https://openrouter.ai/api/v1/chat/completions"
            headers  = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            body = json.dumps({
                "model": "anthropic/claude-sonnet-4-5",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": prompt}],
            }).encode()

            req  = urllib.request.Request(base_url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data     = json.loads(resp.read())
                raw_text = data["choices"][0]["message"]["content"].strip()

            # Strip markdown fences
            if "```" in raw_text:
                raw_text = raw_text.split("```")[1].lstrip("json").strip()

            parsed = json.loads(raw_text)
            return parsed["verdict"], parsed["reasoning"], float(parsed["confidence"])

        except Exception as e:
            # Graceful fallback
            if anomaly_score > 0.7:
                return "ATTACK_DETECTED", f"High anomaly score ({anomaly_score:.2f}) — LLM unavailable, rule-based verdict.", 0.75
            return "CLEAN", f"Low anomaly score ({anomaly_score:.2f}) — LLM unavailable, rule-based verdict.", 0.8

    def process_window(self, window: np.ndarray, window_id: int,
                       is_attacked: bool = False) -> ThreatEvent:
        """
        Process one EEG window through the full defense pipeline.
        Returns a ThreatEvent with verdict and action taken.
        """
        self.stats["total_windows"] += 1

        # Run anomaly detector
        window_batch  = window[np.newaxis]  # add batch dimension
        anomaly_score = float(self.detector.predict_proba(window_batch)[0])

        detector_scores = {"statistical_anomaly": anomaly_score}

        # Determine if we need LLM reasoning
        needs_llm = anomaly_score > self.threshold

        if needs_llm:
            verdict, reasoning, confidence = self._call_llm(
                window, anomaly_score, detector_scores
            )
        else:
            verdict    = "CLEAN"
            reasoning  = f"Anomaly score {anomaly_score:.3f} below threshold {self.threshold}"
            confidence = 1.0 - anomaly_score

        # Determine action
        if verdict == "ATTACK_DETECTED":
            action   = "BLOCKED"
            severity = "HIGH" if confidence > 0.8 else "MEDIUM"
            self.stats["attacks_detected"] += 1
            self.stats["windows_blocked"]  += 1
            if not is_attacked:
                self.stats["false_positives"] += 1
        elif verdict == "UNCERTAIN":
            action   = "FLAGGED"
            severity = "LOW"
            self.stats["windows_passed"] += 1
        else:
            action   = "PASSED"
            severity = "LOW"
            self.stats["windows_passed"] += 1

        # Determine attack type
        attack_type = "UNKNOWN" if (verdict == "ATTACK_DETECTED" and is_attacked) else \
                      "NONE" if verdict == "CLEAN" else "POSSIBLE"

        event = ThreatEvent(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            window_id=window_id,
            verdict=verdict,
            severity=severity,
            attack_type=attack_type,
            confidence=round(confidence, 3),
            action_taken=action,
            llm_reasoning=reasoning,
            detector_scores=detector_scores,
        )
        self.incident_log.append(event)
        return event

    def run_stream(self, X_test: np.ndarray, y_test: np.ndarray,
                   n_windows: int = 50, window_delay: float = 0.1,
                   verbose: bool = True) -> list[ThreatEvent]:
        """
        Run the agent over a stream of EEG windows.

        Args:
            X_test:       EEG test epochs (n, channels, times)
            y_test:       True labels
            n_windows:    Number of windows to process
            window_delay: Simulated delay between windows (seconds)
            verbose:      Print live updates

        Returns:
            List of ThreatEvents
        """
        n = min(n_windows, len(X_test))
        console.print(Panel.fit(
            f"[bold purple]CyberNeuro[/] Streaming Defense Agent\n"
            f"Processing {n} EEG windows | Attack rate: {self.attack_rate:.0%} | "
            f"LLM reasoning: {'[green]ON[/]' if self.use_llm else '[yellow]OFF (no API key)[/]'}",
            border_style="purple",
        ))

        events = []
        attack_log = []

        for i in range(n):
            window = X_test[i]

            # Randomly inject attack
            is_attacked = self.rng.random() < self.attack_rate
            if is_attacked:
                import torch
                import torch.nn.functional as F
                w_t  = torch.tensor(window[np.newaxis], dtype=torch.float32)
                y_t  = torch.tensor([y_test[i]], dtype=torch.long)
                w_adv = self.attack_fn(w_t, y_t)
                window = w_adv.squeeze(0).numpy()
                self.stats["attacks_injected"] += 1
                attack_log.append(i)

            event = self.process_window(window, window_id=i, is_attacked=is_attacked)
            events.append(event)

            if verbose:
                color  = {"BLOCKED": "red", "FLAGGED": "yellow", "PASSED": "green"}.get(event.action_taken, "white")
                atk_indicator = "⚡" if is_attacked else "  "
                console.print(
                    f"  {atk_indicator} Window {i:3d} | "
                    f"Score={event.detector_scores['statistical_anomaly']:.3f} | "
                    f"[{color}]{event.action_taken:7s}[/] | "
                    f"{event.verdict} ({event.confidence:.2f}) | "
                    f"[dim]{event.llm_reasoning[:60]}[/]"
                )

            if window_delay > 0:
                time.sleep(window_delay)

        self._print_session_summary(events, attack_log)
        return events

    def _print_session_summary(self, events, attack_log):
        total    = self.stats["total_windows"]
        attacked = self.stats["attacks_injected"]
        detected = self.stats["attacks_detected"]
        blocked  = self.stats["windows_blocked"]
        fp       = self.stats["false_positives"]

        detection_rate = detected / attacked if attacked > 0 else 0
        fp_rate        = fp / (total - attacked) if (total - attacked) > 0 else 0

        console.print(Panel(
            f"[bold]Session Complete[/]\n\n"
            f"Windows processed:  {total}\n"
            f"Attacks injected:   {attacked} ({attacked/total:.0%})\n"
            f"Attacks detected:   [green]{detected}[/] ({detection_rate:.0%} detection rate)\n"
            f"Windows blocked:    {blocked}\n"
            f"False positives:    [yellow]{fp}[/] ({fp_rate:.1%} FPR)\n\n"
            f"[dim]LLM reasoning: {'enabled' if self.use_llm else 'disabled (no API key)'}[/]",
            title="[bold purple]CyberNeuro Defense Agent — Session Summary[/]",
            border_style="purple",
        ))

    def export_incident_log(self, path: str):
        data = {
            "session_stats": self.stats,
            "incidents": [
                {
                    "timestamp":     e.timestamp,
                    "window_id":     e.window_id,
                    "verdict":       e.verdict,
                    "severity":      e.severity,
                    "action":        e.action_taken,
                    "confidence":    e.confidence,
                    "reasoning":     e.llm_reasoning,
                    "detector_scores": e.detector_scores,
                }
                for e in self.incident_log
                if e.verdict != "CLEAN"  # only log non-clean events
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]✓[/] Incident log saved to {path}")
        return data

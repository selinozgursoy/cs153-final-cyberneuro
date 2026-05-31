"""
CyberNeuro — Neural Data Security Platform
===========================================
Agentic security and privacy platform for Brain-Computer Interfaces.

Four security modules running autonomously:
  1. EEGNet Classifier       — learns to decode brain signals (the attack target)
  2. Adversarial Defense     — attacks the classifier, then detects those attacks
  3. Compliance Agent        — audits BCI vendors against neurorights laws (needs API key)
  4. De-Anonymization Scorer — measures privacy leakage from "anonymized" brain data

Plus the agentic streaming defense agent that monitors live EEG in real time.

HOW TO RUN:
-----------
  # Full platform (synthetic EEG, no accounts needed):
  python cyberneuro.py

  # With real PhysioNet data:
  python cyberneuro.py --subjects 10

  # With streaming agent demo:
  python cyberneuro.py --stream

  # With compliance auditing (needs API key):
  export OPENROUTER_API_KEY=your_key_here
  python cyberneuro.py --compliance

  # Everything:
  python cyberneuro.py --subjects 10 --stream --compliance

  # Skip slow components for quick test:
  python cyberneuro.py --quick

OUTPUT:
-------
  results/security_report.json     — full security assessment
  results/incident_log.json        — streaming agent incident log
  results/compliance_*.json        — vendor compliance reports
  results/deanon_report.json       — privacy risk assessment
  results/robustness_curve.png     — attack strength vs accuracy
  results/detector_roc.png         — anomaly detector performance
  results/eeg_comparison.png       — clean vs adversarial signal
  results/training_history.png     — classifier training curves
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.panel import Panel

console = Console()
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_PATH  = ROOT / "models" / "eegnet_trained.pt"


def parse_args():
    p = argparse.ArgumentParser(description="CyberNeuro — Neural Data Security Platform")
    p.add_argument("--subjects",    type=int,  default=5,
                   help="Subjects to load (1-109). Default: 5.")
    p.add_argument("--stream",      action="store_true",
                   help="Run real-time streaming defense agent demo.")
    p.add_argument("--compliance",  action="store_true",
                   help="Run vendor compliance auditing (requires API key).")
    p.add_argument("--quick",       action="store_true",
                   help="Quick mode — fewer epochs, faster run.")
    p.add_argument("--device",      type=str,  default="cpu")
    p.add_argument("--epsilon",     type=float, default=0.05)
    p.add_argument("--attack-rate", type=float, default=0.25,
                   help="Fraction of streaming windows that get attacked.")
    return p.parse_args()


def load_data(args):
    """Load EEG — real PhysioNet or synthetic fallback."""
    try:
        import mne
        from mne.datasets import eegbci
        subjects = list(range(1, args.subjects + 1))
        console.print(f"[bold]Loading PhysioNet EEG — {args.subjects} subjects[/]")
        console.print("[dim]First run downloads ~20MB/subject into mne_data/[/]")

        mne.set_log_level("WARNING")
        all_X, all_y, all_s = [], [], []
        sfreq = ch_names = None

        for subj in subjects:
            try:
                raw_fnames = eegbci.load_data(subj, runs=[4, 8], verbose=False)
                raws = []
                for fname in raw_fnames:
                    raw = mne.io.read_raw_edf(fname, preload=True, verbose=False)
                    eegbci.standardize(raw)
                    raws.append(raw)
                raw_c = mne.concatenate_raws(raws)
                raw_c.filter(8.0, 30.0, method="iir", verbose=False)
                events, event_id = mne.events_from_annotations(raw_c, verbose=False)
                valid = {k: v for k, v in event_id.items() if v in [2, 3]}
                if len(valid) < 2:
                    continue
                epochs = mne.Epochs(raw_c, events, event_id=valid, tmin=0.0, tmax=2.0,
                                    baseline=None, preload=True, verbose=False)
                epochs.drop_bad(verbose=False)
                X = epochs.get_data().astype(np.float32)
                mean = X.mean(axis=2, keepdims=True)
                std  = X.std(axis=2, keepdims=True) + 1e-8
                X    = (X - mean) / std
                y    = (epochs.events[:, 2] == 3).astype(np.int64)
                all_X.append(X); all_y.append(y)
                all_s.extend([subj] * len(y))
                if sfreq is None:
                    sfreq    = epochs.info["sfreq"]
                    ch_names = epochs.ch_names
            except Exception:
                continue

        if all_X:
            from core.synthetic_eeg import EEGDataset
            X_all = np.concatenate(all_X)
            y_all = np.concatenate(all_y)
            console.print(f"[green]✓[/] Loaded {len(X_all)} real EEG epochs")
            return EEGDataset(X=X_all, y=y_all, subject_ids=np.array(all_s),
                              sfreq=sfreq, ch_names=ch_names, tmin=0.0, tmax=2.0,
                              n_subjects=len(subjects))
    except Exception:
        pass

    console.print("[yellow]PhysioNet unavailable — using synthetic EEG.[/]\n"
                  "[dim]To use real data: register at physionet.org, then:\n"
                  "  python3 -c \"import mne; mne.set_config('PHYSIONET_USER','username')\"\n"
                  "  python3 -c \"import mne; mne.set_config('PHYSIONET_PASS','password')\"[/]")
    from core.synthetic_eeg import generate_synthetic_dataset
    return generate_synthetic_dataset(n_subjects=args.subjects, n_epochs_per_subject=60)


def run():
    args = parse_args()

    console.print(Panel.fit(
        "[bold purple]CyberNeuro[/] — Neural Data Security Platform\n"
        "Stanford CS153 · Agentic BCI Security Research\n\n"
        "[dim]Module 1: EEG Signal Classifier (EEGNet)\n"
        "Module 2: Adversarial Attack Engine + Real-Time Detection\n"
        "Module 3: Vendor Compliance Agent (neurorights laws)\n"
        "Module 4: De-Anonymization Risk Scorer\n"
        "Module 5: Streaming Defense Agent (agentic)[/]",
        border_style="purple",
    ))

    # ── Load Data ─────────────────────────────────────────────────────────────
    console.print(Rule("[bold]Loading EEG Data[/]"))
    dataset = load_data(args)
    console.print(dataset)

    from core.synthetic_eeg import train_test_split_by_subject
    all_subj   = list(np.unique(dataset.subject_ids))
    test_subj  = all_subj[-2:] if len(all_subj) >= 4 else all_subj[-1:]
    train_data, test_data = train_test_split_by_subject(dataset, test_subj)
    console.print(f"  Train: {train_data.n_epochs} epochs | Test: {test_data.n_epochs} epochs")

    # ── Module 1: EEGNet ──────────────────────────────────────────────────────
    console.print(Rule("[bold]Module 1 — EEGNet Signal Classifier[/]"))
    from models.eegnet import EEGNet, train_classifier, evaluate_classifier

    model = EEGNet(n_channels=train_data.n_channels, n_times=train_data.n_times,
                   n_classes=2, sfreq=train_data.sfreq,
                   dropout=0.25 if args.quick else 0.5)
    console.print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    rng  = np.random.RandomState(42)
    idx  = rng.permutation(train_data.n_epochs)
    n_tr = int(0.8 * train_data.n_epochs)

    history = train_classifier(
        model,
        X_train=train_data.X[idx[:n_tr]], y_train=train_data.y[idx[:n_tr]],
        X_val=train_data.X[idx[n_tr:]],   y_val=train_data.y[idx[n_tr:]],
        epochs=30 if args.quick else 150,
        device=args.device, save_path=MODEL_PATH,
    )
    if MODEL_PATH.exists(): model.load_state_dict(torch.load(MODEL_PATH, map_location=args.device, weights_only=True))
    clean_metrics = evaluate_classifier(model, test_data.X, test_data.y, args.device)

    try:
        from evaluation.visualize import plot_training_history, plot_eeg_comparison, \
            plot_robustness_curve, plot_detector_roc
        plot_training_history(history)
    except Exception:
        pass

    # ── Module 2: Adversarial Attacks + Detection ─────────────────────────────
    console.print(Rule("[bold]Module 2 — Adversarial Attacks + Detection[/]"))
    from attacks.adversarial import AttackConfig, run_attack_evaluation, run_epsilon_sweep
    from evaluation.anomaly_detector import (
        StatisticalAnomalyDetector, IsolationForestDetector,
        ConfidenceBasedDetector, evaluate_detector, compare_detectors,
    )

    max_s  = 80 if args.quick else 200
    cfg    = AttackConfig(epsilon=args.epsilon, alpha=args.epsilon/8,
                           n_steps=10 if args.quick else 40)
    attacks = run_attack_evaluation(model, test_data.X, test_data.y, cfg,
                                     args.device, max_s)

    fgsm_adv = attacks["FGSM"].adversarial_examples
    n_adv    = min(max_s, len(fgsm_adv))
    X_clean  = test_data.X[:n_adv]
    X_adv    = fgsm_adv[:n_adv]
    n_tr_det = int(0.6 * n_adv)

    stat_det = StatisticalAnomalyDetector(sfreq=test_data.sfreq)
    stat_det.fit(X_clean[:n_tr_det], X_adv[:n_tr_det])
    stat_m = evaluate_detector(stat_det, X_clean[n_tr_det:], X_adv[n_tr_det:], "Statistical")

    iso_det = IsolationForestDetector(sfreq=test_data.sfreq)
    iso_det.fit(X_clean[:n_tr_det])
    iso_m = evaluate_detector(iso_det, X_clean[n_tr_det:], X_adv[n_tr_det:], "Isolation Forest")

    conf_det = ConfidenceBasedDetector(model, threshold=0.35, device=args.device)
    conf_m   = evaluate_detector(conf_det, X_clean[n_tr_det:], X_adv[n_tr_det:], "Confidence")

    det_metrics = [stat_m, iso_m, conf_m]
    compare_detectors(det_metrics)

    epsilons = [0.01, 0.05, 0.1] if args.quick else [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    sweep = run_epsilon_sweep(model, test_data.X, test_data.y,
                               epsilons=epsilons, device=args.device, max_samples=max_s)

    try:
        plot_eeg_comparison(X_clean, X_adv, test_data.ch_names, test_data.sfreq)
        plot_robustness_curve(sweep)
        plot_detector_roc(X_clean[n_tr_det:], X_adv[n_tr_det:],
                          [stat_det, iso_det, conf_det],
                          ["Statistical", "Isolation Forest", "Confidence"])
    except Exception:
        pass

    # ── Module 3: Compliance Agent ────────────────────────────────────────────
    console.print(Rule("[bold]Module 3 — Vendor Compliance Agent[/]"))
    compliance_results = {}
    api_key = (os.environ.get("OPENROUTER_API_KEY") or
               os.environ.get("ANTHROPIC_API_KEY"))

    if not args.compliance:
        console.print("[yellow]Skipped — run with --compliance to audit vendors[/]")
        from core.compliance_agent import _demo_structure_only
        _demo_structure_only()
    elif not api_key:
        console.print("[yellow]No API key found — set OPENROUTER_API_KEY or ANTHROPIC_API_KEY[/]")
        from core.compliance_agent import _demo_structure_only
        _demo_structure_only()
    else:
        from core.compliance_agent import ComplianceAgent, DEMO_POLICIES
        agent = ComplianceAgent(api_key=api_key)
        for vendor, policy in DEMO_POLICIES.items():
            report = agent.audit(policy.strip(), vendor_name=vendor)
            agent.export_json(report,
                str(RESULTS_DIR / f"compliance_{vendor.split()[0].lower()}.json"))
            compliance_results[vendor] = {
                "score": report.overall_score,
                "risk": report.risk_level,
                "critical_violations": report.n_critical,
            }

    # ── Module 4: De-Anonymization Scorer ─────────────────────────────────────
    console.print(Rule("[bold]Module 4 — De-Anonymization Risk Scorer[/]"))
    from core.deanon_scorer import score_dataset, export_report_json

    deanon = score_dataset(
        X=dataset.X, y_class=dataset.y,
        subject_ids=dataset.subject_ids, sfreq=dataset.sfreq,
        dataset_description=f"CyberNeuro EEG ({dataset.n_subjects} subjects)",
    )
    export_report_json(deanon, str(RESULTS_DIR / "deanon_report.json"))

    # ── Module 5: Streaming Defense Agent ────────────────────────────────────
    incident_data = {}
    if args.stream:
        console.print(Rule("[bold]Module 5 — Streaming Defense Agent[/]"))
        from core.streaming_agent import StreamingDefenseAgent
        import functools

        def _fgsm_fn(X_t, y_t):
            from attacks.adversarial import fgsm_attack
            return fgsm_attack(model, X_t, y_t, epsilon=args.epsilon, device=args.device)

        agent = StreamingDefenseAgent(
            detector=stat_det, model=model, attack_fn=_fgsm_fn,
            api_key=api_key, attack_rate=args.attack_rate,
            detection_threshold=0.5, use_llm=bool(api_key),
        )
        n_stream = 20 if args.quick else 50
        events   = agent.run_stream(test_data.X, test_data.y,
                                     n_windows=n_stream, window_delay=0.05)
        incident_data = agent.export_incident_log(
            str(RESULTS_DIR / "incident_log.json")
        )

    # ── Save Master Report ────────────────────────────────────────────────────
    console.print(Rule("[bold]Saving Security Report[/]"))
    best_det = max(det_metrics, key=lambda m: m.auc_roc)

    report = {
        "platform": "CyberNeuro",
        "dataset": {
            "n_subjects": int(dataset.n_subjects),
            "n_epochs": int(dataset.n_epochs),
            "n_channels": int(dataset.n_channels),
            "sfreq": float(dataset.sfreq),
        },
        "module_1_classifier": {
            "architecture": "EEGNet",
            "parameters": sum(p.numel() for p in model.parameters()),
            "accuracy":   round(clean_metrics["accuracy"], 4),
            "roc_auc":    round(clean_metrics["roc_auc"], 4),
        },
        "module_2_adversarial": {
            "attacks": {
                name: {
                    "epsilon": round(r.epsilon, 4),
                    "attack_success_rate": round(r.attack_success_rate, 4),
                    "adversarial_accuracy": round(r.adversarial_accuracy, 4),
                    "snr_db": round(r.snr_db, 2),
                }
                for name, r in attacks.items()
            },
            "detectors": {
                m.method: {
                    "auc_roc": round(m.auc_roc, 4),
                    "recall":  round(m.recall,  4),
                    "fpr":     round(m.false_positive_rate, 4),
                    "f1":      round(m.f1_score, 4),
                }
                for m in det_metrics
            },
            "robustness_curve": {
                "epsilons": sweep["epsilons"],
                "fgsm_accuracy": [round(x,4) for x in sweep["fgsm_acc"]],
                "pgd_accuracy":  [round(x,4) for x in sweep["pgd_acc"]],
            },
        },
        "module_3_compliance": compliance_results,
        "module_4_deanonymization": {
            "overall_risk_score": deanon.overall_risk_score,
            "overall_risk_level": deanon.overall_risk_level,
            "tasks": {t.name: {"accuracy": t.accuracy, "lift": t.lift_over_chance,
                                "auc": t.auc_roc, "risk": t.risk_level}
                      for t in deanon.tasks},
        },
        "module_5_streaming": incident_data.get("session_stats", {}),
    }

    out = RESULTS_DIR / "security_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"[green]✓[/] Security report saved to {out}")

    # ── Final Summary ─────────────────────────────────────────────────────────
    stream_line = ""
    if args.stream and incident_data.get("session_stats"):
        s = incident_data["session_stats"]
        dr = s.get("attacks_detected",0) / max(s.get("attacks_injected",1), 1)
        stream_line = f"\n[bold]Module 5:[/] {s.get('attacks_injected',0)} attacks detected in real time ({dr:.0%} detection rate)"

    console.print(Panel.fit(
        f"[bold green]CyberNeuro Security Assessment Complete[/]\n\n"
        f"[bold]Module 1:[/] EEGNet accuracy = {clean_metrics['accuracy']:.1%} on clean signals\n"
        f"[bold]Module 2:[/] FGSM {attacks['FGSM'].attack_success_rate:.0%} attack success at ε={args.epsilon} | "
        f"Best detector AUC = {best_det.auc_roc:.3f}\n"
        f"[bold]Module 3:[/] {'Compliance audits complete — see results/' if compliance_results else 'Run with --compliance to audit BCI vendors'}\n"
        f"[bold]Module 4:[/] De-anonymization risk = {deanon.overall_risk_level} ({deanon.overall_risk_score:.0f}/100)"
        f"{stream_line}\n\n"
        f"[dim]Full report: results/security_report.json[/]",
        border_style="green",
        title="[bold]Security Assessment Summary[/]",
    ))


if __name__ == "__main__":
    run()

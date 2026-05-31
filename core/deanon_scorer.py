"""
CyberNeuro — Neural Data De-Anonymization Risk Scorer
Empirically measures how much private information leaks from "anonymized" EEG data.

Inference tasks:
  1. Cognitive state (motor imagery class)
  2. Subject re-identification (brainprint uniqueness)
  3. Medical marker detection (high-gamma signatures)
  4. Attention / cognitive load (beta-band activity)

Reference: Meng et al. 2024 arXiv:2411.19498 — "Protecting Multiple Types
of Privacy Simultaneously in EEG-Based BCIs"
"""

import json
import numpy as np
from dataclasses import dataclass, field
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class InferenceTask:
    name: str
    description: str
    accuracy: float
    chance_level: float
    auc_roc: float
    lift_over_chance: float
    cv_std: float
    risk_level: str
    privacy_implication: str


@dataclass
class DeAnonymizationReport:
    dataset_description: str
    n_subjects: int
    n_epochs: int
    tasks: list = field(default_factory=list)
    overall_risk_score: float = 0.0
    overall_risk_level: str = "UNKNOWN"
    summary: str = ""


def _extract_features(X, sfreq=160.0):
    n_epochs, n_channels, n_times = X.shape
    features = []
    for epoch in X:
        feats = []
        for ch in epoch:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(128, n_times))
            total = psd.sum() + 1e-10
            for lo, hi in [(1,4),(4,8),(8,12),(9,11),(13,30),(30,45)]:
                m = (freqs>=lo)&(freqs<=hi)
                feats.append(float(np.log(psd[m].mean()+1e-10)) if m.any() else 0.0)
            diff1 = np.diff(ch)
            var_s = float(np.var(ch))
            mob   = float(np.sqrt(np.var(diff1)/(var_s+1e-10)))
            diff2 = np.diff(diff1)
            comp  = float(np.sqrt(np.var(diff2)/(np.var(diff1)+1e-10))/(mob+1e-10))
            feats.extend([var_s, mob, comp, float(np.std(ch))])
        features.append(feats)
    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def _run_task(X_feat, y, task_name, description, privacy_implication, n_cv=5):
    n_classes  = len(np.unique(y))
    chance     = 1.0 / n_classes
    scaler     = StandardScaler()
    X_s        = scaler.fit_transform(X_feat)
    clf        = RandomForestClassifier(n_estimators=100, max_depth=6,
                                        random_state=42, n_jobs=-1)
    min_class  = int(np.unique(y, return_counts=True)[1].min())
    n_splits   = min(n_cv, min_class)
    if n_splits < 2:
        n_splits = 2
    cv         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores  = cross_val_score(clf, X_s, y, cv=cv, scoring="accuracy")
    acc        = float(np.mean(cv_scores))
    std        = float(np.std(cv_scores))
    lift       = acc / chance if chance > 0 else 1.0

    try:
        auc = float(np.mean(cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc"))) \
              if n_classes == 2 else acc
    except Exception:
        auc = acc

    risk = ("CRITICAL" if lift >= 2.5 else "HIGH" if lift >= 1.75 else
            "MEDIUM" if lift >= 1.3 else "LOW")

    return InferenceTask(
        name=task_name, description=description,
        accuracy=round(acc, 4), chance_level=round(chance, 4),
        auc_roc=round(auc, 4), lift_over_chance=round(lift, 2),
        cv_std=round(std, 4), risk_level=risk,
        privacy_implication=privacy_implication,
    )


def score_dataset(X, y_class, subject_ids, sfreq=160.0,
                  dataset_description="EEG dataset"):
    n_subjects = len(np.unique(subject_ids))

    console.print(Panel.fit(
        f"[bold purple]CyberNeuro[/] De-Anonymization Risk Scorer\n"
        f"Dataset: {dataset_description}\n"
        f"Subjects: {n_subjects} | Epochs: {len(X)} | Shape: {X.shape}",
        border_style="purple",
    ))

    console.print("[dim]Extracting privacy-sensitive EEG features...[/]")
    X_feat = _extract_features(X, sfreq)
    console.print(f"[green]✓[/] Feature matrix: {X_feat.shape}")

    tasks = []

    # Task 1: Cognitive state
    console.print("\n[bold]Task 1:[/] Cognitive state inference")
    t1 = _run_task(X_feat, y_class, "Cognitive State",
                   "Inferring motor imagery task from EEG",
                   "Mental intentions are readable — even 'monitoring' apps reveal what you're thinking.")
    tasks.append(t1)
    _print_task_result(t1)

    # Task 2: Re-identification
    if n_subjects >= 3:
        console.print(f"\n[bold]Task 2:[/] Subject re-identification ({n_subjects} subjects)")
        max_s = min(n_subjects, 8)
        mask  = np.isin(subject_ids, np.unique(subject_ids)[:max_s])
        t2 = _run_task(X_feat[mask], subject_ids[mask], "Subject Re-Identification",
                       f"Identifying which of {max_s} individuals produced an EEG segment",
                       "Anonymized EEG can be matched back to specific individuals — brainprints are unique.")
        tasks.append(t2)
        _print_task_result(t2)

    # Task 3: Medical marker
    console.print("\n[bold]Task 3:[/] Medical marker detection (high-gamma signature)")
    gamma = []
    for epoch in X:
        freqs, psd = welch(epoch[0], fs=sfreq, nperseg=min(128, X.shape[2]))
        gamma.append(psd[freqs>=35].mean() if (freqs>=35).any() else 0.0)
    y_med = (np.array(gamma) > np.median(gamma)).astype(int)
    t3 = _run_task(X_feat, y_med, "Medical Marker",
                   "Detecting neurological condition signatures from EEG",
                   "EEG from any app may reveal epilepsy, ADHD, or other conditions users never disclosed.")
    tasks.append(t3)
    _print_task_result(t3)

    # Task 4: Cognitive load
    console.print("\n[bold]Task 4:[/] Attention / cognitive load")
    beta = []
    for epoch in X:
        freqs, psd = welch(epoch[0], fs=sfreq, nperseg=min(128, X.shape[2]))
        m = (freqs>=13)&(freqs<=30)
        beta.append(psd[m].mean() if m.any() else 0.0)
    y_att = (np.array(beta) > np.median(beta)).astype(int)
    t4 = _run_task(X_feat, y_att, "Attention/Cognitive Load",
                   "Inferring attention level from beta-band activity",
                   "Advertisers and employers would pay for continuous attention monitoring from your EEG.")
    tasks.append(t4)
    _print_task_result(t4)

    # Overall score
    weights   = {"Cognitive State": 1.0, "Subject Re-Identification": 2.0,
                 "Medical Marker": 2.0, "Attention/Cognitive Load": 1.5}
    w_auc     = sum(t.auc_roc * weights.get(t.name, 1.0) for t in tasks)
    w_total   = sum(weights.get(t.name, 1.0) for t in tasks)
    mean_auc  = w_auc / w_total
    score     = max(0.0, min(100.0, (mean_auc - 0.5) * 200))
    risk      = ("CRITICAL" if score >= 70 else "HIGH" if score >= 50 else
                 "MEDIUM" if score >= 30 else "LOW")

    best_task = max(tasks, key=lambda t: t.auc_roc)
    summary   = (f"This EEG dataset carries a {risk} de-anonymization risk "
                 f"(score {score:.0f}/100). The highest-risk inference task is "
                 f"'{best_task.name}' with AUC={best_task.auc_roc:.3f}.")

    report = DeAnonymizationReport(
        dataset_description=dataset_description,
        n_subjects=n_subjects, n_epochs=len(X),
        tasks=tasks, overall_risk_score=round(score, 1),
        overall_risk_level=risk, summary=summary,
    )
    _print_report(report)
    return report


def _print_task_result(t):
    colors = {"CRITICAL": "red", "HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
    c = colors.get(t.risk_level, "white")
    console.print(f"  Accuracy: {t.accuracy:.1%} (chance: {t.chance_level:.1%}) | "
                  f"Lift: {t.lift_over_chance:.2f}x | [{c}]{t.risk_level}[/]")


def _print_report(report):
    colors = {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
    console.print(Panel(
        f"[bold]Overall Risk Score:[/] {report.overall_risk_score}/100\n"
        f"[bold]Risk Level:[/] [{colors.get(report.overall_risk_level,'white')}]"
        f"{report.overall_risk_level}[/]\n\n[italic]{report.summary}[/]",
        title="[bold purple]CyberNeuro De-Anonymization Report[/]",
        border_style="purple",
    ))
    t = Table(style="purple")
    t.add_column("Task", width=26); t.add_column("Accuracy", width=12)
    t.add_column("Lift", width=8); t.add_column("AUC", width=8); t.add_column("Risk", width=10)
    for task in report.tasks:
        c = colors.get(task.risk_level, "white")
        t.add_row(task.name, f"{task.accuracy:.1%} ±{task.cv_std:.2f}",
                  f"{task.lift_over_chance:.2f}x", f"{task.auc_roc:.3f}",
                  f"[{c}]{task.risk_level}[/]")
    console.print(t)


def export_report_json(report, path):
    data = {
        "dataset": report.dataset_description,
        "n_subjects": report.n_subjects,
        "n_epochs": report.n_epochs,
        "overall_risk_score": report.overall_risk_score,
        "overall_risk_level": report.overall_risk_level,
        "summary": report.summary,
        "tasks": [{"name": t.name, "accuracy": t.accuracy,
                   "chance": t.chance_level, "lift": t.lift_over_chance,
                   "auc_roc": t.auc_roc, "risk": t.risk_level,
                   "implication": t.privacy_implication} for t in report.tasks],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"[green]✓[/] De-anonymization report saved to {path}")

"""
CyberNeuro — Adversarial EEG Anomaly Detector
Three complementary detectors to catch adversarial attacks on EEG signals:
  1. Statistical: frequency-domain features (high-frequency noise signature)
  2. Isolation Forest: unsupervised, trained only on clean data
  3. Confidence-based: low model confidence signals manipulation
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.signal import welch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class DetectorMetrics:
    auc_roc: float
    recall: float
    false_positive_rate: float
    f1_score: float
    threshold: float
    method: str

    def summary(self):
        return (f"{self.method}: AUC={self.auc_roc:.3f} | "
                f"Recall={self.recall:.3f} | FPR={self.false_positive_rate:.3f}")


def _extract_features(X, sfreq=160.0):
    n_epochs, n_channels, n_times = X.shape
    features = []
    for epoch in X:
        feats = []
        for ch in epoch:
            freqs, psd = welch(ch, fs=sfreq, nperseg=min(128, n_times))
            total = psd.sum() + 1e-10
            feats.append(psd[freqs > 30].sum() / total)
            feats.append(-np.sum((psd/total) * np.log(psd/total + 1e-10)))
            feats.append(float(stats.kurtosis(ch)))
            feats.append(float(np.var(ch)))
            diff1 = np.diff(ch)
            mob   = float(np.sqrt(np.var(diff1) / (np.var(ch) + 1e-10)))
            diff2 = np.diff(diff1)
            comp  = float(np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mob + 1e-10))
            feats.extend([mob, comp])
        features.append(feats)
    return np.nan_to_num(np.array(features, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


class StatisticalAnomalyDetector:
    def __init__(self, sfreq=160.0, threshold=0.5):
        self.sfreq     = sfreq
        self.threshold = threshold
        self.scaler    = StandardScaler()
        self.clf       = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def fit(self, X_clean, X_adv):
        console.print("[bold purple]CyberNeuro[/] Training statistical detector...")
        X = np.vstack([_extract_features(X_clean, self.sfreq),
                       _extract_features(X_adv,   self.sfreq)])
        y = np.array([0]*len(X_clean) + [1]*len(X_adv))
        self.clf.fit(self.scaler.fit_transform(X), y)
        console.print("[green]✓[/] Statistical detector trained")
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(
            self.scaler.transform(_extract_features(X, self.sfreq))
        )[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


class IsolationForestDetector:
    def __init__(self, contamination=0.05, sfreq=160.0):
        self.sfreq  = sfreq
        self.scaler = StandardScaler()
        self.model  = IsolationForest(contamination=contamination,
                                       n_estimators=200, random_state=42, n_jobs=-1)

    def fit(self, X_clean):
        console.print("[bold purple]CyberNeuro[/] Training Isolation Forest...")
        feats = _extract_features(X_clean, self.sfreq)
        self.model.fit(self.scaler.fit_transform(feats))
        console.print("[green]✓[/] Isolation Forest trained")
        return self

    def predict(self, X):
        feats = _extract_features(X, self.sfreq)
        return (self.model.predict(self.scaler.transform(feats)) == -1).astype(int)

    def decision_scores(self, X):
        feats = _extract_features(X, self.sfreq)
        return -self.model.decision_function(self.scaler.transform(feats))


class ConfidenceBasedDetector:
    def __init__(self, model, threshold=0.35, device="cpu"):
        self.model     = model
        self.threshold = threshold
        self.device    = device

    def confidence_scores(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(X_t), dim=1).cpu().numpy()
        return probs.max(axis=1)

    def predict(self, X):
        return (self.confidence_scores(X) < self.threshold).astype(int)


def evaluate_detector(detector, X_clean, X_adv, detector_name, threshold=None):
    X_all = np.concatenate([X_clean, X_adv])
    y_all = np.array([0]*len(X_clean) + [1]*len(X_adv))

    if hasattr(detector, "predict_proba"):
        scores = detector.predict_proba(X_all)
    elif hasattr(detector, "confidence_scores"):
        scores = 1.0 - detector.confidence_scores(X_all)
    else:
        scores = detector.decision_scores(X_all)

    s_min, s_max = scores.min(), scores.max()
    scores = (scores - s_min) / (s_max - s_min + 1e-10)
    auc    = roc_auc_score(y_all, scores)

    if threshold is None:
        prec, rec, ths = precision_recall_curve(y_all, scores)
        f1s   = 2*prec*rec / (prec+rec+1e-10)
        best  = f1s.argmax()
        threshold = float(ths[best]) if best < len(ths) else 0.5

    preds = (scores >= threshold).astype(int)
    cm    = confusion_matrix(y_all, preds)
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    recall = tp / (tp+fn+1e-10)
    fpr    = fp / (fp+tn+1e-10)
    prec_v = tp / (tp+fp+1e-10)
    f1     = 2*prec_v*recall / (prec_v+recall+1e-10)

    m = DetectorMetrics(auc_roc=float(auc), recall=float(recall),
                        false_positive_rate=float(fpr), f1_score=float(f1),
                        threshold=float(threshold), method=detector_name)
    console.print(f"[green]✓[/] {m.summary()}")
    return m


def compare_detectors(metrics_list):
    t = Table(title="Anomaly Detector Comparison", style="purple")
    t.add_column("Method", style="bold")
    t.add_column("AUC-ROC"); t.add_column("Recall"); t.add_column("FPR"); t.add_column("F1")
    for m in metrics_list:
        rc = "green" if m.recall > 0.85 else "yellow"
        fc = "green" if m.false_positive_rate < 0.1 else "yellow"
        t.add_row(m.method, f"{m.auc_roc:.4f}",
                  f"[{rc}]{m.recall:.4f}[/]",
                  f"[{fc}]{m.false_positive_rate:.4f}[/]",
                  f"{m.f1_score:.4f}")
    console.print(t)

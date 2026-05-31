"""CyberNeuro — Visualization Module"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
COLORS = {"clean":"#534AB7","adversarial":"#E24B4A","detected":"#1D9E75","missed":"#EF9F27"}


def plot_training_history(history, filename="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EEGNet Training", fontsize=12, fontweight="bold")
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, key_pair, title in zip(axes,
        [("train_loss","val_loss"),("train_acc","val_acc")],
        ["Loss","Accuracy"]):
        ax.plot(epochs, history[key_pair[0]], color=COLORS["clean"], lw=2, label="Train")
        ax.plot(epochs, history[key_pair[1]], color=COLORS["adversarial"], lw=2, label="Val")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_eeg_comparison(X_clean, X_adv, ch_names, sfreq,
                         n_channels_show=4, filename="eeg_comparison.png"):
    n_ch = min(n_channels_show, X_clean.shape[1])
    n_times = X_clean.shape[2]
    times   = np.arange(n_times) / sfreq
    fig, axes = plt.subplots(n_ch, 2, figsize=(14, 2.5*n_ch))
    fig.suptitle("CyberNeuro: Clean vs Adversarial EEG", fontsize=12, fontweight="bold")
    for i in range(n_ch):
        axes[i,0].plot(times, X_clean[0,i], color=COLORS["clean"],       lw=1.2, label="Clean")
        axes[i,0].plot(times, X_adv[0,i],   color=COLORS["adversarial"], lw=1.0, label="Adversarial", alpha=0.8)
        axes[i,0].set_ylabel(ch_names[i] if i < len(ch_names) else f"Ch{i}", fontsize=8)
        if i == 0: axes[i,0].legend(fontsize=7); axes[i,0].set_title("Time Domain")
        delta = X_adv[0,i] - X_clean[0,i]
        axes[i,1].plot(times, delta, color=COLORS["adversarial"], lw=1.0)
        axes[i,1].fill_between(times, delta, 0, alpha=0.2, color=COLORS["adversarial"])
        if i == 0: axes[i,1].set_title("Perturbation (adversarial − clean)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_robustness_curve(sweep, filename="robustness_curve.png"):
    eps  = sweep["epsilons"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("CyberNeuro: Adversarial Robustness", fontsize=12, fontweight="bold")
    axes[0].plot(eps, sweep["fgsm_acc"], "o-", color=COLORS["adversarial"], label="FGSM", lw=2)
    axes[0].plot(eps, sweep["pgd_acc"],  "s-", color="#7B3FA0",             label="PGD",  lw=2)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1, label="Chance")
    axes[0].set_xlabel("ε"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Attack Strength"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(eps, sweep["fgsm_asr"], "o-", color=COLORS["adversarial"], label="FGSM", lw=2)
    axes[1].plot(eps, sweep["pgd_asr"],  "s-", color="#7B3FA0",             label="PGD",  lw=2)
    axes[1].set_xlabel("ε"); axes[1].set_ylabel("Attack Success Rate")
    axes[1].set_title("Attack Success Rate vs Strength"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_detector_roc(X_clean, X_adv, detectors, detector_names,
                       filename="detector_roc.png"):
    from sklearn.metrics import roc_curve, roc_auc_score
    X_all = np.concatenate([X_clean, X_adv])
    y_all = np.array([0]*len(X_clean) + [1]*len(X_adv))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],"k--",lw=1,label="Random")
    colors = ["#534AB7","#E24B4A","#1D9E75"]
    for det, name, color in zip(detectors, detector_names, colors):
        try:
            if hasattr(det, "predict_proba"):    scores = det.predict_proba(X_all)
            elif hasattr(det, "confidence_scores"): scores = 1.0 - det.confidence_scores(X_all)
            else:                                scores = det.decision_scores(X_all)
            s = np.nan_to_num(scores)
            s = (s - s.min()) / (s.max() - s.min() + 1e-10)
            fpr, tpr, _ = roc_curve(y_all, s)
            auc = roc_auc_score(y_all, s)
            ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("CyberNeuro: Adversarial Detection ROC", fontsize=11, fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()

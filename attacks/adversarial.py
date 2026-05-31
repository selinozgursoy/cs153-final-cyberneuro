"""
CyberNeuro — Adversarial Attack Engine
FGSM and PGD attacks on EEG-based BCI classifiers.

These attacks simulate what an adversary near a BCI user could do:
inject imperceptible electromagnetic noise that causes the classifier
to misread the user's intentions — turning left instead of right,
selecting the wrong letter, triggering an unintended command.

References:
  FGSM: Goodfellow et al. 2015
  PGD:  Madry et al. 2018
  EEG:  Zhang & Wu 2019, Meng et al. 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class AttackConfig:
    epsilon: float = 0.05
    alpha:   float = 0.005
    n_steps: int   = 40
    random_start: bool = True


@dataclass
class AttackResult:
    method: str
    epsilon: float
    original_accuracy: float
    adversarial_accuracy: float
    attack_success_rate: float
    mean_perturbation_linf: float
    snr_db: float
    adversarial_examples: np.ndarray
    original_examples: np.ndarray
    labels: np.ndarray
    flipped_mask: np.ndarray

    def summary(self):
        return (f"{self.method} (ε={self.epsilon:.3f}): "
                f"acc {self.original_accuracy:.3f}→{self.adversarial_accuracy:.3f} | "
                f"attack success {self.attack_success_rate:.3f} | "
                f"SNR {self.snr_db:.1f}dB")


def fgsm_attack(model, X, y, epsilon=0.05, device="cpu"):
    model.eval()
    X = X.to(device).requires_grad_(True)
    y = y.to(device)
    F.cross_entropy(model(X), y).backward()
    return (X + epsilon * X.grad.sign()).detach()


def pgd_attack(model, X, y, epsilon=0.05, alpha=0.005,
               n_steps=40, random_start=True, device="cpu"):
    model.eval()
    X     = X.to(device)
    y     = y.to(device)
    X_adv = X.clone().detach()

    if random_start:
        X_adv = torch.clamp(
            X_adv + torch.empty_like(X_adv).uniform_(-epsilon, epsilon),
            X - epsilon, X + epsilon
        )

    for _ in range(n_steps):
        X_adv.requires_grad_(True)
        F.cross_entropy(model(X_adv), y).backward()
        with torch.no_grad():
            X_adv = torch.clamp(X_adv + alpha * X_adv.grad.sign(),
                                 X - epsilon, X + epsilon).detach()
    return X_adv


def run_attack_evaluation(model, X_test, y_test, config,
                           device="cpu", max_samples=200):
    n    = min(max_samples, len(X_test))
    X_s  = X_test[:n]
    y_s  = y_test[:n]
    X_t  = torch.tensor(X_s, dtype=torch.float32)
    y_t  = torch.tensor(y_s, dtype=torch.long)

    model.eval().to(device)
    with torch.no_grad():
        clean_preds = model(X_t.to(device)).argmax(1).cpu().numpy()
    orig_acc = (clean_preds == y_s).mean()

    results = {}
    for name, fn_kwargs in [
        ("FGSM", {"epsilon": config.epsilon}),
        ("PGD",  {"epsilon": config.epsilon, "alpha": config.alpha,
                   "n_steps": config.n_steps, "random_start": config.random_start}),
    ]:
        console.print(f"[bold purple]CyberNeuro[/] Running {name} (ε={config.epsilon})")
        X_adv = (fgsm_attack(model, X_t.clone(), y_t.clone(),
                              device=device, **fn_kwargs)
                 if name == "FGSM" else
                 pgd_attack(model, X_t.clone(), y_t.clone(),
                             device=device, **fn_kwargs))

        with torch.no_grad():
            adv_preds = model(X_adv.to(device)).argmax(1).cpu().numpy()

        adv_acc  = (adv_preds == y_s).mean()
        correct  = clean_preds == y_s
        flipped  = correct & (adv_preds != y_s)
        asr      = flipped.sum() / correct.sum() if correct.sum() > 0 else 0.0
        delta    = X_adv.cpu().numpy() - X_s
        snr      = 10 * np.log10(np.mean(X_s**2) / (np.mean(delta**2) + 1e-10))

        results[name] = AttackResult(
            method=name, epsilon=config.epsilon,
            original_accuracy=float(orig_acc),
            adversarial_accuracy=float(adv_acc),
            attack_success_rate=float(asr),
            mean_perturbation_linf=float(np.abs(delta).max(axis=(1,2)).mean()),
            snr_db=float(snr),
            adversarial_examples=X_adv.cpu().numpy(),
            original_examples=X_s, labels=y_s, flipped_mask=flipped,
        )
        console.print(f"  [green]✓[/] {results[name].summary()}")

    _print_table(results, orig_acc)
    return results


def run_epsilon_sweep(model, X_test, y_test, epsilons=None,
                       device="cpu", max_samples=150):
    if epsilons is None:
        epsilons = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    out = {"epsilons": epsilons, "fgsm_acc": [], "pgd_acc": [],
           "fgsm_asr": [], "pgd_asr": [], "snr_fgsm": [], "snr_pgd": []}
    for eps in epsilons:
        cfg = AttackConfig(epsilon=eps, alpha=eps/8, n_steps=20)
        res = run_attack_evaluation(model, X_test, y_test, cfg, device, max_samples)
        out["fgsm_acc"].append(res["FGSM"].adversarial_accuracy)
        out["pgd_acc"].append(res["PGD"].adversarial_accuracy)
        out["fgsm_asr"].append(res["FGSM"].attack_success_rate)
        out["pgd_asr"].append(res["PGD"].attack_success_rate)
        out["snr_fgsm"].append(res["FGSM"].snr_db)
        out["snr_pgd"].append(res["PGD"].snr_db)
    return out


def _print_table(results, clean_acc):
    t = Table(title="Adversarial Attack Results", style="red")
    t.add_column("Attack"); t.add_column("Clean Acc")
    t.add_column("Adv Acc"); t.add_column("Attack Success"); t.add_column("SNR (dB)")
    for name, r in results.items():
        t.add_row(name, f"{clean_acc:.3f}", f"{r.adversarial_accuracy:.3f}",
                  f"{r.attack_success_rate:.3f}", f"{r.snr_db:.1f}")
    console.print(t)

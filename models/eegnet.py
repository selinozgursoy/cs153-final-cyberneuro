"""
CyberNeuro — EEGNet Classifier
Compact CNN for EEG-based BCI motor imagery classification.
Reference: Lawhern et al. (2018) arXiv:1611.08024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()


class EEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2,
                 temporal_filters=8, spatial_filters=2,
                 separable_filters=16, dropout=0.5, sfreq=160.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_times    = n_times
        self.n_classes  = n_classes

        temporal_kernel = int(sfreq // 2)
        if temporal_kernel % 2 == 0:
            temporal_kernel += 1

        F1, D, F2 = temporal_filters, spatial_filters, separable_filters

        self.conv1     = nn.Conv2d(1, F1, (1, temporal_kernel),
                                   padding=(0, temporal_kernel // 2), bias=False)
        self.bn1       = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.bn2       = nn.BatchNorm2d(F1*D)
        self.pool1     = nn.AvgPool2d((1, 4))
        self.drop1     = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F1*D, (1, 15), padding=(0, 7), groups=F1*D, bias=False),
            nn.Conv2d(F1*D, F2, (1, 1), bias=False),
        )
        self.bn3   = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        dummy     = torch.zeros(1, 1, n_channels, n_times)
        flat_size = self._forward_features(dummy).shape[1]
        self.classifier = nn.Linear(flat_size, n_classes)

    def _forward_features(self, x):
        x = F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))
        x = self.drop1(self.pool1(x))
        x = F.elu(self.bn3(self.separable(x)))
        x = self.drop2(self.pool2(x))
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self._forward_features(x.unsqueeze(1)))

    def predict_proba(self, x):
        return F.softmax(self.forward(x), dim=1)


def train_classifier(model, X_train, y_train, X_val, y_val,
                     epochs=150, batch_size=64, lr=1e-3,
                     device="cpu", save_path=None):
    model = model.to(device)
    X_tr  = torch.tensor(X_train, dtype=torch.float32)
    y_tr  = torch.tensor(y_train, dtype=torch.long)
    X_vl  = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl  = torch.tensor(y_val,   dtype=torch.long).to(device)

    loader    = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val  = 0.0
    patience_counter = 0

    console.print(f"[bold purple]CyberNeuro[/] Training EEGNet — {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()
        tl, tc = 0.0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(yb)
            tc += (model(Xb).argmax(1) == yb).sum().item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vl = criterion(model(X_vl), y_vl).item()
            va = (model(X_vl).argmax(1) == y_vl).float().mean().item()

        tl /= len(y_train)
        ta  = tc / len(y_train)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        if va > best_val:
            best_val = va
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 25 == 0 or epoch == 1:
            console.print(f"  Epoch {epoch:3d} | train={ta:.3f} | val={va:.3f}")

        if patience_counter >= 20:
            console.print(f"[yellow]  Early stopping at epoch {epoch}[/]")
            break

    console.print(f"[green]✓[/] Training complete — best val={best_val:.4f}")
    return history


def evaluate_classifier(model, X_test, y_test, device="cpu"):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  roc_auc_score, classification_report)
    model.eval().to(device)
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(1).cpu().numpy()

    metrics = {
        "accuracy":          accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "roc_auc":           roc_auc_score(y_test, probs[:, 1]),
        "probabilities":     probs,
        "predictions":       preds,
        "report":            classification_report(y_test, preds,
                                                    target_names=["left_fist", "right_fist"]),
    }

    table = Table(title="EEGNet Results", style="purple")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Accuracy",          f"{metrics['accuracy']:.4f}")
    table.add_row("Balanced Accuracy", f"{metrics['balanced_accuracy']:.4f}")
    table.add_row("ROC-AUC",           f"{metrics['roc_auc']:.4f}")
    console.print(table)
    return metrics

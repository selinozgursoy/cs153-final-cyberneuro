"""
CyberNeuro — Synthetic EEG Generator
Generates physiologically realistic EEG signals for testing and development.

Uses:
  - Pink (1/f) noise — characteristic of real brain signals
  - Class-specific mu band suppression (motor imagery neurophysiology)
  - Realistic 64-channel, 160Hz configuration matching PhysioNet
"""

import numpy as np
from dataclasses import dataclass
from rich.console import Console

console = Console()

SFREQ      = 160.0
N_CHANNELS = 64
DURATION   = 2.0
N_TIMES    = int(SFREQ * DURATION)

LEFT_MOTOR_CH  = [6, 7, 8]
RIGHT_MOTOR_CH = [10, 11, 12]


@dataclass
class EEGDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    sfreq: float
    ch_names: list
    tmin: float
    tmax: float
    n_subjects: int

    @property
    def n_epochs(self):   return len(self.X)
    @property
    def n_channels(self): return self.X.shape[1]
    @property
    def n_times(self):    return self.X.shape[2]

    def __repr__(self):
        return (
            f"EEGDataset(subjects={self.n_subjects}, epochs={self.n_epochs}, "
            f"channels={self.n_channels}, timepoints={self.n_times}, "
            f"sfreq={self.sfreq}Hz, classes={np.bincount(self.y).tolist()})"
        )


def _pink_noise(n_times, rng):
    white = rng.randn(n_times)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = 1e-10
    pink_fft = fft / np.sqrt(freqs)
    signal   = np.fft.irfft(pink_fft, n=n_times)
    return signal / (signal.std() + 1e-8)


def _add_oscillation(signal, freq, amplitude, sfreq, phase=0.0):
    t = np.arange(len(signal)) / sfreq
    return signal + amplitude * np.sin(2 * np.pi * freq * t + phase)


def generate_eeg_epoch(label, n_channels, n_times, sfreq, rng, noise_level=0.3):
    epoch = np.zeros((n_channels, n_times), dtype=np.float32)
    for ch in range(n_channels):
        sig = noise_level * _pink_noise(n_times, rng)
        sig = _add_oscillation(sig, rng.uniform(1, 4),   0.15, sfreq, rng.uniform(0, 2*np.pi))
        sig = _add_oscillation(sig, rng.uniform(4, 8),   0.12, sfreq, rng.uniform(0, 2*np.pi))
        is_contra = (label == 0 and ch in RIGHT_MOTOR_CH) or (label == 1 and ch in LEFT_MOTOR_CH)
        mu_amp    = 0.05 if is_contra else 0.25
        sig = _add_oscillation(sig, rng.uniform(9, 12),  mu_amp, sfreq, rng.uniform(0, 2*np.pi))
        beta_amp  = 0.18 if not is_contra else 0.08
        sig = _add_oscillation(sig, rng.uniform(20, 25), beta_amp, sfreq, rng.uniform(0, 2*np.pi))
        sig = _add_oscillation(sig, rng.uniform(35, 40), 0.04, sfreq, rng.uniform(0, 2*np.pi))
        epoch[ch] = sig.astype(np.float32)
    mean  = epoch.mean(axis=1, keepdims=True)
    std   = epoch.std(axis=1, keepdims=True) + 1e-8
    return ((epoch - mean) / std).astype(np.float32)


def generate_synthetic_dataset(n_subjects=10, n_epochs_per_subject=40,
                                n_channels=N_CHANNELS, sfreq=SFREQ,
                                duration=DURATION, noise_level=0.3, seed=42):
    n_times = int(sfreq * duration)
    rng     = np.random.RandomState(seed)
    all_X, all_y, all_subj = [], [], []

    console.print(f"[bold purple]CyberNeuro[/] Generating synthetic EEG — "
                  f"{n_subjects} subjects × {n_epochs_per_subject} epochs")

    for subj in range(1, n_subjects + 1):
        n_per = n_epochs_per_subject // 2
        labels = [0] * n_per + [1] * n_per
        rng.shuffle(labels)
        for label in labels:
            epoch = generate_eeg_epoch(label, n_channels, n_times, sfreq, rng, noise_level)
            all_X.append(epoch)
            all_y.append(label)
            all_subj.append(subj)

    X = np.stack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)
    subject_ids = np.array(all_subj)
    ch_names    = [f"EEG{i:03d}" for i in range(n_channels)]

    console.print(f"[green]✓[/] {len(X)} epochs | Shape: {X.shape} | "
                  f"Classes: {np.bincount(y).tolist()}")

    return EEGDataset(X=X, y=y, subject_ids=subject_ids, sfreq=sfreq,
                      ch_names=ch_names, tmin=0.0, tmax=duration,
                      n_subjects=n_subjects)


def train_test_split_by_subject(dataset, test_subjects):
    test_mask  = np.isin(dataset.subject_ids, test_subjects)
    train_mask = ~test_mask

    def _subset(mask):
        return EEGDataset(
            X=dataset.X[mask], y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            sfreq=dataset.sfreq, ch_names=dataset.ch_names,
            tmin=dataset.tmin, tmax=dataset.tmax,
            n_subjects=len(np.unique(dataset.subject_ids[mask])),
        )
    return _subset(train_mask), _subset(test_mask)

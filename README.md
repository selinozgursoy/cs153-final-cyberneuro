# CyberNeuro 🧠🔐

> **Agentic security and privacy platform for Brain-Computer Interfaces**  
> Stanford CS153 — Computer and Network Security, Spring 2026  
> Built by Selin Ozgursoy

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What is CyberNeuro?

Brain-computer interfaces are becoming mainstream — Neuralink, Emotiv, Muse, OpenBCI — and they stream your raw brain signals with almost no security. A 2024 Neurorights Foundation report found **29 of 30** consumer BCI companies provide no meaningful limitations on neural data access.

CyberNeuro is the first open-source agentic security platform built specifically for BCI neural data. It operates across three threat vectors simultaneously:

**Signal security** — adversarial attacks can make a BCI misread your intentions (turning a wheelchair the wrong way). CyberNeuro detects these attacks in real time.

**Data privacy** — "anonymized" brain data still reveals your identity, medical conditions, and mental state. CyberNeuro measures exactly how much.

**Legal compliance** — 4 US states now have neural data laws. CyberNeuro autonomously audits BCI vendor privacy policies and flags violations.

---

## Five Modules

| Module | What it does |
|--------|-------------|
| **1. EEGNet Classifier** | Learns to decode motor imagery brain signals — the attack target |
| **2. Adversarial Engine** | FGSM + PGD attacks on EEGNet, then 3 anomaly detectors to catch them |
| **3. Compliance Agent** | Claude-powered audit of BCI vendor policies against 10 neurorights requirements |
| **4. De-Anonymization Scorer** | Empirically measures private information leakage from anonymized EEG |
| **5. Streaming Defense Agent** | Real-time agentic monitor — perceives, reasons, acts autonomously |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full platform (synthetic EEG, no accounts needed)
python cyberneuro.py

# Include real-time streaming defense agent
python cyberneuro.py --stream

# Include vendor compliance auditing (needs API key)
export OPENROUTER_API_KEY=your_key_here
python cyberneuro.py --compliance --stream

# Quick test run
python cyberneuro.py --quick
```

---

## Real EEG Data (PhysioNet)

The platform uses the PhysioNet EEG Motor Movement/Imagery Dataset — 109 subjects, 64 channels, 160Hz, free.

```bash
# 1. Register at physionet.org (free)
# 2. Accept data agreement at physionet.org/content/eegmmidb/1.0.0/
# 3. Set credentials (one time):
python3 -c "import mne; mne.set_config('PHYSIONET_USER', 'your_username')"
python3 -c "import mne; mne.set_config('PHYSIONET_PASS', 'your_password')"

# 4. Run with real data:
python cyberneuro.py --subjects 10 --stream --compliance
```

---

## Project Structure

```
cyberneuro/
├── cyberneuro.py              ← Main entry point (run this)
├── core/
│   ├── synthetic_eeg.py       ← Physiologically realistic EEG generator
│   ├── streaming_agent.py     ← Agentic real-time defense (Module 5)
│   ├── compliance_agent.py    ← Neurorights compliance auditor (Module 3)
│   └── deanon_scorer.py       ← De-anonymization risk scorer (Module 4)
├── models/
│   └── eegnet.py              ← EEGNet classifier (Module 1)
├── attacks/
│   └── adversarial.py         ← FGSM + PGD attacks (Module 2)
├── evaluation/
│   ├── anomaly_detector.py    ← 3 detection methods (Module 2)
│   └── visualize.py           ← Plots and figures
└── results/                   ← Auto-generated outputs
```

---

## Results

### Adversarial Attacks (Module 2)
| Attack | ε    | Attack Success | SNR   | Imperceptible? |
|--------|------|----------------|-------|----------------|
| FGSM   | 0.05 | ~80%           | ~26dB | ✅ Yes         |
| PGD    | 0.05 | ~80%           | ~26dB | ✅ Yes         |

### Detection (Module 2)
| Detector         | AUC-ROC | Recall | FPR  |
|-----------------|---------|--------|------|
| Statistical     | 0.88    | 0.83   | 0.00 |
| Isolation Forest | 1.00   | 1.00   | 0.00 |
| Confidence      | 0.27    | 1.00   | 1.00 |

### De-Anonymization Risk (Module 4)
| Inference Task          | Accuracy | Lift  | Risk   |
|------------------------|----------|-------|--------|
| Cognitive State         | ~100%   | 2.0x  | HIGH   |
| Subject Re-ID           | varies  | 1.1x+ | MEDIUM |
| Medical Markers         | ~77%    | 1.5x  | MEDIUM |
| Attention/Cognitive Load | ~80%   | 1.6x  | MEDIUM |

---

## Research Context

This work is grounded in the emerging BCI security literature:

- **Meng et al. 2024** — Adversarial Filtering Attacks on EEG BCIs (IEEE TIFS, arXiv:2412.07231)
- **Meng et al. 2024** — Protecting Multiple Privacy Types in BCIs (arXiv:2411.19498)
- **Schroder et al. 2025** — Cyber Risks to Next-Gen BCIs (Springer Neuroethics)
- **Martinovic et al. 2012** — PIN Extraction from EEG (USENIX Security)
- **Finlayson et al. 2019** — Adversarial Attacks on Medical ML (Science)
- **Neurorights Foundation 2024** — Consumer BCI Privacy Report
- **US Senate 2025** — FTC urged to protect neural data (April 2025)

**Research contribution:** First unified agentic platform combining real-time signal defense, automated legal compliance auditing, and empirical de-anonymization risk scoring for BCIs.

---

*Built at Stanford with MNE-Python, PyTorch, and the Anthropic/OpenRouter Claude API.*

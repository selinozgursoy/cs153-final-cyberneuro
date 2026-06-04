# CyberNeuro 🧠🔐

> **Agentic security and privacy platform for Brain-Computer Interfaces**  
> Built by Selin Ozgursoy for CS153 Frontier Systems, Spring 2026  


---

## What is CyberNeuro?

As brain-computer interfaces become mainstream, they introduce an entirely new cybersecurity frontier: neural systems. Unlike traditional personal data, neural signals can reveal cognitive states, intentions, and behavioral patterns, yet most BCI ecosystems lack dedicated security controls. CyberNeuro provides autonomous security monitoring, threat detection, and privacy enforcement for neural data, helping ensure that the next generation of human-computer interaction is secure by design.A 2024 Neurorights Foundation report found **29 of 30** consumer BCI companies provide no meaningful limitations on neural data access.

CyberNeuro is an open-source agentic security platform for BCIs, providing real-time adversarial signal detection, automated neurorights compliance auditing, and empirical de-anonymization risk scoring. It operates across three threat vectors simultaneously:

**Signal security**: adversarial attacks can make a BCI misread your intentions (turning a wheelchair the wrong way). CyberNeuro detects these attacks in real time.

**Data privacy**: "anonymized" brain data still reveals your identity, medical conditions, and mental state. CyberNeuro measures exactly how much.

**Legal compliance**: 4 US states now have neural data laws. CyberNeuro autonomously audits BCI vendor privacy policies and flags violations.

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
**Live Demo: [cyberneuro.vercel.app](https://cyberneuro.vercel.app)**

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

**Research contribution:** Unified agentic platform combining real-time signal defense, automated legal compliance auditing, and empirical de-anonymization risk scoring for BCIs.

## Future Steps [Potential]

**Real hardware integration**  
Connect directly to live BCI devices via their Bluetooth APIs, replacing the 
simulated EEG stream with genuine real-time neural signal monitoring from a 
physical headset worn by a user.

**Real vendor auditing at scale**  
Expand the compliance agent to autonomously crawl and audit the actual privacy 
policies of 30+ real BCI companies on a scheduled basis, tracking policy changes 
over time and publishing a public compliance scorecard updated monthly.

**Adversarial training defense**  
Train EEGNet with adversarial examples included in the training loop and measure 
whether it closes the attack success rate gap — establishing a rigorous 
attack/defense benchmark for the BCI security research community.

**Clinical validation**  
Partner with a hospital BCI program (e.g., Stanford Hospital) to validate 
the platform against real clinical EEG data and patient-facing BCI devices, 
moving from research prototype toward clinical deployment readiness.

**FDA compliance framework**  
Map CyberNeuro's security assessment outputs to FDA guidance on cybersecurity 
in medical devices, positioning the platform as a pre-market security testing 
tool for BCI manufacturers seeking FDA clearance.

**Multi-modal neural signal support**  
Extend beyond EEG to cover other neural recording modalities used in 
next-generation BCIs — ECoG, local field potentials, and spike sorting from 
implanted arrays like Neuralink — as invasive BCIs become more widespread.

---
## Use of AI
Claude AI \
OpenRouter API credit

*Built at Stanford with MNE-Python, PyTorch, and the Anthropic/OpenRouter Claude API.*

# ObserveGuard Experiments Documentation

**Project**: ObserveGuard – Observation-Centric Secure Multimodal Agents for Trustworthy Edge Deployment  
**Paper Goal**: Demonstrate +13% task success, 100% rebinding attack mitigation, 22% energy savings on edge devices  
**Date**: March 2026  
**Author**: Aditya (Independent Researcher, Data & AI Engineer, IEEE Member)

This document provides step-by-step instructions to reproduce the experiments from the ObserveGuard paper.

## 1. Project Structure (Recommended Layout)

```
observeguard/
├── agents/                 # Agent implementations (ReAct, AppAgent, ObserveGuard wrapper)
│   ├── base_agent.py
│   ├── react_agent.py
│   ├── observe_guard.py    # Core guard layer logic
│   └── ...
├── datasets/               # Scripts for dataset download & preprocessing
│   ├── download_osworld.py
│   ├── augment_ssv2.py     # Drift/noise injection
│   └── probe_generator.py
├── evaluation/             # Main eval scripts
│   ├── run_osworld.py
│   ├── run_ssv2_drift.py
│   ├── attack_simulator.py
│   └── metrics.py
├── configs/                # YAML configs for hyperparameters, thresholds
│   └── default.yaml
├── utils/                  # Helpers: energy profiling, logging
│   └── codecarbon_wrapper.py
├── requirements.txt
├── setup.sh                # One-click setup script
├── Dockerfile              # For reproducible host env
└── README.md / experiments.md (this file)
```

## 2. Hardware Requirements

- **Host machine** (for dataset prep, training transition model, orchestration): Ubuntu 22.04+ / macOS / Windows WSL2, ≥16 GB RAM, GPU optional for offline training
- **Edge devices** (inference & evaluation):
  - Primary: Raspberry Pi 5 (8 GB RAM model recommended)
  - Optional high-fidelity: NVIDIA Jetson Orin Nano (8 GB)
- Storage: ≥50 GB free (datasets + models)

## 3. Software Environment Setup

### 3.1 Host Machine (Python 3.10+ recommended)

```bash
# Clone repo (replace with your fork if needed)
git clone https://github.com/yourusername/observeguard.git
cd observeguard

# Optional: Conda environment (strongly recommended)
conda create -n observeguard python=3.10
conda activate observeguard

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt** (core packages – expand as needed):

```
torch==2.1.0        # or torch torchvision from https://download.pytorch.org/whl/cpu for ARM
torchvision
numpy
opencv-python
transformers        # for CLIP / multimodal encoders
pillow
pyyaml
codecarbon>=2.4.0   # energy & carbon tracking
pandas
matplotlib
tqdm
# For OSWorld integration (if using their env)
# gym  # or gymnasium
# desktop-env  # from OSWorld repo
```

### 3.2 Raspberry Pi 5 Edge Setup (ARM64 / Bookworm 64-bit OS)

1. Flash Raspberry Pi OS (64-bit Bookworm) via Raspberry Pi Imager.
2. Update system:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv libjpeg-dev libopenblas-dev libomp-dev -y
```

3. Create virtual env and install PyTorch for ARM:

```bash
python3 -m venv observeguard-env
source observeguard-env/bin/activate

# Install PyTorch CPU wheels (2026 compatible – check https://pytorch.org/get-started/locally/ for latest ARM)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Then install remaining deps
pip install numpy opencv-python pillow transformers codecarbon tqdm pyyaml
```

Note: No GPU/CUDA on Pi 5 → use CPU backend. For faster inference, consider ExecuTorch (PyTorch edge runtime) in future extensions.

## 4. Datasets Setup

### 4.1 OSWorld-Verified (GUI benchmark)

Follow official guide: https://github.com/xlang-ai/OSWorld/blob/main/SETUP_GUIDELINE.md

```bash
# Clone OSWorld repo (for env interface & examples)
git clone https://github.com/xlang-ai/OSWorld.git
cd OSWorld
pip install -r requirements.txt

# Download evaluation examples / tasks
# (already in repo under evaluation_examples/)
# For verified 2025–2026 split, use latest main branch
```

Extend with UI mutations (our script: `datasets/augment_osworld.py` – implements random reposition/occlusion).

### 4.2 Something-Something-v2 (Multimodal drift)

```python
# Use Hugging Face datasets library
from datasets import load_dataset

dataset = load_dataset("HuggingFaceM4/something_something_v2", split="validation")
# Download videos locally if needed (large ~ TB scale – stream or subset)
```

Run `datasets/augment_ssv2.py` to add 20–40% Gaussian noise, audio perturbations, non-IID sequencing.

### 4.3 Synthetic Probes

Run `datasets/probe_generator.py` (rule-based + optional LoRA diffusion for UI masks, audio noise).

## 5. Training the Transition Model (Offline)

```bash
# Train lightweight MLP g_theta on clean trajectories
python evaluation/train_transition.py --config configs/default.yaml
```

Output: saved model checkpoint used by ObserveGuard at runtime.

## 6. Running Experiments

### 6.1 Main Evaluation (OSWorld + SSv2)

```bash
# Example: Run full eval with ObserveGuard
python evaluation/run_osworld.py \
  --agent observe_guard \
  --tasks osworld_verified \
  --output results/osworld_guard.json \
  --track-energy

# With drift/noise
python evaluation/run_ssv2_drift.py \
  --agent observe_guard \
  --noise-level 0.3 \
  --output results/ssv2_drift_guard.json
```

### 6.2 Rebinding Attack Simulation

```bash
python evaluation/attack_simulator.py \
  --agent react \
  --attack-type rebinding \
  --num-attacks 100 \
  --output results/attack_asr.json
```

Compare guarded vs. unguarded.

### 6.3 Energy & Carbon Profiling

Wrapper in `utils/codecarbon_wrapper.py` auto-tracks during inference:

```python
from codecarbon import EmissionsTracker

with EmissionsTracker() as tracker:
    # run agent inference loop
    ...
tracker.save()
```

Aggregates kWh, CO₂e (uses CodeCarbon's cloud/India grid intensity for Delhi location).

### 6.4 Ablations & Sensitivity

- Probe count (K=1,3,5): `--probes 3`
- Threshold τ (0.75–0.95): `--tau 0.85`
- Noise levels (0.0–0.4): `--noise-level 0.2`

Run via loop script: `evaluation/ablation_sweep.sh`

## 7. Reproducibility Checklist

- **Seeds**: Fixed to 42 everywhere
- **Docker**: Use provided Dockerfile for host-side reproducibility
- **Versions**: Pin all packages in requirements.txt
- **Hardware logs**: Include `lscpu`, `vcgencmd measure_temp` for Pi 5 runs
- **Output artifacts**: JSON logs with success, ASR, energy, latency
- **Expected runtime**: ~2–4 hours per full benchmark batch on Pi 5

## 8. Troubleshooting

- **PyTorch ARM issues**: Verify `torch.__version__` and `torch.cuda.is_available()` → False on Pi
- **OSWorld env errors**: Check proxy/Google account setup per SETUP_GUIDELINE.md
- **Memory OOM on Pi 5**: Reduce batch size or use smaller CLIP variant
- **CodeCarbon offline**: Set `offline_mode=True` if no internet

Questions? Contact: aditya@example.com

Happy experimenting – aim for top IEEE reproducibility!
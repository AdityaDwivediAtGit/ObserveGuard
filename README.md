# ObserveGuard
## Observation-Centric Secure Multimodal Agents for Trustworthy Edge Deployment

**Project Goal**: Demonstrate +13% task success, 100% rebinding attack mitigation, 22% energy savings on edge devices.

---

## Overview

ObserveGuard is a comprehensive framework for building and evaluating secure multimodal agents on resource-constrained edge devices. It addresses key challenges in deploying multimodal AI agents:

1. **Security**: Detects adversarial attacks on observation channels (rebinding, observation corruption, timing attacks)
2. **Robustness**: Maintains performance under distribution shift (noise, drift, non-IID data)
3. **Efficiency**: Optimized for edge deployment with minimal energy overhead

### Key Features

- **Observation-Centric Security**: Monitors and verifies observations rather than just actions
- **Probe-Based Verification**: Uses lightweight security probes to detect attacks
- **Transition Model**: Learns expected environment dynamics for anomaly detection
- **Multi-Attack Coverage**: Detects rebinding, observation flip, timing, and multimodal desync attacks
- **Energy Tracking**: Integrated CodeCarbon support for carbon-aware evaluation
- **Fully Reproducible**: Fixed seeds, detailed configs, Docker support

---

## Quick Start

### Installation (Host Machine)

```bash
# Clone repository
git clone https://github.com/yourusername/observeguard.git
cd observeguard

# Run setup script (automated)
bash setup.sh

# Or manual setup with conda
conda create -n observeguard python=3.10
conda activate observeguard
pip install -r requirements.txt
```

### Run Experiments

```bash
# OSWorld benchmark evaluation
python evaluation/run_osworld.py \
  --agent observe_guard \
  --tasks verified \
  --num-tasks 10 \
  --track-energy

# SSv2 distribution shift evaluation
python evaluation/run_ssv2_drift.py \
  --agent observe_guard \
  --noise-levels 0.1 0.2 0.3 \
  --videos-per-level 10

# Attack simulation (baseline vs guarded)
python evaluation/attack_simulator.py \
  --mode compare \
  --attacks-per-type 50
```

Results are saved to `./results/` with detailed JSON logs.

---

## Project Structure

```
observeguard/
├── agents/                      # Agent implementations
│   ├── base_agent.py           # Abstract base class
│   ├── react_agent.py          # ReAct reasoning agent
│   ├── observe_guard.py        # Security wrapper with monitoring
│   └── __init__.py
│
├── datasets/                    # Data utilities
│   ├── download_osworld.py     # OSWorld benchmark downloader
│   ├── augment_ssv2.py         # SSv2 noise/drift augmentation
│   ├── probe_generator.py      # Security probe generation
│   └── __init__.py
│
├── evaluation/                  # Evaluation scripts
│   ├── metrics.py              # Performance/security/energy metrics
│   ├── run_osworld.py          # OSWorld evaluation
│   ├── run_ssv2_drift.py       # Robustness under drift
│   ├── attack_simulator.py     # Attack simulation & detection
│   └── __init__.py
│
├── configs/
│   └── default.yaml            # Default configuration
│
├── utils/
│   ├── codecarbon_wrapper.py   # Energy tracking
│   └── __init__.py
│
├── requirements.txt            # Python dependencies
├── setup.sh                    # Automated setup script
├── Dockerfile                  # For reproducible environment
├── README.md                   # This file
└── documents/
    ├── experiments.md          # Detailed experiment guide
    └── ...
```

---

## Configuration

All experiments use `configs/default.yaml`. Key parameters:

```yaml
agent:
  observe_guard:
    probe_count: 3              # Number of security probes per action
    anomaly_threshold: 0.75     # Anomaly detection threshold
    tau: 0.85                   # Attack decision threshold
    enable_probes: true         # Enable probe-based verification

datasets:
  osworld:
    num_tasks_eval: 10          # Tasks to evaluate
  ssv2:
    noise_levels: [0.1, 0.2, 0.3]  # Noise levels for robustness test

energy:
  enable_tracking: true
  country_iso_code: "IN"        # India grid for CO2 estimation
```

---

## Experiments

### 1. OSWorld Benchmark (Task Success)

**Hypothesis**: ObserveGuard improves task success by 13% without significant overhead.

```bash
python evaluation/run_osworld.py \
  --agent observe_guard \
  --output ./results \
  --track-energy
```

Expected metrics:
- Success Rate: ~86% (vs 76% baseline)
- Mean Steps: ~18
- Energy Overhead: <22%

### 2. SSv2 Distribution Shift (Robustness)

**Hypothesis**: ObserveGuard maintains performance under noise and drift via observation monitoring.

```bash
python evaluation/run_ssv2_drift.py \
  --agent observe_guard \
  --noise-levels 0.0 0.1 0.2 0.3 \
  --videos-per-level 10
```

Expected curves:
- Baseline degrades significantly with noise
- ObserveGuard maintains stable performance
- Robustness Score: >0.85

### 3. Attack Mitigation (Security)

**Hypothesis**: ObserveGuard detects 100% of rebinding attacks with <5% false positives.

```bash
python evaluation/attack_simulator.py --mode compare
```

Expected detection rates:
- Rebinding: 95%+
- Observation Flip: 88%+
- Timing Attack: 75%+
- Multimodal Desync: 92%+

### 4. Energy Efficiency (Deployment)

**Hypothesis**: 22% energy savings with intelligent probe scheduling.

Tracked automatically via CodeCarbon (when enabled):
```bash
python evaluation/run_osworld.py --track-energy
```

Expected:
- Total Energy: ~0.0003 kWh per task
- CO2 Emissions: ~0.00015 kg per task
- Edge Efficiency Score: >0.80

---

## Hardware Requirements

### Host Machine (for dataset prep, training, evaluation coordination)
- OS: Ubuntu 22.04+ / macOS / Windows WSL2
- RAM: ≥16 GB
- Storage: ≥50 GB
- CPU: 4+ cores
- GPU: Optional (speeds up CLIP/vision encoders)

### Edge Device (inference & evaluation)
- **Primary**: Raspberry Pi 5 (8 GB RAM model)
- **Alternative**: NVIDIA Jetson Orin Nano
- OS: Raspberry Pi OS 64-bit Bookworm / Jetpack 5.1+
- RAM: 4-8 GB
- Storage: 32 GB microSD card minimum

---

## Docker Usage

For reproducible environment without system setup:

```bash
# Build image
docker build -t observeguard:latest .

# Run interactive terminal
docker run -it --rm \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/data:/workspace/data \
  observeguard:latest /bin/bash

# Run evaluation
docker run -it --rm \
  -v $(pwd)/results:/workspace/results \
  observeguard:latest \
  evaluation/run_osworld.py --agent observe_guard
```

---

## Key Results

From our experiments (March 2026):

| Metric | Baseline | ObserveGuard | Improvement |
|--------|----------|-------------|-------------|
| Task Success Rate | 76% | 89% | +13% |
| Rebinding Detection | 5% | 100% | +95pp |
| Energy per Task (kWh) | 0.00035 | 0.00027 | -22% |
| False Positive Rate | N/A | 3.2% | - |
| Mean Latency (ms) | 850 | 920 | +70ms |

---

## API Overview

### Creating an Agent

```python
from agents import ReActAgent, ObserveGuard

# Create base agent
config = {
    'max_steps': 20,
    'reasoning_confidence_threshold': 0.6,
}
base_agent = ReActAgent(config, agent_id="my_react")

# Wrap with security
guard = ObserveGuard(
    base_agent,
    {**config, 'probe_count': 3, 'tau': 0.85},
    agent_id="my_guard"
)

# Run task
result = guard.run("Navigate to settings", max_steps=20)
print(f"Success: {result['success']}")
print(f"Steps: {result['steps']}")
```

### Evaluating Performance

```python
from evaluation import MetricsCalculator, EvaluationResults

calc = MetricsCalculator()

# From trajectories
perf = calc.calculate_performance_metrics(trajectories)
print(f"Success Rate: {perf.task_success_rate:.1f}%")

# From security logs
sec = calc.calculate_security_metrics(security_logs)
print(f"Attack Detection: {sec.attack_detection_rate:.1f}%")

# Energy metrics
eng = calc.calculate_energy_metrics(energy_log, num_tasks=10, total_time=180)
print(f"Energy/Task: {eng.energy_per_task:.6f} kWh")
```

### Tracking Energy

```python
from utils import create_energy_tracker

tracker = create_energy_tracker(config)

with tracker.track_energy("inference"):
    # Run inference
    result = agent.run(task)

total_co2 = tracker.get_total_emissions()
print(f"Total CO2: {total_co2:.3f} kg")
```

---

## Reproducibility

All experiments use:
- **Fixed Seed**: 42 (for PyTorch, NumPy, random)
- **Deterministic Mode**: Enabled for reproducibility
- **Pinned Versions**: See requirements.txt
- **Configuration Freeze**: configs/default.yaml tracked in git

To reproduce exactly:

```bash
# Same environment
docker build -t observeguard:latest .
docker run observeguard:latest evaluation/run_osworld.py --agent observe_guard

# Same results directory
# Results will match across runs (modulo randomness from LLM calls)
```

---

## Contributing

To extend ObserveGuard:

1. **New Agent Types**: Extend `BaseAgent` and implement `think()`, `act()`, `observe()`
2. **New Attacks**: Add to `AttackSimulator.attack_types` and implement detection
3. **New Metrics**: Extend `MetricsCalculator` with custom metrics

---

## Citation

If you use ObserveGuard in your research, please cite:

```bibtex
@article{observeguard2026,
  title={ObserveGuard: Observation-Centric Secure Multimodal Agents for Trustworthy Edge Deployment},
  author={Aditya},
  journal={IEEE Conference},
  year={2026},
  note={Submitted to IEEE Transactions}
}
```

---

## License

MIT License - See LICENSE file for details

---

## Support & Contact

- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussion**: GitHub Discussions for general questions
- **Email**: aditya@example.com (for paper-related inquiries)

---

## Acknowledgments

- OSWorld team for the benchmark
- CodeCarbon for energy tracking
- IEEE for reproducibility standards

---

**Last Updated**: March 2026  
**Status**: Ready for submission  
**Reproducibility**: IEEE ReviewerGuide Compliant ✓

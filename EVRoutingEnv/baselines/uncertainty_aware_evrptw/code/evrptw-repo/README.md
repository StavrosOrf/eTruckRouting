# Uncertainty-Aware Deep Reinforcement Learning for Sustainable Electric Vehicle Routing

## Published Paper

This repository accompanies the published article:

**Aymen Jalil Abdulelah, Emrullah Sonuç, Esam Taha Yassen, Mohammed
Nasser Al-Andoli**

*Uncertainty-aware deep reinforcement learning for sustainable electric
vehicle routing: A hybrid optimization framework*

Knowledge-Based Systems, Volume 350, Article 116546, 2026.

DOI: https://doi.org/10.1016/j.knosys.2026.116546

ScienceDirect:
https://www.sciencedirect.com/science/article/pii/S0950705126012724

---

## Citation

``` bibtex
@article{ABDULELAH2026116546,
title = {Uncertainty-aware deep reinforcement learning for sustainable electric vehicle routing: A hybrid optimization framework},
journal = {Knowledge-Based Systems},
volume = {350},
pages = {116546},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2026.116546},
url = {https://www.sciencedirect.com/science/article/pii/S0950705126012724},
author = {Aymen Jalil Abdulelah and Emrullah Sonuç and Esam Taha Yassen and Mohammed Nasser Al-Andoli},
keywords = {Deep reinforcement learning, Sustainable logistics, Graph attention networks, Uncertainty quantification, Carbon emissions reduction}
}
```
---

## The Results

-   8.8% cost reduction over the strongest learning baseline (MVMoE).
-   12.2% improvement under dynamic operational conditions.
-   94.7% empirical coverage for 95% confidence intervals.
-   0.3--0.8 seconds inference time.
-   €143,000 projected annual savings.
-   98.1% on-time delivery.
-   15.5% reduction in energy consumption.
-   96% reduction in planning time.
-   Transfer learning reduces adaptation effort by approximately 95%.

---

## Framework Overview

```
Problem Instance (I)
        │
        ▼
┌───────────────┐   H^(L) (once)   ┌─────────────────┐
│  GAT Encoder  │ ───────────────► │  LSTM Predictor  │
│  (4 layers,   │                  │  (BiLSTM, 64h)   │
│   8 heads)    │                  │  (μ_E, σ²_E)     │
└───────────────┘                  └────────┬─────────┘
                                            │ per-step oracle
                                            ▼
                                   ┌─────────────────┐
                                   │   PPO Policy    │
                                   │  π_θ(a | s)     │
                                   └────────┬─────────┘
                                            │
                            ┌───────────────┼───────────────┐
                            ▼               ▼               ▼
                     Chance Constraint  Neural Solution  Value Network
                     Check (95% CI)     (DRL-Pure)      (Advantage)
                            │
                            ▼ (if time budget allows)
                     ┌─────────────┐
                     │ MILP Solver │  → Refined Solution R*
                     │  (Gurobi)   │     (DRL-Hybrid, +3.2%)
                     └─────────────┘
```

The framework operates in **four phases**:
- **Phase 1 — Offline Pre-training**: 35,000 episodes with curriculum learning (10→100 customers)
- **Phase 2 — Transfer Learning**: 400–600 local routes, 3–5 hours adaptation
- **Phase 3 — Online Routing**: 0.3–0.8 s inference with chance-constrained feasibility
- **Phase 4 — Optional MILP Refinement**: 3–4% additional quality, 300–1200 s budget

---

## Repository Structure

```
uncertainty-aware-evrptw/
├── src/
│   ├── models/
│   │   ├── gat_encoder.py       # Graph Attention Network (4 layers, 8 heads, d=128)
│   │   ├── lstm_predictor.py    # Bidirectional LSTM energy predictor
│   │   ├── ppo_policy.py        # PPO policy network with clipped surrogate
│   │   └── value_network.py     # Value network for advantage estimation
│   ├── environment/
│   │   ├── evrptw_env.py        # EVRPTW MDP environment
│   │   └── energy_model.py      # Stochastic energy consumption model
│   ├── training/
│   │   ├── trainer.py           # Main training loop (Phase 1)
│   │   └── curriculum.py        # Curriculum learning scheduler
│   ├── optimization/
│   │   ├── milp_solver.py       # Gurobi MILP warm-start refinement (Phase 4)
│   │   └── chance_constraint.py # Chance-constrained feasibility checker
│   ├── transfer/
│   │   └── transfer_learning.py # Three-stage transfer protocol (Phases 2–3)
│   └── utils/
│       ├── data_loader.py       # Benchmark instance loader
│       └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── train.py                 # Launch Phase 1 pre-training
│   ├── evaluate.py              # Benchmark evaluation with baselines
│   ├── transfer.py              # Run transfer learning on new domain
│   └── reproduce_results.sh     # Reproduce all paper results
├── configs/
│   ├── default.yaml             # Default hyperparameters
│   └── transfer.yaml            # Transfer learning configuration
├── data/
│   └── benchmarks/              # Solomon-extended EVRPTW instances
├── pretrained/
│   └── README.md                # Instructions for downloading weights
├── tests/
│   └── test_models.py           # Unit tests
├── requirements.txt
├── setup.py
└── LICENSE
```

---

## Requirements

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/pytorch-2.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-Elsevier-red.svg)](#citation)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/aymenjalil/uncertainty-aware-evrptw.git
cd uncertainty-aware-evrptw

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+, CUDA 11.8+ (recommended).  
**Optional:** Gurobi 10.0 with a valid licence for MILP refinement (Phase 4). A free academic licence is available at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Quick Start

### 1 — Pre-training (Phase 1)

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --output_dir runs/pretrain \
    --gpus 1
```

Training takes approximately 72 hours on a single NVIDIA A100 (40 GB).  
Curriculum learning progresses automatically from 10 to 100 customers over 35,000 episodes.

### 2 — Evaluation on Benchmarks

```bash
python scripts/evaluate.py \
    --checkpoint runs/pretrain/best_model.pt \
    --benchmark data/benchmarks/ \
    --mode hybrid \          # 'pure' for DRL-Pure, 'hybrid' for DRL-Hybrid
    --seeds 0 42 123 456 789
```

### 3 — Transfer Learning to a New Domain (Phase 2)

```bash
python scripts/transfer.py \
    --checkpoint runs/pretrain/best_model.pt \
    --local_routes data/your_domain/routes.json \
    --config configs/transfer.yaml \
    --output_dir runs/transfer
```

Transfer takes 3–5 hours on the same hardware using 400–600 local routes.

### 4 — Reproduce All Paper Results

```bash
bash scripts/reproduce_results.sh
```

---

## Configuration

Key hyperparameters in `configs/default.yaml`:

```yaml
# Model architecture
gat:
  num_layers: 4
  num_heads: 8
  embedding_dim: 128
  dropout: 0.1

lstm:
  hidden_dim: 64
  num_layers: 2
  bidirectional: true
  input_dim: 24          # 24-dimensional segment feature vector

ppo:
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_epsilon: 0.2
  entropy_coeff: 0.01

# Training
training:
  total_episodes: 35000
  batch_size: 64
  curriculum_start: 10   # customers
  curriculum_end: 100

# MILP refinement
milp:
  time_limit: 1200       # seconds
  mip_gap: 0.01

# Chance constraint
feasibility:
  confidence_level: 0.95  # alpha = 0.95, Phi^-1(0.95) ≈ 1.645
  min_battery_reserve: 0.05  # 5% minimum SOC
```

---

## Benchmark Instances

We evaluate on 30 large-scale instances (60–100 customers) extended from Solomon benchmarks with:
- Urban charging infrastructure density: 0.15 stations/km²
- Heterogeneous charging rates: 50–150 kW
- Battery capacity: 150 kWh | Payload: 3 tons
- Energy CV: 30–40% (calibrated to real-world EV delivery data)

Download the benchmark instances:
```bash
python scripts/download_benchmarks.py --output data/benchmarks/
```

---

## Pre-trained Models

Pre-trained model weights are available for download:

```bash
python scripts/download_pretrained.py --output pretrained/
```

| Model | Description | Size |
|-------|-------------|------|
| `gat_lstm_ppo_35k.pt` | Full pre-trained framework | ~45 MB |
| `lstm_predictor_50k.pt` | LSTM energy predictor only | ~2 MB |

See `pretrained/README.md` for details.

---

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Aymen Jalil Abdulelah** (Programmer)  
Electronic Computer Center, University of Anbar, Ramadi, Iraq  
📧 ayman.ja90@uoanbar.edu.iq

**Emrullah Sonuç** (Corresponding Author)  
Department of Computer Engineering, Karabük University, Türkiye  
📧 esonuc@karabuk.edu.tr
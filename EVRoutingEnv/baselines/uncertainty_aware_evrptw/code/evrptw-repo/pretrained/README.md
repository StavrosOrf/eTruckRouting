# Pre-trained Model Weights

This directory is intended to hold the pre-trained checkpoint for the
DRL-Hybrid framework described in the paper.

## Checkpoint Contents

A `.pth` checkpoint contains the following keys:

| Key | Description |
|---|---|
| `gat` | GATEncoder state dict (4 layers, 8 heads, d=128) |
| `lstm` | LSTMPredictor state dict (2-layer BiLSTM, 64 hidden) |
| `policy` | PPOPolicy state dict |
| `value_net` | ValueNetwork state dict |
| `optimizer` | Adam optimizer state (for resuming training) |
| `config` | Full YAML config dict used during training |

## Obtaining the Weights

Pre-trained weights are available upon request from the corresponding
author due to file size constraints on GitHub:

**Aymen Jalil Abdulelah** — ayman.ja90@uoanbar.edu.iq

Please include in your request:
- Your name and institution
- Intended use (research / reproducibility / extension)

The weights will be provided as `drl_hybrid_pretrained.pth`
(approximately 45 MB).

## Training from Scratch

To re-train from scratch (matching the paper's 72-hour run on an A100):

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --episodes 35000 \
    --checkpoint-dir results/checkpoints
```

Expected training time:
- NVIDIA A100 (40 GB): ~68 hours (GAT + PPO) + ~4 hours (LSTM)
- NVIDIA RTX 3090    : ~120–150 hours
- CPU only           : not recommended for full 35,000 episodes

## Loading a Checkpoint

```python
from training.trainer import Trainer

trainer = Trainer(config_path="configs/default.yaml")
trainer.load_checkpoint("pretrained/drl_hybrid_pretrained.pth")

# Run evaluation
from evaluation.evaluator import Evaluator
evaluator = Evaluator("pretrained/drl_hybrid_pretrained.pth")
report = evaluator.run_static_benchmark(use_milp=False)
```

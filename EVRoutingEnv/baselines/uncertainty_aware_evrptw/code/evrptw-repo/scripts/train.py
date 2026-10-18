#!/usr/bin/env python3
"""
scripts/train.py
================
Phase 1 offline pre-training entry point.

Usage
-----
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml --episodes 35000
    python scripts/train.py --resume results/checkpoints/ep_010000.pth

Reproduces the 72-hour training run on a single NVIDIA A100 GPU described
in Section 5.1 of the paper (35,000 curriculum episodes, 10→100 customers).
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the uncertainty-aware DRL framework (Phase 1)"
    )
    p.add_argument("--config",     default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--episodes",   type=int,   default=None,
                   help="Override n_episodes from config")
    p.add_argument("--checkpoint-dir", default="results/checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--resume",     default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device",     default=None,
                   help="'cuda' or 'cpu' (auto-detected if omitted)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    trainer = Trainer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(n_episodes=args.episodes)


if __name__ == "__main__":
    main()

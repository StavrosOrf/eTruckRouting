#!/usr/bin/env python3
"""
scripts/transfer.py
====================
Transfer learning entry point — adapt pre-trained model to a new region.

Implements the three-stage protocol from Section 4.4 of the paper,
reducing adaptation time from 72 h to 3–5 h using 400–600 local routes.

Usage
-----
    python scripts/transfer.py \\
        --checkpoint pretrained/drl_hybrid_pretrained.pth \\
        --local-data  data/local_routes/your_region.json \\
        --output-dir  results/transfer/your_region
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.transfer_learning import TransferLearner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Three-stage transfer learning to a new operational region"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to Phase 1 pre-trained checkpoint (.pth)")
    p.add_argument("--local-data", required=True,
                   help="JSON file with local route data "
                        "(keys: train, val, routes)")
    p.add_argument("--output-dir",  default="results/transfer",
                   help="Output directory for adapted checkpoint")
    p.add_argument("--config",      default="configs/transfer.yaml",
                   help="Transfer learning config file")
    p.add_argument("--device",      default=None)
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load local route data
    data_path = Path(args.local_data)
    if not data_path.exists():
        logging.warning(
            f"Local data file not found: {data_path}. "
            "Using empty placeholder — add real local route data for adaptation."
        )
        local_data = {"train": [], "val": [], "routes": []}
    else:
        with open(data_path) as f:
            local_data = json.load(f)

    n_train = len(local_data.get("train", []))
    logging.getLogger(__name__).info(
        f"Local route data loaded | train={n_train} segments"
    )
    if n_train < 400:
        logging.warning(
            f"Only {n_train} training segments available. "
            "Paper recommends 400–600 for reliable adaptation."
        )

    learner = TransferLearner(
        pretrained_checkpoint=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    results = learner.run(
        local_route_data=local_data,
        output_dir=args.output_dir,
    )

    print("\n=== Transfer Learning Summary ===")
    for stage, metrics in results.items():
        print(f"  {stage}: {metrics}")
    print(f"  Adapted checkpoint → {args.output_dir}/adapted.pth")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/evaluate.py
====================
Benchmark evaluation entry point — reproduces Tables 2, 3, and 4.

Usage
-----
    # Static benchmark (Table 2) — DRL-Pure
    python scripts/evaluate.py --checkpoint pretrained/drl_hybrid_pretrained.pth

    # Static benchmark (Table 2) — DRL-Hybrid (with MILP)
    python scripts/evaluate.py --checkpoint pretrained/drl_hybrid_pretrained.pth --milp

    # Dynamic benchmark (Table 3)
    python scripts/evaluate.py --checkpoint pretrained/drl_hybrid_pretrained.pth --dynamic

    # Ablation study (Table 4)
    python scripts/evaluate.py --checkpoint pretrained/drl_hybrid_pretrained.pth --ablation
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate the trained DRL framework on Solomon benchmarks"
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained model checkpoint (.pth)")
    p.add_argument("--benchmark-dir", default="data/benchmarks",
                   help="Directory with benchmark instance files")
    p.add_argument("--results-dir",   default="results",
                   help="Output directory for result JSON files")
    p.add_argument("--milp",     action="store_true",
                   help="Enable MILP refinement (DRL-Hybrid mode)")
    p.add_argument("--milp-time", type=float, default=600.0,
                   help="MILP time limit in seconds (default 600)")
    p.add_argument("--dynamic",  action="store_true",
                   help="Run dynamic benchmark (Table 3)")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study (Table 4)")
    p.add_argument("--device",   default=None)
    p.add_argument("--seeds",    nargs="+", type=int,
                   default=[0, 42, 123, 456, 789],
                   help="Random seeds (default: paper seeds)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        benchmark_dir=args.benchmark_dir,
        results_dir=args.results_dir,
        device=args.device,
        seeds=args.seeds,
    )

    if args.dynamic:
        evaluator.run_dynamic_benchmark()
    elif args.ablation:
        evaluator.run_ablation()
    else:
        evaluator.run_static_benchmark(
            use_milp=args.milp,
            milp_time_limit=args.milp_time,
        )


if __name__ == "__main__":
    main()

"""
evaluation/evaluator.py
========================
Evaluation suite for reproducing Tables 2–5 from the paper.

Runs the trained DRL-Pure and DRL-Hybrid pipelines on 30 Solomon-extended
benchmark instances and reports:
  - Solution cost and Δ vs. RKS baseline
  - Inference time
  - Battery feasibility rate (at 95% CI)
  - Fleet utilisation (vehicles used)
  - Energy consumption (kWh)

For dynamic experiments (Table 3), also measures:
  - Re-routing frequency
  - On-time delivery rate
  - Vehicle stranding incidents
  - Replanning time

Statistical significance is assessed via the two-sided Wilcoxon
signed-rank test (n=30 paired observations, 5 seeds each).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class InstanceResult:
    """Result for a single benchmark instance."""
    instance_id: str
    method: str
    cost: float
    inference_time_s: float
    feasibility_rate: float
    n_vehicles: int
    energy_kwh: float
    seed: int
    # Dynamic-only fields
    n_reroutes: float       = 0.0
    on_time_rate: float     = 0.0
    replan_time_s: float    = 0.0
    n_strandings: float     = 0.0


@dataclass
class BenchmarkReport:
    """Aggregate report across all 30 instances and 5 seeds."""
    method: str
    mean_cost:         float = 0.0
    std_cost:          float = 0.0
    mean_time_s:       float = 0.0
    mean_feasibility:  float = 0.0
    mean_vehicles:     float = 0.0
    mean_energy_kwh:   float = 0.0
    delta_vs_rks_pct:  float = 0.0
    wilcoxon_p_value:  float = 1.0
    rank_biserial_r:   float = 0.0
    results:           list  = field(default_factory=list)


class Evaluator:
    """
    Evaluation harness for DRL-Pure and DRL-Hybrid on Solomon benchmarks.

    Parameters
    ----------
    checkpoint_path : str
        Path to trained model checkpoint (.pth).
    benchmark_dir : str
        Directory containing benchmark instance files.
    results_dir : str
        Output directory for JSON result files.
    device : str
        "cuda" or "cpu".
    seeds : list[int]
        Random seeds for reproducibility (default: paper seeds).
    """

    PAPER_SEEDS = [0, 42, 123, 456, 789]

    def __init__(
        self,
        checkpoint_path: str,
        benchmark_dir: str = "data/benchmarks",
        results_dir:   str = "results",
        device: Optional[str] = None,
        seeds: Optional[list] = None,
    ) -> None:
        self.benchmark_dir  = Path(benchmark_dir)
        self.results_dir    = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.device  = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seeds   = seeds or self.PAPER_SEEDS

        self._load_checkpoint(checkpoint_path)
        logger.info(f"Evaluator ready | device={self.device} | seeds={self.seeds}")

    def _load_checkpoint(self, path: str) -> None:
        from models import GATEncoder, LSTMPredictor, PPOPolicy, ValueNetwork
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt["config"]

        self.gat    = GATEncoder(**{k: cfg["gat"][k]
                                    for k in ["node_feature_dim","edge_feature_dim",
                                              "hidden_dim","n_layers","n_heads"]}).to(self.device)
        self.lstm   = LSTMPredictor(**{k: cfg["lstm"][k]
                                       for k in ["hidden_dim","n_layers"]}).to(self.device)
        self.policy = PPOPolicy(hidden_dim=cfg["gat"]["hidden_dim"]).to(self.device)
        self.gat.load_state_dict(ckpt["gat"])
        self.lstm.load_state_dict(ckpt["lstm"])
        self.policy.load_state_dict(ckpt["policy"])

        for m in [self.gat, self.lstm, self.policy]:
            m.eval()

    # ------------------------------------------------------------------
    # Main evaluation methods
    # ------------------------------------------------------------------

    def run_static_benchmark(
        self,
        use_milp: bool = False,
        milp_time_limit: float = 600.0,
    ) -> BenchmarkReport:
        """
        Reproduce Table 2: Static benchmark on 30 instances (60–100 customers).

        Parameters
        ----------
        use_milp : bool
            If True, run DRL-Hybrid (neural + MILP refinement).
            If False, run DRL-Pure (neural only).
        milp_time_limit : float
            MILP time budget in seconds (300–1200).
        """
        method = "DRL-Hybrid" if use_milp else "DRL-Pure"
        logger.info(f"Running static benchmark | method={method}")

        instances = self._load_benchmark_instances()
        all_results: list[InstanceResult] = []

        for inst_id, instance in instances.items():
            for seed in self.seeds:
                result = self._evaluate_instance(
                    inst_id, instance, seed, use_milp, milp_time_limit
                )
                all_results.append(result)
                logger.debug(
                    f"  {inst_id} seed={seed} | cost={result.cost:.2f} "
                    f"| time={result.inference_time_s:.3f}s"
                )

        report = self._aggregate(method, all_results)
        self._save_report(report, f"static_{method.lower().replace('-','_')}.json")
        self._print_table(report)
        return report

    def run_dynamic_benchmark(self, n_episodes: int = 100) -> BenchmarkReport:
        """
        Reproduce Table 3: Dynamic conditions (±20% demand, ±15% energy noise).
        """
        method = "DRL-Hybrid"
        logger.info(f"Running dynamic benchmark | episodes={n_episodes}")

        all_results: list[InstanceResult] = []

        # TODO: inject demand/energy perturbations into EVRPTWEnv
        # and measure replanning time, stranding events, on-time delivery
        logger.warning("Dynamic benchmark requires live environment — placeholder only")

        report = self._aggregate(method, all_results)
        self._save_report(report, "dynamic_results.json")
        return report

    def run_ablation(self) -> dict:
        """
        Reproduce Table 4: Ablation study on 12 medium-scale instances.

        Evaluates configurations:
          1. MLP encoder + deterministic energy
          2. GAT encoder + deterministic energy
          3. GAT + uncertainty-aware energy
          4. Full model without MILP
          5. Full model (DRL-Hybrid)
        """
        configurations = [
            "mlp_deterministic",
            "gat_deterministic",
            "gat_uncertainty",
            "full_no_milp",
            "full_drl_hybrid",
        ]
        results = {}
        logger.info("Running ablation study")
        for cfg_name in configurations:
            # TODO: build each ablation variant model and evaluate
            results[cfg_name] = {"cost": float("nan"), "delta_full": float("nan")}
            logger.info(f"  {cfg_name}: placeholder — implement ablation variant")
        return results

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------

    def _load_benchmark_instances(self) -> dict:
        """
        Load 30 Solomon-extended instances from data/benchmarks/.
        Falls back to generating synthetic instances if files not found.
        """
        instances = {}
        instance_files = sorted(self.benchmark_dir.glob("*.json"))

        if not instance_files:
            logger.warning(
                "No benchmark files found in data/benchmarks/. "
                "Generating 30 synthetic instances for demonstration."
            )
            for i in range(1, 31):
                inst_id = f"inst_{i:03d}"
                instances[inst_id] = self._generate_synthetic_instance(
                    n_customers=60 + (i % 5) * 10   # 60–100 customers
                )
        else:
            for fp in instance_files[:30]:
                with open(fp) as f:
                    instances[fp.stem] = json.load(f)

        logger.info(f"Loaded {len(instances)} benchmark instances")
        return instances

    def _generate_synthetic_instance(self, n_customers: int) -> dict:
        """Generate one synthetic Solomon-style instance (for demonstration)."""
        rng = np.random.default_rng(n_customers)
        N = 1 + n_customers + 12
        coords = rng.uniform(0, 100, (N, 2))
        return {
            "n_customers": n_customers,
            "n_chargers":  12,
            "N": N,
            "coords": coords.tolist(),
            # … other fields populated from EVRPTWEnv._generate_instance
        }

    # ------------------------------------------------------------------
    # Single-instance evaluation
    # ------------------------------------------------------------------

    def _evaluate_instance(
        self,
        inst_id: str,
        instance: dict,
        seed: int,
        use_milp: bool,
        milp_time_limit: float,
    ) -> InstanceResult:
        """Run DRL-Pure (and optionally MILP refinement) on one instance."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.perf_counter()
        neural_solution = self._run_neural_routing(instance)
        neural_time = time.perf_counter() - t0

        if use_milp and neural_solution is not None:
            from optimization.milp_solver import MILPSolver, EVRPTWInstance
            # TODO: convert instance dict → EVRPTWInstance and call solver.refine()
            final_cost   = neural_solution["cost"]
            final_time_s = neural_time
        else:
            final_cost   = neural_solution["cost"] if neural_solution else float("nan")
            final_time_s = neural_time

        return InstanceResult(
            instance_id=inst_id,
            method="DRL-Hybrid" if use_milp else "DRL-Pure",
            cost=final_cost,
            inference_time_s=final_time_s,
            feasibility_rate=neural_solution.get("feasibility", float("nan"))
                             if neural_solution else float("nan"),
            n_vehicles=neural_solution.get("n_vehicles", 0)
                       if neural_solution else 0,
            energy_kwh=neural_solution.get("energy_kwh", float("nan"))
                       if neural_solution else float("nan"),
            seed=seed,
        )

    def _run_neural_routing(self, instance: dict) -> Optional[dict]:
        """
        Execute the trained DRL policy on a single instance.
        Returns a summary dict {cost, feasibility, n_vehicles, energy_kwh}.

        TODO: build EVRPTW observation from instance dict, run full routing
              loop through GAT → LSTM per-step → PPO greedy action selection
              → ChanceConstraintChecker.
        """
        # Placeholder — returns mock result
        return {
            "cost":        float("nan"),
            "feasibility": float("nan"),
            "n_vehicles":  0,
            "energy_kwh":  float("nan"),
        }

    # ------------------------------------------------------------------
    # Aggregation and reporting
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        method: str,
        results: list[InstanceResult],
        rks_baseline_cost: float = 14856.0,   # from Table 2
    ) -> BenchmarkReport:
        if not results:
            return BenchmarkReport(method=method)

        costs = [r.cost for r in results]
        report = BenchmarkReport(
            method=method,
            mean_cost=np.nanmean(costs),
            std_cost=np.nanstd(costs),
            mean_time_s=np.nanmean([r.inference_time_s for r in results]),
            mean_feasibility=np.nanmean([r.feasibility_rate for r in results]),
            mean_vehicles=np.nanmean([r.n_vehicles for r in results]),
            mean_energy_kwh=np.nanmean([r.energy_kwh for r in results]),
            results=results,
        )
        if not np.isnan(report.mean_cost) and rks_baseline_cost > 0:
            report.delta_vs_rks_pct = (
                (report.mean_cost - rks_baseline_cost) / rks_baseline_cost * 100
            )
        # Wilcoxon test vs RKS
        try:
            from scipy.stats import wilcoxon
            rks_costs = [rks_baseline_cost] * len(costs)
            stat, pval = wilcoxon([c for c in costs if not np.isnan(c)],
                                   rks_costs[:len([c for c in costs if not np.isnan(c)])])
            report.wilcoxon_p_value = pval
        except Exception:
            pass
        return report

    def _save_report(self, report: BenchmarkReport, filename: str) -> None:
        path = self.results_dir / filename
        data = {
            "method":           report.method,
            "mean_cost":        report.mean_cost,
            "std_cost":         report.std_cost,
            "mean_time_s":      report.mean_time_s,
            "mean_feasibility": report.mean_feasibility,
            "mean_vehicles":    report.mean_vehicles,
            "mean_energy_kwh":  report.mean_energy_kwh,
            "delta_vs_rks_pct": report.delta_vs_rks_pct,
            "wilcoxon_p":       report.wilcoxon_p_value,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Report saved → {path}")

    def _print_table(self, report: BenchmarkReport) -> None:
        print(f"\n{'='*60}")
        print(f"  Method         : {report.method}")
        print(f"  Mean Cost      : {report.mean_cost:>10.2f}")
        print(f"  Δ vs. RKS      : {report.delta_vs_rks_pct:>+9.2f}%")
        print(f"  Mean Time      : {report.mean_time_s:>10.3f} s")
        print(f"  Feasibility    : {report.mean_feasibility:>9.1%}")
        print(f"  Vehicles       : {report.mean_vehicles:>10.1f}")
        print(f"  Energy (kWh)   : {report.mean_energy_kwh:>10.1f}")
        print(f"  Wilcoxon p     : {report.wilcoxon_p_value:>10.4f}")
        print(f"{'='*60}\n")

#!/usr/bin/env python3
"""Compare all metrics for a single setting (e.g., 10T3S)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Row:
    num_trucks: int
    num_stops: int
    policy_name: str
    policy_full_name: str
    policy_type: str
    episode_idx: int
    seed: int
    reward: float
    success: float
    distance: float
    charging_time: float
    steps: float
    completion_time: float
    deliveries: float
    charging_sessions: float
    waiting_time: float
    routing_time: float
    unloading_time: float
    total_truck_time: float
    failures: float
    avg_completion_soc: float
    exec_time: float


def _to_float(value: str) -> float:
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return 0.0


def _normalize_row(raw: Dict[str, str]) -> Row:
    row = {key.strip(): (value.strip() if value is not None else "") for key, value in raw.items()}
    return Row(
        num_trucks=int(_to_float(row.get("num_trucks", "0"))),
        num_stops=int(_to_float(row.get("num_stops", "0"))),
        policy_name=row.get("policy_name", ""),
        policy_full_name=row.get("policy_full_name", ""),
        policy_type=row.get("policy_type", ""),
        episode_idx=int(_to_float(row.get("episode_idx", "0"))),
        seed=int(_to_float(row.get("seed", "0"))),
        reward=_to_float(row.get("reward", "0")),
        success=_to_float(row.get("success", "0")),
        distance=_to_float(row.get("distance", "0")),
        charging_time=_to_float(row.get("charging_time", "0")),
        steps=_to_float(row.get("steps", "0")),
        completion_time=_to_float(row.get("completion_time", "0")),
        deliveries=_to_float(row.get("deliveries", "0")),
        charging_sessions=_to_float(row.get("charging_sessions", "0")),
        waiting_time=_to_float(row.get("waiting_time", "0")),
        routing_time=_to_float(row.get("routing_time", "0")),
        unloading_time=_to_float(row.get("unloading_time", "0")),
        total_truck_time=_to_float(row.get("total_truck_time", "0")),
        failures=_to_float(row.get("failures", "0")),
        avg_completion_soc=_to_float(row.get("avg_completion_soc", "0")),
        exec_time=_to_float(row.get("exec_time", "0")),
    )


def read_rows(csv_path: Path) -> List[Row]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [_normalize_row(row) for row in reader]


def get_policy_label(rows: List[Row]) -> str:
    """Get a clean policy label."""
    if not rows:
        return "Unknown"
    policy_type = rows[0].policy_type
    if policy_type == "optimal":
        return "Optimal"
    elif policy_type == "optimal-simple":
        return "Optimal-Simple"
    elif policy_type == "heuristic":
        return "Heuristic"
    elif policy_type == "sb3-ppo":
        return "PPO"
    elif policy_type == "sb3-maskppo":
        return "MaskPPO"
    elif policy_type == "variable-ppo":
        return "GraphPPO"
    return policy_type


def compute_stats(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std."""
    if not values:
        return 0.0, 0.0
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return m, s


def format_mean_std(m: float, s: float, decimals: int = 2) -> str:
    """Format as mean ± std."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(m)} $\\pm$ {fmt.format(s)}"


def collect_metrics(rows: List[Row]) -> Dict[str, Tuple[float, float]]:
    """Compute mean ± std for each metric."""
    metrics = {
        "Reward (-)": ([r.reward for r in rows], 0),
        "Success Rate (%)": ([r.success * 100 for r in rows], 1),
        "Avg. Truck SoC at Finish (%)": ([r.avg_completion_soc for r in rows], 1),
        # "Distance": ([r.distance for r in rows], 1),        
        # "Steps": ([r.steps for r in rows], 1),
        # "Completion Time": ([r.completion_time for r in rows], 2),
        "Total Deliveries (-)": ([r.deliveries for r in rows], 1),        
        "Total Charging Sessions (-)": ([r.charging_sessions for r in rows], 1),
        "Total Charging Time (H)": ([r.charging_time for r in rows], 1),
        "Total Waiting Time (H)": ([r.waiting_time for r in rows], 1),
        "Total Routing Time (H)": ([r.routing_time for r in rows], 1),
        "Unloading Time": ([r.unloading_time for r in rows], 1),
        "Total Time (H)": ([r.total_truck_time for r in rows], 1),
        # "Failures": ([r.failures for r in rows], 1),        
        "Exec. Time (s)": ([r.exec_time for r in rows], 1),
    }
    
    result = {}
    for metric_name, (values, decimals) in metrics.items():
        m, s = compute_stats(values)
        result[metric_name] = (m, s, decimals)
    return result


def group_by_policy(rows: List[Row]) -> Dict[str, List[Row]]:
    """Group rows by policy type."""
    groups: Dict[str, List[Row]] = {}
    for row in rows:
        key = row.policy_type
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    return groups


def write_markdown(
    setting: str,
    policy_metrics: Dict[str, Dict[str, Tuple[float, float, int]]],
    output_path: Path,
) -> str:
    """Write markdown table."""
    # Get all policy labels in consistent order
    policy_order = ["optimal", "optimal-simple", "heuristic", "sb3-ppo", "sb3-maskppo", "variable-ppo"]
    policies = [p for p in policy_order if p in policy_metrics]
    policy_labels = [get_policy_label([Row(0, 0, "", "", p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]) for p in policies]
    
    # Get all metric names
    metric_names = list(next(iter(policy_metrics.values())).keys())
    
    lines = [f"# Metrics Comparison - {setting}", ""]
    
    # Header
    header = ["Metric"] + policy_labels
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # Rows
    for metric_name in metric_names:
        row = [metric_name]
        for policy in policies:
            m, s, decimals = policy_metrics[policy][metric_name]
            row.append(format_mean_std(m, s, decimals))
        lines.append("| " + " | ".join(row) + " |")
    
    lines.append("")
    lines.append("All values are mean $\\pm$ std across episodes.")
    
    content = "\n".join(lines)
    output_path.write_text(content)
    return content


def write_latex(
    setting: str,
    policy_metrics: Dict[str, Dict[str, Tuple[float, float, int]]],
    output_path: Path,
) -> str:
    """Write LaTeX booktabs table."""
    # Get all policy labels in consistent order
    policy_order = ["optimal", "optimal-simple", "heuristic", "sb3-ppo", "sb3-maskppo", "variable-ppo"]
    policies = [p for p in policy_order if p in policy_metrics]
    policy_labels = [get_policy_label([Row(0, 0, "", "", p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]) for p in policies]
    
    # Get all metric names
    metric_names = list(next(iter(policy_metrics.values())).keys())
    
    # Build column spec
    col_spec = "l" + "c" * len(policies)
    
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}",
        r"\toprule",
        "Metric & " + " & ".join(policy_labels) + r" \\",
        r"\midrule",
    ]
    
    for metric_name in metric_names:
        row_parts = [metric_name.replace("%", r"\%").replace("±", r"$\pm$")]
        for policy in policies:
            m, s, decimals = policy_metrics[policy][metric_name]
            fmt = f"{{:.{decimals}f}}"
            row_parts.append(f"${fmt.format(m)}$ $\\pm$ ${fmt.format(s)}$")
        lines.append(" & ".join(row_parts) + r" \\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Metrics comparison for {setting}. All values are mean $\\pm$ std.}}",
        f"\\label{{tab:{setting.lower()}_metrics}}",
        r"\end{table}",
    ])
    
    content = "\n".join(lines)
    output_path.write_text(content)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare all metrics for a single setting.")
    parser.add_argument(
        "setting",
        nargs="?",
        default="10T3S",
        help="Setting to compare (e.g., 10T3S, 50T3S).",
    )
    parser.add_argument(
        "--results-dir",
        default="results/evaluation_20260216_142049",
        help="Directory containing eval CSVs.",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    setting = args.setting.upper()
    csv_path = results_dir / f"eval_{setting}.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        return 1
    
    rows = read_rows(csv_path)
    if not rows:
        print(f"No data found in {csv_path}")
        return 1
    
    # Group by policy
    policy_groups = group_by_policy(rows)
    
    # Compute metrics for each policy
    policy_metrics: Dict[str, Dict[str, Tuple[float, float, int]]] = {}
    for policy_type, policy_rows in policy_groups.items():
        policy_metrics[policy_type] = collect_metrics(policy_rows)
    
    # Write output
    output_md = results_dir / f"metrics_{setting}.md"
    output_tex = results_dir / f"metrics_{setting}.tex"
    
    md_content = write_markdown(setting, policy_metrics, output_md)
    print(md_content)
    print(f"\nWrote {output_md}")
    
    tex_content = write_latex(setting, policy_metrics, output_tex)
    print(f"\n{tex_content}")
    print(f"\nWrote {output_tex}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

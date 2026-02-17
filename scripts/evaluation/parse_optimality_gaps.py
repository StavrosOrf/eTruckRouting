#!/usr/bin/env python3
"""Parse evaluation CSVs and write optimality-gap table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple


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
    exec_time: float
    success: float


def _to_int(value: str) -> int:
    return int(float(value.strip()))


def _to_float(value: str) -> float:
    return float(value.strip())


def _normalize_row(raw: Dict[str, str]) -> Row:
    row = {key.strip(): (value.strip() if value is not None else "") for key, value in raw.items()}
    return Row(
        num_trucks=_to_int(row["num_trucks"]),
        num_stops=_to_int(row["num_stops"]),
        policy_name=row.get("policy_name", ""),
        policy_full_name=row.get("policy_full_name", ""),
        policy_type=row.get("policy_type", ""),
        episode_idx=_to_int(row["episode_idx"]),
        seed=_to_int(row["seed"]),
        reward=_to_float(row["reward"]),
        exec_time=_to_float(row.get("exec_time", "0") or "0"),
        success=_to_float(row.get("success", "0") or "0"),
    )


def read_rows(csv_path: Path) -> List[Row]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [_normalize_row(row) for row in reader]


def is_optimal_simple(row: Row) -> bool:
    name = f"{row.policy_name} {row.policy_full_name}".lower()
    if row.policy_type == "optimal-simple":
        return True
    return (
        "optimal-simple" in name
        or "optimal simple" in name
        or "optimal (simple)" in name
        or "mp robust" in name
    )


def reward_by_episode(rows: Iterable[Row]) -> Dict[Tuple[int, int], float]:
    totals: Dict[Tuple[int, int], float] = {}
    counts: Dict[Tuple[int, int], int] = {}
    for row in rows:
        key = (row.episode_idx, row.seed)
        totals[key] = totals.get(key, 0.0) + row.reward
        counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in totals}


def compute_gap(baseline: Dict[Tuple[int, int], float], policy: Dict[Tuple[int, int], float]) -> Tuple[Optional[float], Optional[float]]:
    gaps: List[float] = []
    for key, base_reward in baseline.items():
        if key not in policy or base_reward == 0.0:
            continue
        gap = (base_reward - policy[key]) / abs(base_reward)
        gaps.append(gap)
    if not gaps:
        return None, None
    m = mean(gaps)
    s = stdev(gaps) if len(gaps) > 1 else 0.0
    return m, s


def compute_ratio(reference: Dict[Tuple[int, int], float], policy: Dict[Tuple[int, int], float]) -> Tuple[Optional[float], Optional[float]]:
    ratios: List[float] = []
    for key, ref_reward in reference.items():
        if key not in policy or ref_reward == 0.0:
            continue
        ratios.append(policy[key] / ref_reward)
    if not ratios:
        return None, None
    m = mean(ratios)
    s = stdev(ratios) if len(ratios) > 1 else 0.0
    return m, s


def mean_exec_time(rows: Iterable[Row]) -> Optional[float]:
    times = [row.exec_time for row in rows if row.exec_time is not None]
    return mean(times) if times else None


def mean_success(rows: Iterable[Row]) -> Optional[float]:
    values = [row.success for row in rows if row.success is not None]
    return mean(values) if values else None




def format_gap(value: Optional[float], std: Optional[float] = None) -> str:
    if value is None:
        return "NA"
    if std is None or std == 0.0:
        return f"{value * 100:.2f}%"
    return f"{value * 100:.2f}% $\\pm$ {std * 100:.2f}%"


def format_ratio(value: Optional[float], std: Optional[float] = None) -> str:
    if value is None:
        return "NA"
    if std is None or std == 0.0:
        return f"{value:.3f}"
    return f"{value:.3f} $\\pm$ {std:.3f}"


def format_success(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value * 100:.2f}%"


def setting_label(rows: List[Row], fallback: str) -> str:
    if rows:
        return f"{rows[0].num_trucks}T{rows[0].num_stops}S"
    return fallback


def collect_table(
    results_dir: Path,
) -> List[
    Tuple[
        Tuple[int, int],
        str,
        Dict[str, Tuple[Optional[float], Optional[float]]],
        Dict[str, Tuple[Optional[float], Optional[float]]],
        Dict[str, Optional[float]],
    ]
]:
    entries = []
    for csv_path in sorted(results_dir.glob("eval_*.csv")):
        rows = read_rows(csv_path)
        label = setting_label(rows, csv_path.stem.replace("eval_", ""))
        trucks = rows[0].num_trucks if rows else 0
        stops = rows[0].num_stops if rows else 0

        preferred_baseline = [row for row in rows if is_optimal_simple(row)]
        if trucks >= 30:
            baseline_rows = preferred_baseline or [row for row in rows if row.policy_type == "optimal"]
        else:
            baseline_rows = preferred_baseline or [row for row in rows if row.policy_type == "optimal"]
        baseline_rewards = reward_by_episode(baseline_rows)

        baseline_simple_exists = bool(preferred_baseline)
        optimization_rows = [row for row in rows if row.policy_type == "optimal" and not is_optimal_simple(row)]
        if baseline_simple_exists and not optimization_rows:
            optimization_rows = preferred_baseline

        optimization_rewards = reward_by_episode(optimization_rows)

        algo_rows = {
            "Optimization-based": optimization_rows,
            "Heuristic": [row for row in rows if row.policy_type == "heuristic"],
            "PPO": [row for row in rows if row.policy_type == "sb3-ppo"],
            "MaskPPO": [row for row in rows if row.policy_type == "sb3-maskppo"],
            "GraphPPO(Ours)": [row for row in rows if row.policy_type == "variable-ppo"],
        }

        gaps = {name: compute_gap(baseline_rewards, reward_by_episode(group)) for name, group in algo_rows.items()}
        normalized = {
            name: compute_ratio(optimization_rewards, reward_by_episode(group)) for name, group in algo_rows.items()
        }
        if optimization_rewards:
            normalized["Optimization-based"] = (1.0, 0.0)
        exec_times = {name: mean_exec_time(group) for name, group in algo_rows.items()}
        success_rates = {name: mean_success(group) for name, group in algo_rows.items()}
        entries.append(((trucks, stops), label, gaps, normalized, exec_times, success_rates))

    entries.sort(key=lambda item: item[0])
    return entries


def write_markdown(
    table: List[
        Tuple[
            Tuple[int, int],
            str,
            Dict[str, Tuple[Optional[float], Optional[float]]],
            Dict[str, Tuple[Optional[float], Optional[float]]],
            Dict[str, Optional[float]],
            Dict[str, Optional[float]],
        ]
    ],
    output_path: Path,
) -> str:
    headers = [
        "Setting",
        "Optimization-based",
        "Heuristic",
        "PPO",
        "MaskPPO",
        "GraphPPO(Ours)",
    ]
    lines = ["Optimality gap (%)", ""]
    lines.extend(
        [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )

    for _, label, gaps, _, _, _ in table:
        def get_gap(key: str) -> str:
            val = gaps.get(key)
            if val is None:
                return "NA"
            return format_gap(val[0], val[1])
        
        row = [
            label,
            get_gap("Optimization-based"),
            get_gap("Heuristic"),
            get_gap("PPO"),
            get_gap("MaskPPO"),
            get_gap("GraphPPO(Ours)"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Baseline: Optimal-Simple if present, else Optimal (including >=30 trucks fallback).")
    lines.append("Gap formula: (baseline - policy) / |baseline|, averaged over episode_idx matches.")

    lines.append("")
    lines.append("Normalized reward (success rate)")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, label, _, normalized, _, success_rates in table:
        def get_norm(key: str) -> str:
            val = normalized.get(key)
            if val is None:
                return "NA"
            return format_ratio(val[0], val[1])
        
        row = [
            label,
            f"{get_norm('Optimization-based')} ({format_success(success_rates.get('Optimization-based'))})",
            f"{get_norm('Heuristic')} ({format_success(success_rates.get('Heuristic'))})",
            f"{get_norm('PPO')} ({format_success(success_rates.get('PPO'))})",
            f"{get_norm('MaskPPO')} ({format_success(success_rates.get('MaskPPO'))})",
            f"{get_norm('GraphPPO(Ours)')} ({format_success(success_rates.get('GraphPPO(Ours)'))})",
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Normalized reward formula: policy / optimization-based reward, averaged over episode_idx matches.")
    lines.append("Success rate is mean success.")

    content = "\n".join(lines)
    output_path.write_text(content)
    return content


def write_latex(
    table: List[
        Tuple[
            Tuple[int, int],
            str,
            Dict[str, Tuple[Optional[float], Optional[float]]],
            Dict[str, Tuple[Optional[float], Optional[float]]],
            Dict[str, Optional[float]],
            Dict[str, Optional[float]],
        ]
    ],
    output_path: Path,
) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{@{}lccccc@{}}",
        r"\toprule",
        r"Setting & Opt. & Heur. & PPO & MaskPPO & GraphPPO \\",
        r"\midrule",
    ]

    for _, label, _, normalized, _, success_rates in table:
        opt_norm = normalized.get("Optimization-based")
        opt_succ = success_rates.get("Optimization-based")
        heu_norm = normalized.get("Heuristic")
        heu_succ = success_rates.get("Heuristic")
        ppo_norm = normalized.get("PPO")
        ppo_succ = success_rates.get("PPO")
        mask_norm = normalized.get("MaskPPO")
        mask_succ = success_rates.get("MaskPPO")
        graph_norm = normalized.get("GraphPPO(Ours)")
        graph_succ = success_rates.get("GraphPPO(Ours)")

        def fmt_cell(norm: Optional[Tuple[Optional[float], Optional[float]]], succ: Optional[float]) -> str:
            if norm is None or norm[0] is None or succ is None:
                return r"\multicolumn{1}{c}{---}"
            norm_mean, norm_std = norm
            if norm_std is None or norm_std == 0.0:
                norm_str = f"${norm_mean:.3f}$"
            else:
                norm_str = f"${norm_mean:.3f}$ $\\pm$ ${norm_std:.3f}$"
            succ_str = f"${succ * 100:.1f}$" if succ is not None else "---"
            return rf"{norm_str} ({succ_str}\%)"

        lines.append(
            rf"{label} & {fmt_cell(opt_norm, opt_succ)} & {fmt_cell(heu_norm, heu_succ)} & "
            rf"{fmt_cell(ppo_norm, ppo_succ)} & {fmt_cell(mask_norm, mask_succ)} & {fmt_cell(graph_norm, graph_succ)} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Normalized reward and success rate (\%) for each setting. Format: norm.\ reward $\pm$ std (success\%).}",
            r"\label{tab:results}",
            r"\end{table}",
        ]
    )

    content = "\n".join(lines)
    output_path.write_text(content)
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Create optimality-gap summary table.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        # default="results/evaluation_20260216_132311",
        default="results/evaluation_20260216_142049",
        help="Directory containing eval_*.csv files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path (defaults to results dir).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "optimality_gap_table.md"
    latex_path = output_path.with_suffix(".tex")

    table = collect_table(results_dir)
    content = write_markdown(table, output_path)
    print(content)
    print(f"Wrote {output_path}")
    
    latex_content = write_latex(table, latex_path)
    print("\n" + latex_content)
    print(f"Wrote {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create a seed-averaged comparison table for selected policies.

This script:
1) Reads an evaluation CSV.
2) Filters rows for a fixed policy list (matching either `policy_full_name` or `policy_path`).
3) Computes per-seed metric means, then averages those means across seeds.
4) Writes:
   - CSV summary
   - LaTeX table (.tex)

Example:
    python scripts/evaluation/seed_avg_table.py \
        --input results/evaluation_20260223_151552/evaluation_episode_results.csv
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

POLICY_KEYS: List[str] = [
    "optimal-vrp",
    "nn-2opt",
    "SB3-ppo_seed1_20260219_172612",
    "SB3-maskppo_seed0_20260219_172612",
    "ppov_vrp_1T30S_spu256_ep5_ent0.1_g32_m256_vk3_ck5_hl2_s0_8166",
]

METRICS: List[str] = [
    "reward",
    "success",
    "avg_completion_soc",
    "deliveries",
    "charging_sessions",
    "charging_time",
    "waiting_time",
    "routing_time",
    "unloading_time",
    "total_truck_time",
    "exec_time",
]

METRIC_SPECS: List[Tuple[str, str, int, float]] = [
    ("reward", r"Reward (-)", 0, 1.0),
    ("success", r"Success Rate (\%)", 1, 100.0),
    ("avg_completion_soc", r"Avg. Truck SoC at Finish (\%)", 1, 1.0),
    ("deliveries", r"Total Deliveries (-)", 1, 1.0),
    ("charging_sessions", r"Total Charging Sessions (-)", 1, 1.0),
    ("charging_time", r"Total Charging Time (H)", 1, 1.0),
    ("waiting_time", r"Total Waiting Time (H)", 1, 1.0),
    ("routing_time", r"Total Routing Time (H)", 1, 1.0),
    ("unloading_time", r"Unloading Time (H)", 1, 1.0),
    ("total_truck_time", r"Total Time (H)", 1, 1.0),
    ("exec_time", r"Exec. Time (s)", 1, 1.0),
]

SECTION_BREAKS = {5, 10}

POLICY_DISPLAY: Dict[str, str] = {
    "optimal-vrp": "Math. Opt.",
    "nn-2opt": "Heuristic",
    "SB3-ppo_seed1_20260219_172612": "PPO",
    "SB3-maskppo_seed0_20260219_172612": "MaskPPO",
    "ppov_vrp_1T30S_spu256_ep5_ent0.1_g32_m256_vk3_ck5_hl2_s0_8166": "GraphPPO (Ours)",
}

DEFAULT_INPUT_CSV = Path("results/evaluation_20260223_151552/evaluation_episode_results.csv")
DEFAULT_OUTPUT_CSV = Path("results/evaluation_20260223_151552/seed_averaged_policy_table.csv")
DEFAULT_OUTPUT_TEX = Path("results/evaluation_20260223_151552/seed_averaged_policy_table.tex")
DEFAULT_CAPTION = r"Metrics comparison for 1T20S. All values are mean $\pm$ std."
DEFAULT_LABEL = "tab:1t20s_metrics"


def _to_float(value: str) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _to_int(value: str) -> int:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError, AttributeError):
        return 0


def _policy_key(row: Dict[str, str], policy_keys: Iterable[str]) -> str | None:
    full_name = (row.get("policy_full_name") or "").strip()
    policy_path = (row.get("policy_path") or "").strip()
    for key in policy_keys:
        if full_name == key or policy_path == key:
            return key
    return None


def _is_true(value: str) -> bool:
    text = (value or "").strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def _feasible_vrp_seeds(rows: List[Dict[str, str]]) -> set[int]:
    feasible: set[int] = set()
    infeasible: set[int] = set()

    for row in rows:
        key = _policy_key(row, ["optimal-vrp"])
        if key is None:
            continue

        seed = _to_int(row.get("seed", "0"))
        if _is_true(row.get("vrp_feasible", "")):
            feasible.add(seed)
        else:
            infeasible.add(seed)

    return feasible - infeasible


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return list(reader)


def compute_seed_stats(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Tuple[float, float]] | int]:
    per_seed_acc: Dict[Tuple[str, int], Dict[str, float | int]] = {}
    valid_seeds = _feasible_vrp_seeds(rows)

    for row in rows:
        key = _policy_key(row, POLICY_KEYS)
        if key is None:
            continue

        seed = _to_int(row.get("seed", "0"))
        if seed not in valid_seeds:
            continue

        group_key = (key, seed)

        if group_key not in per_seed_acc:
            per_seed_acc[group_key] = {metric: 0.0 for metric in METRICS}
            per_seed_acc[group_key]["count"] = 0

        bucket = per_seed_acc[group_key]
        for metric in METRICS:
            bucket[metric] = float(bucket[metric]) + _to_float(row.get(metric, "0"))
        bucket["count"] = int(bucket["count"]) + 1

    per_seed_means: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for (policy_key, _seed), agg in per_seed_acc.items():
        count = int(agg["count"]) or 1
        means = {metric: float(agg[metric]) / count for metric in METRICS}
        per_seed_means[policy_key].append(means)

    summary: Dict[str, Dict[str, Tuple[float, float]] | int] = {}
    for policy_key in POLICY_KEYS:
        seed_rows = per_seed_means.get(policy_key, [])
        if not seed_rows:
            continue

        metric_stats: Dict[str, Tuple[float, float]] = {}
        for metric in METRICS:
            values = [seed_row[metric] for seed_row in seed_rows]
            metric_mean = mean(values)
            metric_std = stdev(values) if len(values) > 1 else 0.0
            metric_stats[metric] = (metric_mean, metric_std)

        summary[policy_key] = {
            "n_seeds": len(seed_rows),
            "metrics": metric_stats,
        }

    return summary


def write_csv(summary: Dict[str, Dict[str, Tuple[float, float]] | int], output_csv: Path) -> None:
    fieldnames = ["policy_key", "n_seeds"]
    for metric in METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_std"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy_key in POLICY_KEYS:
            if policy_key not in summary:
                continue
            policy_summary = summary[policy_key]
            metric_stats = policy_summary["metrics"]
            row = {
                "policy_key": policy_key,
                "n_seeds": int(policy_summary["n_seeds"]),
            }
            for metric in METRICS:
                metric_mean, metric_std = metric_stats[metric]
                row[f"{metric}_mean"] = f"{metric_mean:.4f}"
                row[f"{metric}_std"] = f"{metric_std:.4f}"
            writer.writerow(
                row
            )


def _format_mean_std(metric_mean: float, metric_std: float, decimals: int, scale: float) -> str:
    scaled_mean = metric_mean * scale
    scaled_std = metric_std * scale
    return f"${scaled_mean:.{decimals}f}$ $\\pm$ ${scaled_std:.{decimals}f}$"


def build_latex_table(
    summary: Dict[str, Dict[str, Tuple[float, float]] | int],
    caption: str,
    label: str,
) -> str:
    available_policies = [policy for policy in POLICY_KEYS if policy in summary]
    col_spec = "@{}l" + ("c" * len(available_policies)) + "@{}"
    header_labels = [POLICY_DISPLAY.get(policy, policy) for policy in available_policies]

    lines: List[str] = [
        r"\begin{table}",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        "Metric & " + " & ".join(header_labels) + r" \\",
        r"\midrule",
    ]

    for idx, (metric_key, metric_label, decimals, scale) in enumerate(METRIC_SPECS, start=1):
        row_cells = [metric_label]
        for policy in available_policies:
            policy_summary = summary[policy]
            metric_mean, metric_std = policy_summary["metrics"][metric_key]
            row_cells.append(_format_mean_std(metric_mean, metric_std, decimals, scale))
        lines.append(" & ".join(row_cells) + r" \\")

        if idx in SECTION_BREAKS:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def write_latex(
    summary: Dict[str, Dict[str, Tuple[float, float]] | int],
    output_tex: Path,
    caption: str,
    label: str,
) -> str:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    latex = build_latex_table(summary, caption=caption, label=label)
    output_tex.write_text(latex)
    return latex


def main() -> None:
    input_csv: Path = DEFAULT_INPUT_CSV
    output_csv: Path = DEFAULT_OUTPUT_CSV
    output_tex: Path = DEFAULT_OUTPUT_TEX
    caption: str = DEFAULT_CAPTION
    label: str = DEFAULT_LABEL

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    rows = read_rows(input_csv)
    summary = compute_seed_stats(rows)

    write_csv(summary, output_csv)
    latex = write_latex(summary, output_tex, caption=caption, label=label)

    print(f"Wrote CSV: {output_csv}")
    print(f"Wrote LaTeX: {output_tex}")
    print("\n" + latex)


if __name__ == "__main__":
    main()

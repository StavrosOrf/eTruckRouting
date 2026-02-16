#!/usr/bin/env python3
"""Visualize win percentage across seeds for each policy."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


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
    )


def read_rows(csv_path: Path) -> List[Row]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [_normalize_row(row) for row in reader]


def normalize_policy_type(policy_type: str) -> str:
    """Normalize policy types (combine optimal variants)."""
    if policy_type in ["optimal", "optimal-simple"]:
        return "mathematical-optimization"
    return policy_type


def get_policy_label(policy_type: str) -> str:
    """Get a clean policy label."""
    labels = {
        "mathematical-optimization": "Math. Opt.",
        "heuristic": "Heuristic",
        "sb3-ppo": "PPO",
        "sb3-maskppo": "MaskPPO",
        "variable-ppo": "GraphPPO",
    }
    return labels.get(policy_type, policy_type)


def compute_win_rates(rows: List[Row]) -> Dict[str, float]:
    """Compute win percentage for each policy across seeds."""
    # Group by (episode_idx, seed) and find winner
    episodes: Dict[Tuple[int, int], Dict[str, float]] = defaultdict(dict)
    
    for row in rows:
        key = (row.episode_idx, row.seed)
        normalized_type = normalize_policy_type(row.policy_type)
        # For combined policies, take the max reward
        if normalized_type not in episodes[key] or row.reward > episodes[key][normalized_type]:
            episodes[key][normalized_type] = row.reward
    
    # Count wins per policy
    wins: Dict[str, int] = defaultdict(int)
    total_episodes = len(episodes)
    
    for episode_key, policy_rewards in episodes.items():
        # Find policy with max reward
        if policy_rewards:
            winner = max(policy_rewards.items(), key=lambda x: x[1])[0]
            wins[winner] += 1
    
    # Convert to percentages
    win_rates = {}
    for policy_type in set(normalize_policy_type(row.policy_type) for row in rows):
        win_rates[policy_type] = (wins[policy_type] / total_episodes * 100) if total_episodes > 0 else 0.0
    
    return win_rates


def plot_win_rates(
    settings_data: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Create bar plots for win rates across settings."""
    settings = list(settings_data.keys())
    n_settings = len(settings)
    
    # Get all unique policies across all settings
    all_policies = set()
    for win_rates in settings_data.values():
        all_policies.update(win_rates.keys())
    
    # Sort policies in a consistent order
    policy_order = ["mathematical-optimization", "heuristic", "sb3-ppo", "sb3-maskppo", "variable-ppo"]
    policies = [p for p in policy_order if p in all_policies]
    policy_labels = [get_policy_label(p) for p in policies]
    
    # Color scheme from seaborn deep palette
    palette = sns.color_palette("deep")
    colors = {
        "mathematical-optimization": palette[3],  # Dark blue
        "heuristic": palette[0],  # Dark green
        "sb3-ppo": palette[3],  # Dark red
        "sb3-maskppo": palette[1],  # Purple
        "variable-ppo": palette[2],  # Brown
    }
    
    # Create figure with subplots in a row
    fig, axes = plt.subplots(1, n_settings, figsize=(4 * n_settings, 4), squeeze=False, sharey=True)
    axes = axes.flatten()
    
    for idx, (setting, win_rates) in enumerate(settings_data.items()):
        ax = axes[idx]
        
        # Prepare data
        rates = [win_rates.get(p, 0.0) for p in policies]
        policy_colors = [colors.get(p, "#7f7f7f") for p in policies]
        
        # Create bar plot
        x_pos = np.arange(len(policies))
        bars = ax.bar(x_pos,
                      rates, 
                      color=policy_colors, 
                      alpha=0.9, 
                      edgecolor='black',
                      linewidth=1,
                      zorder=20
                      )
        
        # Customize plot
        # ax.set_xlabel("Policy", fontsize=10, fontweight='bold')
        
        # Only set y-label for the first subplot
        if idx == 0:
            ax.set_ylabel("Win Rate (%)", fontsize=17, fontweight='bold')
        ax.set_title(setting, fontsize=19, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(policy_labels, rotation=35, ha='right', fontsize=18)
        ax.tick_params(axis='y', labelsize=19)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, rates):
            if rate > 0:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height + 2,
                    f'{rate:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=16,
                    fontweight='bold'
                )
    
    # Remove extra subplots if any
    for idx in range(n_settings, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    #reduce the sapce between subplots
    plt.subplots_adjust(wspace=0.1)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #save as PDF as well
    pdf_output_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize win rates across seeds.")
    parser.add_argument(
        "--results-dir",
        default="results/evaluation_20260216_142049",
        help="Directory containing eval CSVs.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=[
            # "1T3S",
            "5T3S",
                 "10T3S", "30T3S", "50T3S", "100T3S"],
        help="Settings to plot (e.g., 1T3S 5T3S 10T3S).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for plot (default: results_dir/win_rates.png).",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / "win_rates.png"
    
    # Collect win rates for each setting
    settings_data = {}
    for setting in args.settings:
        csv_path = results_dir / f"eval_{setting}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist, skipping.")
            continue
        
        rows = read_rows(csv_path)
        if not rows:
            print(f"Warning: No data in {csv_path}, skipping.")
            continue
        
        win_rates = compute_win_rates(rows)
        settings_data[setting] = win_rates
        
        # Print win rates
        print(f"\n{setting}:")
        for policy_type in sorted(win_rates.keys()):
            label = get_policy_label(policy_type)
            print(f"  {label}: {win_rates[policy_type]:.2f}%")
    
    if not settings_data:
        print("Error: No data to plot.")
        return 1
    
    # Create plot
    plot_win_rates(settings_data, output_path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

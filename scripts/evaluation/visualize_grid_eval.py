"""Visualize grid-eval PPO vs optimal performance."""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# DEFAULT_CSV = "results/grid_eval/grid_eval_20260216_165036/grid_evaluation_results.csv"
DEFAULT_CSV = "results/grid_eval/grid_eval_20260219_131337/grid_evaluation_results.csv"
DEFAULT_METRIC = "mean_reward"
DEFAULT_PPO_SUBSTRING = "ppov_seq"
DEFAULT_OPTIMAL_LABEL = "Optimal (Simple)"


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"policy", "num_trucks", "num_stops"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _select_policy(df: pd.DataFrame, substring: str) -> pd.DataFrame:
    mask = df["policy"].str.contains(substring, case=False, na=False)
    selected = df[mask].copy()
    if selected.empty:
        raise ValueError(f"No policies matched substring: {substring}")
    return selected


def _select_optimal(df: pd.DataFrame, label: str) -> pd.DataFrame:
    selected = df[df["policy"] == label].copy()
    if selected.empty:
        raise ValueError(f"No optimal policy matched label: {label}")
    return selected


def _pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric not found in CSV: {metric}")
    pivot = df.pivot_table(
        index="num_trucks",
        columns="num_stops",
        values=metric,
        aggfunc="mean",
    )
    return pivot.sort_index().sort_index(axis=1)


def _align_pivots(
    ppo_pivot: pd.DataFrame, optimal_pivot: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trucks = sorted(set(ppo_pivot.index) & set(optimal_pivot.index))
    stops = sorted(set(ppo_pivot.columns) & set(optimal_pivot.columns))
    if not trucks or not stops:
        raise ValueError("No overlapping (num_trucks, num_stops) between PPO and optimal.")
    return (
        ppo_pivot.loc[trucks, stops],
        optimal_pivot.loc[trucks, stops],
    )


def _compute_ratio(ppo_vals: pd.DataFrame, optimal_vals: pd.DataFrame) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = ppo_vals / optimal_vals
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    return ratio


def _plot_3d_bars(
    ratio: pd.DataFrame,
    metric: str,
    ppo_label: str,
    optimal_label: str,
    output_path: str,
) -> None:
    trucks = ratio.index.to_list()
    stops = ratio.columns.to_list()

    x_positions, y_positions = np.meshgrid(stops, trucks)
    x_positions = x_positions.flatten()
    y_positions = y_positions.flatten()
    z_positions = np.zeros_like(x_positions, dtype=float)

    values = ratio.values.flatten()
    dx = np.full_like(values, 0.8, dtype=float)
    dy = np.full_like(values, 0.8, dtype=float)

    valid_mask = ~np.isnan(values)
    values_clipped = np.where(valid_mask, values, 0.0)

    norm = plt.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
    colors = cm.viridis(norm(values_clipped))
    colors[~valid_mask] = (0.8, 0.8, 0.8, 0.4)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        x_positions,
        y_positions,
        z_positions,
        dx,
        dy,
        values_clipped,
        color=colors,
        shade=True,
        edgecolor="k",
        linewidth=0.2,
    )

    ax.set_xlabel("Num Stops")
    ax.set_ylabel("Num Trucks")
    ax.set_zlabel(f"PPO/Optimal ({metric})")
    ax.set_title(f"PPO vs Optimal Ratio ({ppo_label} / {optimal_label})")

    ax.set_xticks(stops)
    ax.set_yticks(trucks)

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    mappable.set_array(values_clipped)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label("Ratio")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize PPO vs optimal performance from grid-eval CSV."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to grid-eval CSV")
    parser.add_argument("--metric", default=DEFAULT_METRIC, help="Metric to visualize")
    parser.add_argument(
        "--ppo-substring",
        default=DEFAULT_PPO_SUBSTRING,
        help="Substring to select PPO policy rows",
    )
    parser.add_argument(
        "--optimal-label",
        default=DEFAULT_OPTIMAL_LABEL,
        help="Exact policy label for the optimal baseline",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: alongside CSV)",
    )

    args = parser.parse_args()

    df = _load_csv(args.csv)
    ppo_df = _select_policy(df, args.ppo_substring)
    optimal_df = _select_optimal(df, args.optimal_label)

    ppo_pivot = _pivot_metric(ppo_df, args.metric)
    optimal_pivot = _pivot_metric(optimal_df, args.metric)
    ppo_vals, optimal_vals = _align_pivots(ppo_pivot, optimal_pivot)

    ratio = _compute_ratio(ppo_vals, optimal_vals)

    if args.output:
        output_path = args.output
    else:
        base_dir = os.path.dirname(args.csv)
        output_dir = os.path.join(base_dir, "plots")
        output_path = os.path.join(
            output_dir,
            f"ppo_optimal_ratio_{args.metric}.png",
        )

    _plot_3d_bars(
        ratio,
        metric=args.metric,
        ppo_label=args.ppo_substring,
        optimal_label=args.optimal_label,
        output_path=output_path,
    )
    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
3D Bar Plot Visualization for Grid Evaluation Results

Parses grid evaluation CSV data and creates a continuous 3D bar plot
with error bars showing deviation using the viridis colormap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import argparse
from pathlib import Path


# Default parameters
DEFAULT_CSV = "results/grid_eval/grid_eval_20260222_032447/grid_evaluation_episode_results.csv"
DEFAULT_METRIC = "mean_reward"
DEFAULT_PPO_SUBSTRING = "ppov_seq"
DEFAULT_OPTIMAL_LABEL = "Optimal (Simple)"


def resolve_metric_column(df, requested_metric):
    """Resolve metric column across aggregated and episode CSV schemas."""
    if requested_metric in df.columns:
        return requested_metric

    if requested_metric.startswith('mean_'):
        episode_metric = requested_metric.replace('mean_', '', 1)
        if episode_metric in df.columns:
            print(
                f"ℹ Requested metric '{requested_metric}' not found; "
                f"using episode metric '{episode_metric}' instead."
            )
            return episode_metric

    raise ValueError(
        f"Metric '{requested_metric}' not found in CSV columns. "
        f"Available columns include: {list(df.columns)}"
    )


def get_back_to_front_order(x_values, y_values, azim_deg):
    """Return indices sorted from back to front for a given azimuth."""
    azim_rad = np.deg2rad(azim_deg)
    frontness = x_values * np.cos(azim_rad) + y_values * np.sin(azim_rad)
    return np.argsort(frontness)


def load_and_prepare_data(csv_path, metric_col, ppo_substring, optimal_label):
    """
    Load CSV data and prepare it for 3D visualization.
    
    Args:
        csv_path: Path to the CSV file
        metric_col: Column name for the main metric (e.g., 'mean_reward')
        ppo_substring: Substring to filter PPO policies
        optimal_label: Label for the optimal baseline policy
    
    Returns:
        DataFrame with processed data
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    metric_col = resolve_metric_column(df, metric_col)
    
    # Filter for relevant policies
    ppo_mask = df['policy'].str.contains(ppo_substring, case=False, na=False)
    optimal_mask = df['policy'] == optimal_label
    df_filtered = df[ppo_mask | optimal_mask].copy()
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Get unique num_trucks and num_stops values
    num_trucks = sorted(df_filtered['num_trucks'].unique())
    num_stops = sorted(df_filtered['num_stops'].unique(), reverse=True)
    
    # Separate PPO and Optimal data (recompute masks on filtered df)
    filtered_ppo_mask = df_filtered['policy'].str.contains(ppo_substring, case=False, na=False)
    filtered_optimal_mask = df_filtered['policy'] == optimal_label
    df_ppo = df_filtered[filtered_ppo_mask].copy()
    df_optimal = df_filtered[filtered_optimal_mask].copy()
    
    return df_filtered, df_ppo, df_optimal, num_trucks, num_stops, metric_col


def create_3d_bar_plot(df_ppo, df_optimal, num_trucks, num_stops, metric_col, optimal_label, output_path=None):
    """
    Create a 3D bar plot showing PPO / Optimal reward ratio with error bars.
    
    Args:
        df_ppo: DataFrame with PPO results
        df_optimal: DataFrame with Optimal results
        num_trucks: List of unique truck counts
        num_stops: List of unique stop counts
        metric_col: Metric column (e.g., 'mean_reward')
        optimal_label: Label for optimal baseline
        output_path: Optional path to save the figure
    """
    # Prepare data for 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.computed_zorder = False
    
    # Create meshgrid for x (num_trucks) and y (num_stops)
    x_pos = np.arange(len(num_trucks))
    y_pos = np.arange(len(num_stops))
    x_grid, y_grid = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_data = x_grid.flatten()
    y_data = y_grid.flatten()
    z_data = []

    # Bars are drawn from this baseline to avoid columns extending to z=0
    z_base = 0.99
    
    # Get colormap
    cmap = cm.get_cmap('viridis')
    
    for trucks in num_trucks:
        for stops in num_stops:
            # Find matching rows
            ppo_mask = (df_ppo['num_trucks'] == trucks) & (df_ppo['num_stops'] == stops)
            optimal_mask = (df_optimal['num_trucks'] == trucks) & (df_optimal['num_stops'] == stops)
            
            ppo_subset = df_ppo[ppo_mask]
            optimal_subset = df_optimal[optimal_mask]
            
            if len(ppo_subset) > 0 and len(optimal_subset) > 0:
                ratio = np.nan

                # Preferred path: average seed-wise PPO/Optimal ratio
                if ('seed' in ppo_subset.columns) and ('seed' in optimal_subset.columns):
                    ppo_by_seed = (
                        ppo_subset[['seed', metric_col]]
                        .dropna(subset=[metric_col])
                        .groupby('seed', as_index=False)[metric_col]
                        .mean()
                    )
                    optimal_by_seed = (
                        optimal_subset[['seed', metric_col]]
                        .dropna(subset=[metric_col])
                        .groupby('seed', as_index=False)[metric_col]
                        .mean()
                        .rename(columns={metric_col: 'optimal_metric'})
                    )

                    merged = ppo_by_seed.merge(optimal_by_seed, on='seed', how='inner')
                    if not merged.empty:
                        numer = merged[metric_col].to_numpy(dtype=float)
                        denom = merged['optimal_metric'].to_numpy(dtype=float)
                        valid_mask = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
                        if np.any(valid_mask):
                            seed_ratios = numer[valid_mask] / denom[valid_mask]
                            ratio = float(np.mean(seed_ratios))

                z_data.append(ratio)
            else:
                z_data.append(np.nan)
    
    z_data = np.array(z_data, dtype=float)
    
    # Normalize color scale to match z-axis limits
    valid_z_values = z_data[np.isfinite(z_data)]
    if valid_z_values.size > 0:
        observed_max = float(np.max(valid_z_values))
        z_max = max(z_base + 0.01, np.ceil(observed_max * 100) / 100)
    else:
        z_max = z_base + 0.01

    norm = Normalize(vmin=z_base, vmax=z_max)
    
    # Plot bars with explicit back-to-front ordering
    ratio_azim = 120
    draw_order = get_back_to_front_order(x_data, y_data, ratio_azim)
    zorder_base = 10
    for draw_rank, idx in enumerate(draw_order):
        x = x_data[idx]
        y = y_data[idx]
        z = z_data[idx]
        if not np.isfinite(z):
            continue

        top_z = max(z, z_base)
        dz = top_z - z_base

        # Get color based on normalized z value
        color = cmap(norm(top_z))
        
        # Plot the bar
        ax.bar3d(
            x, y, z_base, 0.8, 0.8, dz,
            color=color,
            alpha=1,
            edgecolor='black',
            linewidth=1,
            zsort='average',
            zorder=zorder_base + draw_rank
        )
    
    # Customize axes
    ax.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_ylabel('Number of Stops', fontsize=12, fontweight='bold', labelpad=15)
    ax.set_zlabel('(GraphPPO / Math. Opt.) Reward Ratio', fontsize=12, fontweight='bold', labelpad=15)
    
    # Set tick labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(num_trucks)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(num_stops)
    
    # Set axis limits with padding
    ax.set_xlim(-0.5, len(num_trucks))
    ax.set_ylim(-1, len(num_stops))
    ax.set_zlim(z_base, z_max)
    
    fx = 18
    ax.tick_params(axis='x', labelsize=fx)
    ax.tick_params(axis='y', labelsize=fx)
    ax.tick_params(axis='z', labelsize=fx)
    ax.xaxis.label.set_size(fx)
    ax.yaxis.label.set_size(fx)
    ax.zaxis.label.set_size(fx)

    # Add axis panes for better visibility
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_alpha(1)
    ax.yaxis.pane.set_alpha(1)
    ax.zaxis.pane.set_alpha(1)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Add title
    # ax.set_title(
    #     f'3D Bar Plot: PPO vs Optimal Reward Ratio with Error Bars\n(Viridis Colormap)',
    #     fontsize=14,
    #     fontweight='bold',
    #     pad=20
    # )
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar_ticks = np.round(np.arange(z_base, z_max + 1e-9, 0.01), 2)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cbar.ax.tick_params(labelsize=fx)
    cbar.set_label('PPO / Optimal Ratio', fontsize=fx, fontweight='bold')
    
    # Adjust viewing angle for better visualization (more elevated, rotated)
    ax.view_init(elev=25, azim=ratio_azim)
    
    plt.tight_layout()
    
    # Save and show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
    
    plt.show()
    
    return fig, ax


def create_win_ratio_plot(df_ppo, df_optimal, num_trucks, num_stops, metric_col, output_path=None):
    """
    Create a second 3D figure showing win ratio of PPO over Optimal.

    For each (num_trucks, num_stops), win ratio is:
        count(seeds/scenarios where PPO metric > Optimal metric)
        / count(valid seeds/scenarios).

    If a `seed` column exists in both PPO and Optimal subsets, comparison is
    performed seed-wise. Otherwise each PPO seed/scenario is compared against
    the mean Optimal value for that (num_trucks, num_stops).
    """
    fig2 = plt.figure(figsize=(14, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.computed_zorder = False

    x_pos = np.arange(len(num_trucks))
    y_pos = np.arange(len(num_stops))
    x_grid, y_grid = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_data = x_grid.flatten()
    y_data = y_grid.flatten()
    z_data = []

    z_base = 0.0
    z_max = 100.0
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=z_base, vmax=z_max)

    for trucks in num_trucks:
        for stops in num_stops:
            ppo_mask = (df_ppo['num_trucks'] == trucks) & (df_ppo['num_stops'] == stops)
            optimal_mask = (df_optimal['num_trucks'] == trucks) & (df_optimal['num_stops'] == stops)

            ppo_subset = df_ppo[ppo_mask]
            optimal_subset = df_optimal[optimal_mask]

            if len(ppo_subset) == 0 or len(optimal_subset) == 0:
                z_data.append(np.nan)
                continue

            # Strict seed-wise comparison
            if ('seed' in ppo_subset.columns) and ('seed' in optimal_subset.columns):
                ppo_by_seed = (
                    ppo_subset[['seed', metric_col]]
                    .dropna(subset=[metric_col])
                    .groupby('seed', as_index=False)[metric_col]
                    .mean()
                )
                optimal_by_seed = (
                    optimal_subset[['seed', metric_col]]
                    .dropna(subset=[metric_col])
                    .groupby('seed', as_index=False)[metric_col]
                    .mean()
                    .rename(columns={metric_col: 'optimal_metric'})
                )

                merged = ppo_by_seed.merge(optimal_by_seed, on='seed', how='inner')
                if merged.empty:
                    z_data.append(np.nan)
                    continue

                wins = np.sum(merged[metric_col].to_numpy() > merged['optimal_metric'].to_numpy())
                valid = len(merged)
                win_ratio = (wins / valid) * 100.0
                z_data.append(win_ratio)
                continue

            # No seed info -> cannot compute strict per-seed win ratio
            z_data.append(np.nan)

    z_data = np.array(z_data, dtype=float)

    win_azim = 120
    draw_order = get_back_to_front_order(x_data, y_data, win_azim)
    zorder_base = 10
    for draw_rank, idx in enumerate(draw_order):
        x = x_data[idx]
        y = y_data[idx]
        z = z_data[idx]
        if not np.isfinite(z):
            continue

        color = cmap(norm(z))
        ax2.bar3d(
            x, y, z_base, 0.8, 0.8, z,
            color=color,
            alpha=1,
            edgecolor='black',
            linewidth=1,
            zsort='average',
            zorder=zorder_base + draw_rank
        )

    ax2.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Number of Stops', fontsize=12, fontweight='bold', labelpad=15)
    ax2.set_zlabel('Win Ratio', fontsize=12, fontweight='bold', labelpad=15)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(num_trucks)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(num_stops)

    ax2.set_xlim(-0.5, len(num_trucks))
    ax2.set_ylim(-1, len(num_stops))
    ax2.set_zlim(z_base, z_max)
    ax2.set_zticks(np.arange(0, 101, 10))

    fx = 18
    ax2.tick_params(axis='x', labelsize=fx)
    ax2.tick_params(axis='y', labelsize=fx)
    ax2.tick_params(axis='z', labelsize=fx)
    ax2.xaxis.label.set_size(fx)
    ax2.yaxis.label.set_size(fx)
    ax2.zaxis.label.set_size(fx)

    ax2.xaxis.pane.fill = True
    ax2.yaxis.pane.fill = True
    ax2.zaxis.pane.fill = True
    ax2.xaxis.pane.set_facecolor('white')
    ax2.yaxis.pane.set_facecolor('white')
    ax2.zaxis.pane.set_facecolor('white')
    ax2.xaxis.pane.set_alpha(1)
    ax2.yaxis.pane.set_alpha(1)
    ax2.zaxis.pane.set_alpha(1)

    ax2.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, pad=0.1, shrink=0.8)
    cbar_ticks = np.arange(0, 101, 10)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    cbar.ax.tick_params(labelsize=fx)
    cbar.set_label('Win Ratio GraphPPO vs Optimal (%)', fontsize=fx, fontweight='bold')

    ax2.view_init(elev=25, azim=win_azim)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Win-ratio plot saved to: {output_path}")

    plt.show()

    return fig2, ax2


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D bar plot visualization of grid evaluation results'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=DEFAULT_CSV,
        help=f'Path to CSV file (default: {DEFAULT_CSV})'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default=DEFAULT_METRIC,
        help=f'Metric column to visualize (default: {DEFAULT_METRIC})'
    )
    parser.add_argument(
        '--ppo-substring',
        type=str,
        default=DEFAULT_PPO_SUBSTRING,
        help=f'Substring to filter PPO policies (default: {DEFAULT_PPO_SUBSTRING})'
    )
    parser.add_argument(
        '--optimal-label',
        type=str,
        default=DEFAULT_OPTIMAL_LABEL,
        help=f'Label for optimal baseline (default: {DEFAULT_OPTIMAL_LABEL})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the figure (optional)'
    )
    
    args = parser.parse_args()
    
    # Verify CSV exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"✗ Error: CSV file not found at {csv_path}")
        return
    
    print(f"📊 Loading data from: {csv_path}")
    
    # Load and prepare data
    df_filtered, df_ppo, df_optimal, num_trucks, num_stops, metric_col = load_and_prepare_data(
        args.csv,
        args.metric,
        args.ppo_substring,
        args.optimal_label
    )
    
    print(f"✓ Loaded {len(df_filtered)} records")
    print(f"  - PPO records: {len(df_ppo)}")
    print(f"  - Optimal records: {len(df_optimal)}")
    print(f"  - Trucks: {num_trucks}")
    print(f"  - Stops: {num_stops}")
    print(f"  - Metric: {metric_col} (ratio: PPO / Optimal)")
    print(f"  - Colormap: viridis")
    
    # Create 3D plot
    output_file = args.output
    if output_file is None:
        output_file = 'results/visualization/grid_eval_3d_ratio_plot.png'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🎨 Creating 3D bar plot with PPO/Optimal reward ratio...")
    fig, ax = create_3d_bar_plot(
        df_ppo,
        df_optimal,
        num_trucks,
        num_stops,
        metric_col,
        args.optimal_label,
        output_path=output_file
    )

    win_ratio_output_file = f"{Path(output_file).with_suffix('')}_win_ratio.png"
    print("📈 Creating win-ratio figure (PPO > Optimal)...")
    fig2, ax2 = create_win_ratio_plot(
        df_ppo,
        df_optimal,
        num_trucks,
        num_stops,
        metric_col,
        output_path=win_ratio_output_file
    )
    
    print(f"✓ Visualization complete!")


if __name__ == '__main__':
    main()

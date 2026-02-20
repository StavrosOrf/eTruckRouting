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
DEFAULT_CSV = "results/grid_eval/grid_eval_20260219_131337/grid_evaluation_results.csv"
DEFAULT_METRIC = "mean_reward"
DEFAULT_PPO_SUBSTRING = "ppov_seq"
DEFAULT_OPTIMAL_LABEL = "Optimal (Simple)"


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
    
    # Filter for relevant policies
    ppo_mask = df['policy'].str.contains(ppo_substring, case=False, na=False)
    optimal_mask = df['policy'] == optimal_label
    df_filtered = df[ppo_mask | optimal_mask].copy()
    
    # Reset index
    df_filtered = df_filtered.reset_index(drop=True)
    
    # Get unique num_trucks and num_stops values
    num_trucks = sorted(df_filtered['num_trucks'].unique())
    num_stops = sorted(df_filtered['num_stops'].unique(), reverse=True)
    
    # Separate PPO and Optimal data
    df_ppo = df_filtered[ppo_mask].copy()
    df_optimal = df_filtered[optimal_mask].copy()
    
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
    
    # Create meshgrid for x (num_trucks) and y (num_stops)
    x_pos = np.arange(len(num_trucks))
    y_pos = np.arange(len(num_stops))
    x_grid, y_grid = np.meshgrid(x_pos, y_pos, indexing='ij')
    x_data = x_grid.flatten()
    y_data = y_grid.flatten()
    z_data = []

    # Bars are drawn from this baseline to avoid columns extending to z=0
    z_base = 0.99
    
    # Deviation column name (std of the metric)
    std_col = metric_col.replace('mean', 'std')
    
    # Get colormap
    cmap = cm.get_cmap('viridis')
    
    # Collect data for each combination
    all_z_values = []
    
    for i, trucks in enumerate(num_trucks):
        for j, stops in enumerate(num_stops):
            # Find matching rows
            ppo_mask = (df_ppo['num_trucks'] == trucks) & (df_ppo['num_stops'] == stops)
            optimal_mask = (df_optimal['num_trucks'] == trucks) & (df_optimal['num_stops'] == stops)
            
            ppo_subset = df_ppo[ppo_mask]
            optimal_subset = df_optimal[optimal_mask]
            
            if len(ppo_subset) > 0 and len(optimal_subset) > 0:
                # Get mean values
                ppo_reward = ppo_subset[metric_col].mean()
                optimal_reward = optimal_subset[metric_col].mean()
                
                # Calculate ratio (PPO / Optimal)
                if optimal_reward != 0:
                    ratio = ppo_reward / optimal_reward
                    # Error propagation: err(A/B) ≈ (A/B) * sqrt((errA/A)^2 + (errB/B)^2)
                    ppo_std = ppo_subset[std_col].mean()
                    optimal_std = optimal_subset[std_col].mean()
                    
                    ppo_rel_err = ppo_std / ppo_reward if ppo_reward != 0 else 0
                    optimal_rel_err = optimal_std / optimal_reward if optimal_reward != 0 else 0
                    ratio_err = ratio * np.sqrt(ppo_rel_err**2 + optimal_rel_err**2)
                else:
                    ratio = 0
                    ratio_err = 0
                
                z_data.append(ratio)
                all_z_values.append(ratio)
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
    
    # Plot bars with colors based on values
    for x, y, z in zip(x_data, y_data, z_data):
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
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
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
    
    # Add axis panes for better visibility
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.zaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_alpha(0.95)
    ax.yaxis.pane.set_alpha(0.95)
    ax.zaxis.pane.set_alpha(0.95)
    
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
    cbar.set_label('PPO / Optimal Ratio', fontsize=11, fontweight='bold')
    
    # Adjust viewing angle for better visualization (more elevated, rotated)
    ax.view_init(elev=20, azim=120)
    
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

            # Seed-wise/scenario-wise comparison when possible
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

            # Fallback: compare each PPO seed/scenario against mean Optimal
            optimal_value = optimal_subset[metric_col].mean()
            if not np.isfinite(optimal_value):
                z_data.append(np.nan)
                continue

            ppo_values = ppo_subset[metric_col].to_numpy(dtype=float)
            valid_ppo_values = ppo_values[np.isfinite(ppo_values)]
            if valid_ppo_values.size == 0:
                z_data.append(np.nan)
                continue

            wins = np.sum(valid_ppo_values > optimal_value)
            win_ratio = (wins / valid_ppo_values.size) * 100.0
            z_data.append(win_ratio)

    z_data = np.array(z_data, dtype=float)

    for x, y, z in zip(x_data, y_data, z_data):
        if not np.isfinite(z):
            continue

        color = cmap(norm(z))
        ax2.bar3d(
            x, y, z_base, 0.8, 0.8, z,
            color=color,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )

    ax2.set_xlabel('Number of Trucks', fontsize=12, fontweight='bold', labelpad=15)
    ax2.set_ylabel('Number of Stops', fontsize=12, fontweight='bold', labelpad=15)
    ax2.set_zlabel('Win Ratio vs Optimal (%)', fontsize=12, fontweight='bold', labelpad=15)

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(num_trucks)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(num_stops)

    ax2.set_xlim(-0.5, len(num_trucks))
    ax2.set_ylim(-1, len(num_stops))
    ax2.set_zlim(z_base, z_max)
    ax2.set_zticks(np.arange(0, 101, 10))

    ax2.xaxis.pane.fill = True
    ax2.yaxis.pane.fill = True
    ax2.zaxis.pane.fill = True
    ax2.xaxis.pane.set_facecolor('white')
    ax2.yaxis.pane.set_facecolor('white')
    ax2.zaxis.pane.set_facecolor('white')
    ax2.xaxis.pane.set_alpha(0.95)
    ax2.yaxis.pane.set_alpha(0.95)
    ax2.zaxis.pane.set_alpha(0.95)

    ax2.grid(True, alpha=0.3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, pad=0.1, shrink=0.8)
    cbar_ticks = np.arange(0, 101, 10)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    cbar.set_label('Win Ratio (%)', fontsize=11, fontweight='bold')

    ax2.view_init(elev=20, azim=120)

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
        output_file = 'grid_eval_3d_ratio_plot.png'
    
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

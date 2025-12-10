"""
Analyze delivery route generation to understand route characteristics.

This script generates multiple delivery routes and provides statistical analysis including:
- Distribution of distances between consecutive stops
- Number of deliveries per truck
- Total route distances
- Outlier detection for hop distances
- Feasibility statistics
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def calculate_hop_distances(env, truck):
    """Calculate energy distances (kWh) between consecutive stops in a truck's delivery sequence."""
    distances = []
    sequence = truck.delivery_sequence
    
    for i in range(len(sequence) - 1):
        from_node = int(sequence[i])
        to_node = int(sequence[i + 1])
        
        # Get energy requirement (kWh)
        energy = env.transport_graph.get_path_energy(from_node, to_node)
        distances.append(energy)
    
    return distances


def calculate_travel_times(env, truck):
    """Calculate travel times (hours) between consecutive stops in a truck's delivery sequence."""
    times = []
    sequence = truck.delivery_sequence
    
    for i in range(len(sequence) - 1):
        from_node = int(sequence[i])
        to_node = int(sequence[i + 1])
        
        # Get travel time in hours
        time_hours = env.transport_graph.get_time_distance(from_node, to_node)
        times.append(time_hours)
    
    return times


def check_feasibility(env, truck):
    """Check if a truck's route is feasible with its battery capacity."""
    sequence = truck.delivery_sequence
    battery_capacity = truck.battery_capacity
    charging_nodes = env.charging_nodes
    
    feasible_legs = 0
    total_legs = len(sequence) - 1
    
    for i in range(len(sequence) - 1):
        from_node = int(sequence[i])
        to_node = int(sequence[i + 1])
        
        # Check direct path
        direct_energy = env.transport_graph.get_path_energy(from_node, to_node)
        
        if direct_energy <= battery_capacity:
            feasible_legs += 1
            continue
        
        # Check via charging stations
        can_reach_via_charger = False
        for charger in charging_nodes:
            energy_to_charger = env.transport_graph.get_path_energy(from_node, int(charger))
            energy_from_charger = env.transport_graph.get_path_energy(int(charger), to_node)
            
            if (energy_to_charger <= battery_capacity and 
                energy_from_charger <= battery_capacity):
                can_reach_via_charger = True
                feasible_legs += 1
                break
    
    return feasible_legs, total_legs


def analyze_routes(config_file, num_samples=100, num_trucks=10, num_stops=5, seed=42):
    """Generate and analyze multiple delivery routes."""
    
    print(f"\n{'='*80}")
    print(f"Delivery Route Analysis")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  - Config file: {config_file}")
    print(f"  - Number of samples: {num_samples}")
    print(f"  - Trucks per sample: {num_trucks}")
    print(f"  - Stops per truck: {num_stops}")
    print(f"  - Random seed: {seed}\n")
    
    # Load config
    config = load_config(config_file)
    config['environment']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    
    # Storage for statistics
    all_hop_distances = []  # Energy distance in kWh
    all_hop_times = []
    all_total_distances = []
    all_total_times = []
    all_num_stops = []
    feasibility_stats = []
    start_nodes = []
    
    # Generate multiple environments to get different routes
    np.random.seed(seed)
    
    for i in tqdm(range(num_samples), desc="Generating routes"):
        env = EventDrivenTruckEnv(
            config=config,
            verbose=False,
            enable_plotting=False,
        )
        
        env.reset(seed=seed + i)
        
        # Analyze each truck in this environment
        for truck in env.trucks:
            # Hop distances (energy in kWh)
            hop_distances = calculate_hop_distances(env, truck)
            all_hop_distances.extend(hop_distances)
            
            # Travel times
            hop_times = calculate_travel_times(env, truck)
            all_hop_times.extend(hop_times)
            
            # Total distance and time
            total_distance = sum(hop_distances)
            total_time = sum(hop_times)
            all_total_distances.append(total_distance)
            all_total_times.append(total_time)
            
            # Number of stops
            all_num_stops.append(len(truck.delivery_sequence) - 1)
            
            # Feasibility
            feasible_legs, total_legs = check_feasibility(env, truck)
            feasibility_stats.append({
                'feasible_legs': feasible_legs,
                'total_legs': total_legs,
                'feasibility_rate': feasible_legs / total_legs if total_legs > 0 else 1.0
            })
            
            # Start node
            start_nodes.append(truck.delivery_sequence[0])
        
        env.close()
    
    # Convert to numpy arrays
    all_hop_distances = np.array(all_hop_distances)  # Energy in kWh
    all_hop_times = np.array(all_hop_times)
    all_total_distances = np.array(all_total_distances)
    all_total_times = np.array(all_total_times)
    all_num_stops = np.array(all_num_stops)
    
    # Calculate statistics
    print(f"\n{'='*80}")
    print(f"STATISTICS")
    print(f"{'='*80}\n")
    
    # Hop distance statistics (energy-based)
    print(f"Hop Energy Distances (energy between consecutive stops):")
    print(f"  - Total hops analyzed: {len(all_hop_distances)}")
    print(f"  - Mean: {np.mean(all_hop_distances):.2f} kWh")
    print(f"  - Median: {np.median(all_hop_distances):.2f} kWh")
    print(f"  - Std Dev: {np.std(all_hop_distances):.2f} kWh")
    print(f"  - Min: {np.min(all_hop_distances):.2f} kWh")
    print(f"  - Max: {np.max(all_hop_distances):.2f} kWh")
    print(f"  - 25th percentile: {np.percentile(all_hop_distances, 25):.2f} kWh")
    print(f"  - 75th percentile: {np.percentile(all_hop_distances, 75):.2f} kWh")
    print(f"  - 95th percentile: {np.percentile(all_hop_distances, 95):.2f} kWh")
    
    # Outlier detection using IQR method
    q1 = np.percentile(all_hop_distances, 25)
    q3 = np.percentile(all_hop_distances, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = all_hop_distances[(all_hop_distances < lower_bound) | (all_hop_distances > upper_bound)]
    print(f"  - Outliers (IQR method): {len(outliers)} ({100*len(outliers)/len(all_hop_distances):.1f}%)")
    if len(outliers) > 0:
        print(f"    - Outlier range: [{np.min(outliers):.2f}, {np.max(outliers):.2f}] kWh")
    
    # Energy statistics (same as hop distances since they measure energy)
    print(f"\nEnergy Requirements:")
    print(f"  - Battery capacity: {config['truck']['battery_capacity']:.2f} kWh")
    
    # Check if any hop exceeds battery capacity
    exceeds_capacity = np.sum(all_hop_distances > config['truck']['battery_capacity'])
    print(f"  - Hops exceeding battery capacity: {exceeds_capacity} ({100*exceeds_capacity/len(all_hop_distances):.1f}%)")
    
    # Travel time statistics
    print(f"\nHop Travel Times:")
    print(f"  - Mean: {np.mean(all_hop_times):.2f} hours")
    print(f"  - Median: {np.median(all_hop_times):.2f} hours")
    print(f"  - Std Dev: {np.std(all_hop_times):.2f} hours")
    print(f"  - Min: {np.min(all_hop_times):.2f} hours")
    print(f"  - Max: {np.max(all_hop_times):.2f} hours")
    print(f"  - 95th percentile: {np.percentile(all_hop_times, 95):.2f} hours")
    
    # Total distance statistics (energy-based)
    print(f"\nTotal Route Energy Distances:")
    print(f"  - Mean: {np.mean(all_total_distances):.2f} kWh")
    print(f"  - Median: {np.median(all_total_distances):.2f} kWh")
    print(f"  - Std Dev: {np.std(all_total_distances):.2f} kWh")
    print(f"  - Min: {np.min(all_total_distances):.2f} kWh")
    print(f"  - Max: {np.max(all_total_distances):.2f} kWh")
    
    # Total time statistics
    print(f"\nTotal Route Travel Times:")
    print(f"  - Mean: {np.mean(all_total_times):.2f} hours")
    print(f"  - Median: {np.median(all_total_times):.2f} hours")
    print(f"  - Std Dev: {np.std(all_total_times):.2f} hours")
    print(f"  - Min: {np.min(all_total_times):.2f} hours")
    print(f"  - Max: {np.max(all_total_times):.2f} hours")
    
    # Number of stops
    print(f"\nNumber of Stops per Route:")
    print(f"  - Mean: {np.mean(all_num_stops):.2f}")
    print(f"  - Unique values: {np.unique(all_num_stops)}")
    
    # Feasibility statistics
    feasibility_rates = [stat['feasibility_rate'] for stat in feasibility_stats]
    fully_feasible = sum(1 for rate in feasibility_rates if rate == 1.0)
    print(f"\nFeasibility (can complete with at most 1 charge per leg):")
    print(f"  - Fully feasible routes: {fully_feasible}/{len(feasibility_stats)} ({100*fully_feasible/len(feasibility_stats):.1f}%)")
    print(f"  - Mean feasibility rate: {np.mean(feasibility_rates)*100:.1f}%")
    
    if fully_feasible < len(feasibility_stats):
        infeasible_routes = [stat for stat in feasibility_stats if stat['feasibility_rate'] < 1.0]
        print(f"  - Routes with infeasible legs: {len(infeasible_routes)}")
    
    # Start node distribution
    unique_starts, start_counts = np.unique(start_nodes, return_counts=True)
    print(f"\nStart Node Distribution:")
    print(f"  - Unique start nodes: {len(unique_starts)}")
    print(f"  - Most common start nodes:")
    top_indices = np.argsort(start_counts)[-5:][::-1]
    for idx in top_indices:
        print(f"    - Node {unique_starts[idx]}: {start_counts[idx]} times ({100*start_counts[idx]/len(start_nodes):.1f}%)")
    
    # Create visualizations
    create_visualizations(
        all_hop_distances,
        all_hop_times,
        all_total_distances,
        all_total_times,
        all_num_stops,
        config['truck']['battery_capacity'],
        feasibility_rates
    )
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Plots saved to 'results/route_analysis/'")
    print(f"{'='*80}\n")


def create_visualizations(hop_distances, hop_times, total_distances, total_times, num_stops, battery_capacity, feasibility_rates):
    """Create visualization plots for route analysis."""
    
    # Create output directory
    output_dir = "results/route_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Hop distance and time distributions
    fig, axes = plt.subplots(4, 2, figsize=(15, 24))
    
    # Histogram
    ax = axes[0, 0]
    ax.hist(hop_distances, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(hop_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(hop_distances):.1f} kWh')
    ax.axvline(np.median(hop_distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(hop_distances):.1f} kWh')
    ax.set_xlabel('Energy Distance (kWh)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Hop Energy Distances', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot with outliers
    ax = axes[0, 1]
    bp = ax.boxplot(hop_distances, vert=True, patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('Energy Distance (kWh)', fontsize=12)
    ax.set_title('Hop Energy Distance Box Plot (with outliers)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Travel time distribution
    ax = axes[1, 0]
    ax.hist(hop_times, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(hop_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(hop_times):.2f} hours')
    ax.axvline(np.median(hop_times), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(hop_times):.2f} hours')
    ax.set_xlabel('Travel Time (hours)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Hop Travel Times', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot for travel times
    ax = axes[1, 1]
    bp = ax.boxplot(hop_times, vert=True, patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('mediumpurple')
    bp['boxes'][0].set_alpha(0.7)
    ax.set_ylabel('Travel Time (hours)', fontsize=12)
    ax.set_title('Hop Travel Time Box Plot (with outliers)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Total route travel times
    ax = axes[2, 0]
    ax.hist(total_times, bins=30, edgecolor='black', alpha=0.7, color='teal')
    ax.axvline(np.mean(total_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_times):.1f} hours')
    ax.axvline(np.median(total_times), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(total_times):.1f} hours')
    ax.set_xlabel('Total Travel Time (hours)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Total Route Travel Times', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total route energy distances
    ax = axes[2, 1]
    ax.hist(total_distances, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(total_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_distances):.1f} kWh')
    ax.set_xlabel('Total Energy Distance (kWh)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Total Route Energy Distances', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Number of stops distribution
    ax = axes[3, 0]
    unique_stops, stop_counts = np.unique(num_stops, return_counts=True)
    ax.bar(unique_stops, stop_counts, edgecolor='black', alpha=0.7, color='coral')
    ax.set_xlabel('Number of Stops', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Number of Stops per Route', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (stop, count) in enumerate(zip(unique_stops, stop_counts)):
        ax.text(stop, count, str(count), ha='center', va='bottom', fontsize=10)
    
    # Number of stops pie chart (if multiple values exist)
    ax = axes[3, 1]
    if len(unique_stops) > 1:
        ax.pie(stop_counts, labels=[f'{int(s)} stops' for s in unique_stops], autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3.colors)
        ax.set_title('Distribution of Number of Stops (Percentage)', fontsize=14, fontweight='bold')
    else:
        # If only one value, show a text summary instead
        ax.text(0.5, 0.5, f'All routes have\n{int(unique_stops[0])} stops\n({stop_counts[0]} routes)',
                ha='center', va='center', fontsize=16, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Number of Stops Summary', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/route_distributions.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/route_distributions.png")
    plt.close()
    
    # 2. Detailed outlier analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Violin plot
    ax = axes[0]
    parts = ax.violinplot([hop_distances], positions=[1], showmeans=True, showmedians=True, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    ax.set_ylabel('Energy Distance (kWh)', fontsize=12)
    ax.set_title('Hop Energy Distance Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['Hop Energy Distances'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Percentile analysis
    ax = axes[1]
    percentiles = np.arange(0, 101, 5)
    percentile_values = [np.percentile(hop_distances, p) for p in percentiles]
    ax.plot(percentiles, percentile_values, marker='o', linewidth=2, markersize=6)
    ax.axhline(np.median(hop_distances), color='red', linestyle='--', alpha=0.5, label='Median')
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Energy Distance (kWh)', fontsize=12)
    ax.set_title('Hop Energy Distance Percentile Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outlier_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/outlier_analysis.png")
    plt.close()
    
    # 3. Feasibility analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Feasibility rate histogram
    ax = axes[0]
    ax.hist(feasibility_rates, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Fully Feasible')
    ax.set_xlabel('Feasibility Rate', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Route Feasibility Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Energy vs Battery Capacity scatter
    ax = axes[1]
    sample_energies = hop_distances[:1000] if len(hop_distances) > 1000 else hop_distances
    colors = ['green' if e <= battery_capacity else 'red' for e in sample_energies]
    ax.scatter(range(len(sample_energies)), sample_energies, c=colors, alpha=0.5, s=20)
    ax.axhline(battery_capacity, color='red', linestyle='--', linewidth=2, label='Battery Capacity')
    ax.set_xlabel('Hop Index (sample)', fontsize=12)
    ax.set_ylabel('Energy Required (kWh)', fontsize=12)
    ax.set_title('Energy Requirements vs Battery Capacity', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feasibility_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/feasibility_analysis.png")
    plt.close()
    
    # 4. Time analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Hop time violin plot
    ax = axes[0, 0]
    parts = ax.violinplot([hop_times], positions=[1], showmeans=True, showmedians=True, widths=0.7)
    for pc in parts['bodies']:
        pc.set_facecolor('mediumpurple')
        pc.set_alpha(0.7)
    ax.set_ylabel('Travel Time (hours)', fontsize=12)
    ax.set_title('Hop Travel Time Distribution (Violin Plot)', fontsize=14, fontweight='bold')
    ax.set_xticks([1])
    ax.set_xticklabels(['Hop Travel Times'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Hop time percentile analysis
    ax = axes[0, 1]
    percentiles = np.arange(0, 101, 5)
    time_percentile_values = [np.percentile(hop_times, p) for p in percentiles]
    ax.plot(percentiles, time_percentile_values, marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(np.median(hop_times), color='red', linestyle='--', alpha=0.5, label='Median')
    ax.set_xlabel('Percentile', fontsize=12)
    ax.set_ylabel('Travel Time (hours)', fontsize=12)
    ax.set_title('Hop Travel Time Percentile Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Total route time distribution
    ax = axes[1, 0]
    ax.hist(total_times, bins=30, edgecolor='black', alpha=0.7, color='teal')
    ax.axvline(np.mean(total_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(total_times):.1f} hours')
    ax.axvline(np.median(total_times), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(total_times):.1f} hours')
    ax.set_xlabel('Total Travel Time (hours)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Total Route Travel Times', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot: Energy vs Time for hops
    ax = axes[1, 1]
    sample_size = min(1000, len(hop_times))
    sample_indices = np.random.choice(len(hop_times), sample_size, replace=False)
    ax.scatter(hop_times[sample_indices], hop_distances[sample_indices], alpha=0.5, s=20, c='darkviolet')
    ax.set_xlabel('Travel Time (hours)', fontsize=12)
    ax.set_ylabel('Energy Distance (kWh)', fontsize=12)
    ax.set_title('Hop Energy vs Travel Time (sample)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/time_analysis.png")
    plt.close()
    
    # 5. Summary statistics plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['', ''],
        ['Hop Energy Distances', ''],
        ['  Mean', f'{np.mean(hop_distances):.2f} kWh'],
        ['  Median', f'{np.median(hop_distances):.2f} kWh'],
        ['  Std Dev', f'{np.std(hop_distances):.2f} kWh'],
        ['  Min', f'{np.min(hop_distances):.2f} kWh'],
        ['  Max', f'{np.max(hop_distances):.2f} kWh'],
        ['  Battery Capacity', f'{battery_capacity:.2f} kWh'],
        ['  Exceeds Capacity', f'{100*np.sum(hop_distances > battery_capacity)/len(hop_distances):.1f}%'],
        ['', ''],
        ['Hop Travel Times', ''],
        ['  Mean', f'{np.mean(hop_times):.2f} hours'],
        ['  Median', f'{np.median(hop_times):.2f} hours'],
        ['  Max', f'{np.max(hop_times):.2f} hours'],
        ['', ''],
        ['Feasibility', ''],
        ['  Fully Feasible', f'{100*np.sum(np.array(feasibility_rates) == 1.0)/len(feasibility_rates):.1f}%'],
        ['  Mean Feasibility', f'{100*np.mean(feasibility_rates):.1f}%'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style section headers
    for row in [2, 8, 13]:
        table[(row, 0)].set_facecolor('#E0E0E0')
        table[(row, 0)].set_text_props(weight='bold')
        table[(row, 1)].set_facecolor('#E0E0E0')
    
    ax.set_title('Delivery Route Analysis - Summary Statistics', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(f"{output_dir}/summary_statistics.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/summary_statistics.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze delivery route generation')
    parser.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
                       help='Path to config file')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of environment samples to generate')
    parser.add_argument('--num-trucks', type=int, default=100,
                       help='Number of trucks per sample')
    parser.add_argument('--num-stops', type=int, default=5,
                       help='Number of stops per truck')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    analyze_routes(
        config_file=args.config,
        num_samples=args.num_samples,
        num_trucks=args.num_trucks,
        num_stops=args.num_stops,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

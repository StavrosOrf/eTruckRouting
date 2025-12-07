#!/usr/bin/env python3
"""
Visualize the transportation network graph with charging stations.
Shows node types, charger capacities, and network connectivity.
"""
import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def create_network_visualization(env, output_path=None, show_plot=True):
    """Create a comprehensive visualization of the transportation network."""
    
    G = env.transport_graph.graph
    charging_nodes = set(env.charging_nodes)
    
    # Get charger details
    charger_details = env.transport_graph.get_charger_details()
    
    # Gather statistics
    total_nodes = G.number_of_nodes()
    total_edges = G.number_of_edges()
    num_chargers = len(charging_nodes)
    
    # Count chargers by type
    charger_counts = defaultdict(int)
    total_capacity_by_type = defaultdict(int)
    for node_id, info in charger_details.items():
        types = info.get('types', {})
        for charger_type, capacity in types.items():
            charger_counts[charger_type] += 1
            total_capacity_by_type[charger_type] += capacity
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Main network visualization
    ax_main = fig.add_subplot(gs[:2, :2])
    
    # Calculate node positions using spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Separate nodes by type
    charger_nodes = list(charging_nodes)
    regular_nodes = [n for n in G.nodes() if n not in charging_nodes]
    
    # Draw regular nodes (smaller, light gray)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=regular_nodes,
        node_color='lightgray',
        node_size=30,
        alpha=0.6,
        ax=ax_main
    )
    
    # Draw charger nodes by type
    level2_nodes = []
    dcfast_nodes = []
    mixed_nodes = []
    
    for node_id in charger_nodes:
        types = charger_details[node_id].get('types', {})
        has_level2 = 'Level2' in types
        has_dcfast = 'DCFast' in types
        
        if has_level2 and has_dcfast:
            mixed_nodes.append(node_id)
        elif has_level2:
            level2_nodes.append(node_id)
        elif has_dcfast:
            dcfast_nodes.append(node_id)
    
    # Draw Level2 chargers (green)
    if level2_nodes:
        sizes = [charger_details[n]['types'].get('Level2', 0) * 50 for n in level2_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=level2_nodes,
            node_color='green',
            node_size=sizes,
            alpha=0.7,
            ax=ax_main,
            label='Level2'
        )
    
    # Draw DCFast chargers (red)
    if dcfast_nodes:
        sizes = [charger_details[n]['types'].get('DCFast', 0) * 50 for n in dcfast_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=dcfast_nodes,
            node_color='red',
            node_size=sizes,
            alpha=0.7,
            ax=ax_main,
            label='DCFast'
        )
    
    # Draw mixed chargers (purple)
    if mixed_nodes:
        sizes = [sum(charger_details[n]['types'].values()) * 50 for n in mixed_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=mixed_nodes,
            node_color='purple',
            node_size=sizes,
            alpha=0.7,
            ax=ax_main,
            label='Mixed'
        )
    
    # Draw edges (light, thin)
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.2,
        width=0.5,
        arrows=False,
        ax=ax_main
    )
    
    ax_main.set_title(
        f'Transportation Network\n{total_nodes} nodes, {total_edges} edges, {num_chargers} charging stations',
        fontsize=14,
        fontweight='bold'
    )
    ax_main.legend(loc='upper left', fontsize=10)
    ax_main.axis('off')
    
    # Statistics panel - Top right
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis('off')
    
    stats_text = "Network Statistics\n" + "="*30 + "\n\n"
    stats_text += f"Total Nodes: {total_nodes}\n"
    stats_text += f"Total Edges: {total_edges}\n"
    stats_text += f"Charging Stations: {num_chargers}\n\n"
    stats_text += "Chargers by Type:\n"
    for ctype in sorted(charger_counts.keys()):
        stats_text += f"  • {ctype}: {charger_counts[ctype]} nodes\n"
        stats_text += f"    Total capacity: {total_capacity_by_type[ctype]}\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Degree distribution - Middle right
    ax_degree = fig.add_subplot(gs[1, 2])
    degrees = [d for n, d in G.degree()]
    ax_degree.hist(degrees, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax_degree.set_xlabel('Node Degree')
    ax_degree.set_ylabel('Count')
    ax_degree.set_title('Degree Distribution')
    ax_degree.grid(True, alpha=0.3)
    
    # Charger capacity distribution - Bottom left
    ax_capacity = fig.add_subplot(gs[2, 0])
    
    level2_capacities = []
    dcfast_capacities = []
    
    for node_id, info in charger_details.items():
        types = info.get('types', {})
        if 'Level2' in types:
            level2_capacities.append(types['Level2'])
        if 'DCFast' in types:
            dcfast_capacities.append(types['DCFast'])
    
    x = np.arange(len(charger_counts))
    width = 0.35
    
    if level2_capacities:
        ax_capacity.hist(level2_capacities, bins=20, alpha=0.6, label='Level2', color='green')
    if dcfast_capacities:
        ax_capacity.hist(dcfast_capacities, bins=20, alpha=0.6, label='DCFast', color='red')
    
    ax_capacity.set_xlabel('Capacity (# of ports)')
    ax_capacity.set_ylabel('Count')
    ax_capacity.set_title('Charger Capacity Distribution')
    ax_capacity.legend()
    ax_capacity.grid(True, alpha=0.3)
    
    # Top chargers table - Bottom middle
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis('off')
    
    # Get top 10 chargers by capacity
    charger_list = []
    for node_id, info in charger_details.items():
        types = info.get('types', {})
        total_cap = sum(types.values())
        orig_id = info.get('original_id', node_id)
        type_str = ', '.join([f"{k}:{v}" for k, v in sorted(types.items())])
        charger_list.append((node_id, orig_id, total_cap, type_str))
    
    charger_list.sort(key=lambda x: x[2], reverse=True)
    top_chargers = charger_list[:10]
    
    table_text = "Top 10 Charging Stations\n" + "="*40 + "\n\n"
    table_text += f"{'Node':<6} {'Orig ID':<12} {'Cap':<5} {'Type:Count'}\n"
    table_text += "-"*50 + "\n"
    for node_id, orig_id, cap, type_str in top_chargers:
        table_text += f"{node_id:<6} {orig_id:<12} {cap:<5} {type_str}\n"
    
    ax_table.text(0.05, 0.95, table_text, transform=ax_table.transAxes,
                  fontsize=8, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Edge statistics - Bottom right
    ax_edge_stats = fig.add_subplot(gs[2, 2])
    
    # Sample edge distances
    edge_distances = []
    edge_times = []
    for u, v, data in G.edges(data=True):
        dist = data.get('distance', 0)
        time = data.get('time', 0)
        if dist > 0 and dist < 1000:  # Filter outliers
            edge_distances.append(dist)
        if time > 0 and time < 100:
            edge_times.append(time)
    
    if edge_distances:
        ax_edge_stats.hist(edge_distances[:1000], bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax_edge_stats.set_xlabel('Edge Distance (kWh)')
        ax_edge_stats.set_ylabel('Count')
        ax_edge_stats.set_title('Edge Distance Distribution (sample)')
        ax_edge_stats.grid(True, alpha=0.3)
    
    plt.suptitle('Transportation Network Visualization', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def print_network_summary(env):
    """Print detailed text summary of the network."""
    
    G = env.transport_graph.graph
    charging_nodes = set(env.charging_nodes)
    charger_details = env.transport_graph.get_charger_details()
    
    print("\n" + "="*80)
    print("TRANSPORTATION NETWORK SUMMARY")
    print("="*80)
    
    print(f"\nNetwork Structure:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Charging stations: {len(charging_nodes)}")
    print(f"  Regular nodes: {G.number_of_nodes() - len(charging_nodes)}")
    
    # Connectivity stats
    if nx.is_strongly_connected(G):
        print(f"  Graph connectivity: Strongly connected")
    elif nx.is_weakly_connected(G):
        print(f"  Graph connectivity: Weakly connected")
    else:
        num_components = nx.number_weakly_connected_components(G)
        print(f"  Graph connectivity: {num_components} weakly connected components")
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    print(f"\n  Average degree: {np.mean(degrees):.2f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Min degree: {min(degrees)}")
    
    # Charger statistics
    print(f"\nCharging Stations:")
    charger_counts = defaultdict(int)
    total_capacity_by_type = defaultdict(int)
    
    for node_id, info in charger_details.items():
        types = info.get('types', {})
        for charger_type, capacity in types.items():
            charger_counts[charger_type] += 1
            total_capacity_by_type[charger_type] += capacity
    
    for ctype in sorted(charger_counts.keys()):
        print(f"  {ctype}:")
        print(f"    Nodes: {charger_counts[ctype]}")
        print(f"    Total capacity: {total_capacity_by_type[ctype]}")
        print(f"    Avg capacity per node: {total_capacity_by_type[ctype]/charger_counts[ctype]:.1f}")
    
    # Sample chargers
    print(f"\nSample Charging Stations (first 5):")
    for i, (node_id, info) in enumerate(sorted(charger_details.items())[:5]):
        types = info.get('types', {})
        orig_id = info.get('original_id', node_id)
        type_str = ', '.join([f"{k}:{v}" for k, v in sorted(types.items())])
        degree = G.degree(node_id)
        print(f"  Node {node_id} (orig {orig_id}): {type_str}, degree={degree}")
    
    # Edge statistics
    edge_distances = []
    edge_times = []
    for u, v, data in G.edges(data=True):
        dist = data.get('distance', 0)
        time = data.get('time', 0)
        if dist > 0:
            edge_distances.append(dist)
        if time > 0:
            edge_times.append(time)
    
    if edge_distances:
        print(f"\nEdge Statistics:")
        print(f"  Distance (kWh):")
        print(f"    Mean: {np.mean(edge_distances):.2f}")
        print(f"    Median: {np.median(edge_distances):.2f}")
        print(f"    Min: {min(edge_distances):.2f}")
        print(f"    Max: {max(edge_distances):.2f}")
    
    if edge_times:
        print(f"  Travel Time (hours):")
        print(f"    Mean: {np.mean(edge_times):.2f}")
        print(f"    Median: {np.median(edge_times):.2f}")
        print(f"    Min: {min(edge_times):.2f}")
        print(f"    Max: {max(edge_times):.2f}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Visualize transportation network')
    parser.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='results/network_visualization.png',
                        help='Output path for visualization')
    parser.add_argument('--no-show', action='store_true',
                        help='Don\'t display the plot (only save)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed network information')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Create environment
    print("Creating environment...")
    env = EventDrivenTruckEnv(
        config=config,
        verbose=args.verbose,
        enable_plotting=False
    )
    
    # Print text summary
    print_network_summary(env)
    
    # Create visualization
    print("\nGenerating visualization...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_network_visualization(
        env,
        output_path=args.output,
        show_plot=not args.no_show
    )
    
    print(f"\n✓ Complete! Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()

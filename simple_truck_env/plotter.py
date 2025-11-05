"""
Visualization and plotting utilities for the event-driven truck environment.
"""
import os
from typing import Dict, List, Any
import numpy as np


class EnvironmentPlotter:
    """
    Handles all plotting and visualization for the truck routing environment.
    """
    
    def __init__(self, output_dir: str, verbose: bool = False):
        """
        Initialize the plotter.
        
        Args:
            output_dir: Directory to save plots
            verbose: Print verbose messages
        """
        self.output_dir = output_dir
        self.verbose = verbose
        
    def plot_initial_routes(
        self,
        transport_graph: Any,
        truck_initial_plans: Dict[int, Dict],
        charging_nodes: List[int],
        num_trucks: int,
        num_stops: int
    ):
        """
        Plot initial truck starting points and planned delivery routes.
        
        Args:
            transport_graph: TransportationGraph instance
            truck_initial_plans: Dictionary mapping truck_id to initial plan
            charging_nodes: List of charging station node IDs
            num_trucks: Number of trucks
            num_stops: Number of stops per truck
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import networkx as nx
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            print("Warning: matplotlib not available. Skipping plots.")
            return
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Use spring layout for better visualization
        all_nodes = transport_graph.get_all_nodes()
        print(f"Computing spring layout for {len(all_nodes)} nodes...")
        node_positions = nx.spring_layout(transport_graph.graph, k=1.5, iterations=50, seed=42)
        print("Spring layout computed.")
        
        # Plot all edges (road network) in light gray
        for edge in transport_graph.graph.edges():
            x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
            y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
            ax.plot(x_coords, y_coords, 'lightgray', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Collect all delivery/destination nodes across all trucks
        all_delivery_nodes = set()
        for plan in truck_initial_plans.values():
            all_delivery_nodes.update(plan['deliveries'])
        
        # Plot all non-charging, non-delivery nodes as tiny dots
        regular_nodes = [n for n in all_nodes if n not in charging_nodes and n not in all_delivery_nodes]
        if regular_nodes:
            regular_x = [node_positions[n][0] for n in regular_nodes]
            regular_y = [node_positions[n][1] for n in regular_nodes]
            ax.scatter(regular_x, regular_y, c='lightgray', s=5, marker='o',
                      label=f'Network Nodes ({len(regular_nodes)})', alpha=0.3, zorder=2)
        
        # Plot charging stations with labels
        if charging_nodes:
            charger_x = [node_positions[n][0] for n in charging_nodes]
            charger_y = [node_positions[n][1] for n in charging_nodes]
            ax.scatter(charger_x, charger_y, c='orange', s=250, marker='s', 
                      label=f'Charging Stations ({len(charging_nodes)})', alpha=0.8, 
                      edgecolors='darkorange', linewidths=2, zorder=4)
            
            # Annotate charging station IDs
            for node in charging_nodes:
                pos = node_positions[node]
                ax.annotate(f'C{node}', xy=pos, xytext=(0, 0), textcoords='offset points',
                           fontsize=9, color='white', weight='bold', ha='center', va='center', zorder=5)
        
        # Plot each truck's planned route
        colors = plt.cm.tab10(np.linspace(0, 1, num_trucks))
        
        for truck_id, plan in truck_initial_plans.items():
            color = colors[truck_id]
            
            # Plot start point
            start_pos = node_positions[plan['start']]
            ax.scatter(*start_pos, c=[color], s=300, marker='*', 
                      label=f'Truck {truck_id} Start', edgecolors='black', linewidths=2, zorder=6)
            
            # Plot delivery/destination points
            delivery_x = [node_positions[n][0] for n in plan['deliveries']]
            delivery_y = [node_positions[n][1] for n in plan['deliveries']]
            ax.scatter(delivery_x, delivery_y, c=[color]*len(plan['deliveries']), 
                      s=150, marker='o', alpha=0.8, edgecolors='black', linewidths=1.5, 
                      label=f'Truck {truck_id} Destinations', zorder=5)
            
            # Draw planned route
            route_nodes = [plan['start']] + plan['deliveries']
            for i in range(len(route_nodes) - 1):
                x_coords = [node_positions[route_nodes[i]][0], node_positions[route_nodes[i+1]][0]]
                y_coords = [node_positions[route_nodes[i]][1], node_positions[route_nodes[i+1]][1]]
                ax.plot(x_coords, y_coords, color=color, alpha=0.5, linewidth=2.5, 
                       linestyle='--', zorder=3)
            
            # Annotate delivery sequence numbers
            for idx, node in enumerate(plan['deliveries'], 1):
                pos = node_positions[node]
                ax.annotate(f'{idx}', xy=pos, fontsize=9, ha='center', va='center', 
                           color='white', weight='bold', zorder=7)
        
        ax.set_xlabel('Spring Layout X', fontsize=12)
        ax.set_ylabel('Spring Layout Y', fontsize=12)
        ax.set_title(f'Initial Planned Routes - Transportation Network (Spring Layout)\n{num_trucks} Trucks, {num_stops} Stops Each, {len(charging_nodes)} Charging Stations', 
                    fontsize=14, weight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics box
        stats_text = f'Graph Stats:\nNodes: {len(all_nodes)}\nEdges: {transport_graph.graph.number_of_edges()}\nChargers: {len(charging_nodes)}\nDestinations: {len(all_delivery_nodes)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        
        filepath = os.path.join(self.output_dir, 'initial_routes.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Initial routes plot saved to: {filepath}")
    
    def plot_actual_routes(
        self,
        transport_graph: Any,
        truck_routes: Dict[int, List],
        charging_nodes: List[int],
        num_trucks: int,
        global_clock: float
    ):
        """
        Plot the actual routes taken by trucks during simulation.
        
        Args:
            transport_graph: TransportationGraph instance
            truck_routes: Dictionary mapping truck_id to list of (node, time, event_type) tuples
            charging_nodes: List of charging station node IDs
            num_trucks: Number of trucks
            global_clock: Current simulation time
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            import networkx as nx
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            print("Warning: matplotlib not available. Skipping plots.")
            return
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Use spring layout for better visualization
        all_nodes = transport_graph.get_all_nodes()
        print(f"Computing spring layout for {len(all_nodes)} nodes...")
        node_positions = nx.spring_layout(transport_graph.graph, k=1.5, iterations=50, seed=42)
        print("Spring layout computed.")
        
        # Plot all edges (road network) in light gray
        for edge in transport_graph.graph.edges():
            x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
            y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
            ax.plot(x_coords, y_coords, 'lightgray', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Collect all delivery/destination nodes across all trucks
        all_delivery_nodes = set()
        for route in truck_routes.values():
            all_delivery_nodes.update([r[0] for r in route if r[2] == 'delivery'])
        
        # Plot all non-charging, non-delivery nodes as small dots
        regular_nodes = [n for n in all_nodes if n not in charging_nodes and n not in all_delivery_nodes]
        if regular_nodes:
            non_charge_x = [node_positions[n][0] for n in regular_nodes]
            non_charge_y = [node_positions[n][1] for n in regular_nodes]
            ax.scatter(non_charge_x, non_charge_y, c='lightgray', s=5, marker='o',
                      label=f'Network Nodes ({len(regular_nodes)})', alpha=0.3, zorder=2)
        
        # Plot charging stations with labels
        if charging_nodes:
            charger_x = [node_positions[n][0] for n in charging_nodes]
            charger_y = [node_positions[n][1] for n in charging_nodes]
            ax.scatter(charger_x, charger_y, c='orange', s=250, marker='s', 
                      label=f'Charging Stations ({len(charging_nodes)})', alpha=0.8, 
                      edgecolors='darkorange', linewidths=2, zorder=4)
            
            # Annotate charging station IDs
            for node in charging_nodes:
                pos = node_positions[node]
                ax.annotate(f'C{node}', xy=pos, xytext=(0, 0), textcoords='offset points',
                           fontsize=9, color='white', weight='bold', ha='center', va='center', zorder=5)
        
        # Plot each truck's actual route
        colors = plt.cm.tab10(np.linspace(0, 1, num_trucks))
        
        for truck_id, route in truck_routes.items():
            if not route:
                continue
            
            color = colors[truck_id]
            
            # Separate nodes by type
            start_nodes = [r[0] for r in route if r[2] == 'start']
            delivery_nodes = [r[0] for r in route if r[2] == 'delivery']
            charger_visits = [r[0] for r in route if r[2] == 'charger']
            
            # Plot start
            if start_nodes:
                start_pos = node_positions[start_nodes[0]]
                ax.scatter(*start_pos, c=[color], s=300, marker='*', 
                          label=f'Truck {truck_id} Start', edgecolors='black', linewidths=2, zorder=6)
            
            # Plot deliveries/destinations
            if delivery_nodes:
                delivery_x = [node_positions[n][0] for n in delivery_nodes]
                delivery_y = [node_positions[n][1] for n in delivery_nodes]
                ax.scatter(delivery_x, delivery_y, c=[color]*len(delivery_nodes), 
                          s=150, marker='o', alpha=0.8, edgecolors='black', linewidths=2, 
                          label=f'Truck {truck_id} Deliveries', zorder=5)
            
            # Plot charger visits
            if charger_visits:
                charger_visit_x = [node_positions[n][0] for n in charger_visits]
                charger_visit_y = [node_positions[n][1] for n in charger_visits]
                ax.scatter(charger_visit_x, charger_visit_y, c=[color]*len(charger_visits), 
                          s=120, marker='D', alpha=0.6, edgecolors='black', linewidths=1.5, zorder=4)
            
            # Draw actual route path
            route_nodes = [r[0] for r in route]
            for i in range(len(route_nodes) - 1):
                x_coords = [node_positions[route_nodes[i]][0], node_positions[route_nodes[i+1]][0]]
                y_coords = [node_positions[route_nodes[i]][1], node_positions[route_nodes[i+1]][1]]
                ax.plot(x_coords, y_coords, color=color, alpha=0.7, linewidth=2.5, zorder=3)
            
            # Add arrows to show direction
            for i in range(0, len(route_nodes) - 1, max(1, len(route_nodes) // 5)):  # Every few segments
                x1, y1 = node_positions[route_nodes[i]]
                x2, y2 = node_positions[route_nodes[i+1]]
                dx, dy = x2 - x1, y2 - y1
                ax.annotate('', xy=(x1 + 0.6*dx, y1 + 0.6*dy), xytext=(x1 + 0.4*dx, y1 + 0.4*dy),
                           arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.7), zorder=3)
        
        ax.set_xlabel('Spring Layout X', fontsize=12)
        ax.set_ylabel('Spring Layout Y', fontsize=12)
        ax.set_title(f'Actual Truck Routes - Transportation Network (Spring Layout)\nSimulation Time: {global_clock:.1f}h', 
                    fontsize=14, weight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics box
        total_visits = sum(len(route) for route in truck_routes.values())
        charger_visits = sum(len([r for r in route if r[2] == 'charger']) for route in truck_routes.values())
        stats_text = f'Route Stats:\nTotal Visits: {total_visits}\nCharger Visits: {charger_visits}\nDeliveries: {len(all_delivery_nodes)}\nTime: {global_clock:.1f}h'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        filepath = os.path.join(self.output_dir, 'actual_routes.png')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Actual routes plot saved to: {filepath}")

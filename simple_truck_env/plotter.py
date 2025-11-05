"""
Visualization and plotting utilities for the event-driven truck environment.
"""
import os
from typing import Dict, List, Any, Tuple
import numpy as np


def compute_node_coordinates_from_distances(graph, verbose: bool = False) -> Dict[int, Tuple[float, float]]:
    """
    Compute 2D coordinates for graph nodes such that Euclidean distances 
    match edge weights as closely as possible using Multidimensional Scaling (MDS).
    
    Args:
        graph: NetworkX graph with 'distance' weights on edges
        verbose: Print progress information
        
    Returns:
        Dictionary mapping node_id to (x, y) coordinates
    """
    import networkx as nx
    from sklearn.manifold import MDS
    
    nodes = list(graph.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    if verbose:
        print(f"Computing coordinates for {n} nodes using edge distances...")
    
    # Compute shortest path distances between all pairs of nodes
    # This creates a distance matrix where entry (i,j) is the shortest path distance
    distance_matrix = np.zeros((n, n))
    
    # Use NetworkX to compute all pairs shortest path lengths
    # This handles disconnected graphs gracefully
    try:
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='distance'))
        
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    distance_matrix[i, j] = 0
                elif node_j in shortest_paths.get(node_i, {}):
                    distance_matrix[i, j] = shortest_paths[node_i][node_j]
                else:
                    # If no path exists, use a large distance
                    distance_matrix[i, j] = 1000.0
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute shortest paths, using direct edges only: {e}")
        # Fallback: use only direct edge distances
        distance_matrix.fill(1000.0)
        np.fill_diagonal(distance_matrix, 0)
        for u, v, data in graph.edges(data=True):
            i, j = node_to_idx[u], node_to_idx[v]
            dist = data.get('distance', 1.0)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    if verbose:
        print("Distance matrix computed. Running MDS...")
    
    # Apply MDS to embed nodes in 2D space
    # dissimilarity='precomputed' means we're providing distances, not features
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, 
              max_iter=500, n_init=4, normalized_stress='auto')
    
    try:
        coords_2d = mds.fit_transform(distance_matrix)
    except Exception as e:
        if verbose:
            print(f"MDS failed: {e}. Using spring layout as fallback.")
        # Fallback to spring layout
        pos = nx.spring_layout(graph, weight='distance', k=2, iterations=100, seed=42)
        return {node: (pos[node][0] * 100, pos[node][1] * 100) for node in nodes}
    
    # Create the node_positions dictionary
    node_positions = {}
    for i, node in enumerate(nodes):
        node_positions[node] = (coords_2d[i, 0], coords_2d[i, 1])
    
    if verbose:
        print(f"MDS completed. Stress: {mds.stress_:.2f}")
        print("Coordinates computed successfully.")
    
    return node_positions


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
        
        # Compute node positions from graph edge distances
        all_nodes = transport_graph.get_all_nodes()
        
        # Check if coordinates are already computed
        has_coords = all(transport_graph.graph.nodes[node].get('x') is not None and 
                        transport_graph.graph.nodes[node].get('y') is not None 
                        for node in list(all_nodes)[:10])
        
        if has_coords:
            # Use existing coordinates
            node_positions = {}
            for node in all_nodes:
                data = transport_graph.graph.nodes[node]
                node_positions[node] = (data.get('x', 0), data.get('y', 0))
            print(f"Using pre-computed coordinates for {len(all_nodes)} nodes.")
        else:
            # Compute coordinates from edge distances
            node_positions = compute_node_coordinates_from_distances(
                transport_graph.graph, verbose=self.verbose
            )
            # Store computed coordinates back in the graph
            for node, (x, y) in node_positions.items():
                transport_graph.graph.nodes[node]['x'] = x
                transport_graph.graph.nodes[node]['y'] = y
        
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
        
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title(f'Initial Planned Routes - Transportation Network\n{num_trucks} Trucks, {num_stops} Stops Each, {len(charging_nodes)} Charging Stations', 
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
        
        # Use the same coordinates as computed in plot_initial_routes
        all_nodes = transport_graph.get_all_nodes()
        
        # Check if coordinates are already computed
        has_coords = all(transport_graph.graph.nodes[node].get('x') is not None and 
                        transport_graph.graph.nodes[node].get('y') is not None 
                        for node in list(all_nodes)[:10])
        
        if has_coords:
            # Use existing coordinates
            node_positions = {}
            for node in all_nodes:
                data = transport_graph.graph.nodes[node]
                node_positions[node] = (data.get('x', 0), data.get('y', 0))
            print(f"Using pre-computed coordinates for {len(all_nodes)} nodes.")
        else:
            # Compute coordinates from edge distances
            node_positions = compute_node_coordinates_from_distances(
                transport_graph.graph, verbose=self.verbose
            )
            # Store computed coordinates back in the graph
            for node, (x, y) in node_positions.items():
                transport_graph.graph.nodes[node]['x'] = x
                transport_graph.graph.nodes[node]['y'] = y
        
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
        
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title(f'Actual Truck Routes - Transportation Network\nSimulation Time: {global_clock:.1f}h', 
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

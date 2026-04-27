"""
Visualization script for GNN state representation.

Creates visual plots of the GNN graph structure, showing:
- Node types (Trucks, Deliveries, Chargers) and their features
- Edge connections representing transportation network
- Graph statistics and topology evolution
- Feature distributions and correlations

Simplified GNN Design:
- Nodes: Active trucks + undelivered deliveries + all chargers (NO depot nodes)
- Edges: State-based truck connections + location-to-location connections
  * READY trucks: connect to next delivery + feasible chargers
  * ROUTING trucks: no edges (in transit)
  * WAITING/CHARGING trucks: only to current charger
- Edge Features: [energy_distance (kWh), time_to_traverse (hours)]
- Dynamic: Graph shrinks as deliveries complete and trucks finish
"""

import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


class GNNVisualizer:
    """Visualizes GNN state representations."""

    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """Initialize visualizer with figure size."""
        self.figsize = figsize
        # Updated for simplified GNN: Truck=0, Delivery=1, Charger=2 (no depot)
        self.node_type_names = {0: "Truck", 1: "Delivery", 2: "Charger"}
        self.node_colors = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#45B7D1"}
        self.node_sizes = {0: 800, 1: 600, 2: 600}

    def plot_graph_structure(self, data, env, title: str = "GNN Graph Structure"):
        """
        Plot the graph structure with nodes and edges.

        Args:
            data: PyTorch Geometric Data object
            env: EventDrivenTruckEnv instance
            title: Title for the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Subplot 1: Graph layout with networkx
        ax1 = axes[0, 0]
        self._plot_networkx_layout(data, env, ax1)

        # Subplot 2: Node type distribution
        ax2 = axes[0, 1]
        self._plot_node_distribution(data, ax2)

        # Subplot 3: Node features heatmap
        ax3 = axes[1, 0]
        self._plot_node_features_heatmap(data, ax3)

        # Subplot 4: Edge statistics
        ax4 = axes[1, 1]
        self._plot_edge_statistics(data, env, ax4)

        plt.tight_layout()
        return fig

    def verify_charger_capacity_features(self, data, env, tol: float = 1e-3):
        """Verify charger occupancy_rate feature matches env occupancy/capacity.

        Logs mismatches to stdout for quick debugging.
        """
        try:
            if 'charger' not in getattr(data, 'node_types', []):
                return
            x_ch = data['charger'].x.detach().cpu().numpy()
            node_id_to_type = getattr(data, 'node_id_to_type', {})
            mismatches = 0
            for charger_id in env.charging_nodes:
                mapping = node_id_to_type.get(charger_id)
                if not mapping or mapping[0] != 'charger':
                    continue
                local_idx = int(mapping[1])
                if local_idx < 0 or local_idx >= x_ch.shape[0]:
                    continue
                feature_rate = float(x_ch[local_idx, 2]) if x_ch.shape[1] > 2 else None
                capacity = int(env.charging_station.charger_capacity.get(charger_id, 0))
                occupancy = len(env.charging_station.charger_occupancy.get(charger_id, []))
                expected_rate = (occupancy / capacity) if capacity > 0 else 0.0
                if feature_rate is None:
                    continue
                if abs(feature_rate - expected_rate) > tol:
                    print(f"[VERIFY] Mismatch for charger {charger_id}: feature_rate={feature_rate:.3f}, expected={expected_rate:.3f} (occ={occupancy}/{capacity})")
                    mismatches += 1
            if mismatches == 0:
                print("[VERIFY] Charger occupancy_rate features match env occupancy/capacity.")
        except Exception as e:
            print(f"[VERIFY] Error during charger feature verification: {e}")

    def _plot_edge_type_distribution(self, data, env, ax):
        """Plot distribution of edge types in the simplified design."""
        hv = self._homogeneous_view(data)
        edge_index = hv["edge_index"]
        node_types = hv["node_types"]

        # Count different edge types (Truck=0, Delivery=1, Charger=2)
        edge_counts = {
            "Truck↔Delivery": 0,
            "Truck↔Charger": 0,
            "Charger↔Charger": 0,
            "Charger↔Delivery": 0,
            "Delivery↔Delivery": 0,
        }

        for i in range(edge_index.shape[1]):
            src_type = node_types[edge_index[0, i]]
            dst_type = node_types[edge_index[1, i]]

            # Categorize edge (Truck=0, Delivery=1, Charger=2)
            if (src_type == 0 and dst_type == 1) or (src_type == 1 and dst_type == 0):
                edge_counts["Truck↔Delivery"] += 1
            elif (src_type == 0 and dst_type == 2) or (src_type == 2 and dst_type == 0):
                edge_counts["Truck↔Charger"] += 1
            elif src_type == 2 and dst_type == 2:
                edge_counts["Charger↔Charger"] += 1
            elif (src_type == 2 and dst_type == 1) or (src_type == 1 and dst_type == 2):
                edge_counts["Charger↔Delivery"] += 1
            elif src_type == 1 and dst_type == 1:
                edge_counts["Delivery↔Delivery"] += 1

        edge_types = list(edge_counts.keys())
        counts = list(edge_counts.values())

        colors_list = plt.cm.Set3(np.linspace(0, 1, len(edge_types)))
        ax.barh(edge_types, counts, color=colors_list, edgecolor="black", linewidth=1.5)

        ax.set_xlabel("Count", fontweight="bold")
        ax.set_title("Edge Type Distribution (Simplified Design)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (edge_type, count) in enumerate(zip(edge_types, counts)):
            if count > 0:  # Only show non-zero counts
                ax.text(
                    count + max(counts + [1]) * 0.01,
                    i,
                    str(count),
                    va="center",
                    fontweight="bold",
                )

    def _plot_networkx_layout(self, data, env, ax):
        """Plot graph using NetworkX layout."""
        # Build homogeneous view of HeteroData for visualization
        hv = self._homogeneous_view(data)
        edge_index = hv["edge_index"]
        num_nodes = hv["num_nodes"]

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])

        # Use spring layout for visualization
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax, edge_color="gray", alpha=0.3, arrowsize=15, width=1.5
        )

        # Get charger details for type information
        charger_details = env.transport_graph.get_charger_details()
        node_types = hv["node_types"]
        node_labels_info = hv["labels"]
        
        # Draw trucks and deliveries
        for node_type in [0, 1]:  # Truck and Delivery
            nodes_of_type = [i for i in range(num_nodes) if node_types[i] == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=nodes_of_type,
                    node_color=self.node_colors[node_type],
                    node_size=self.node_sizes[node_type],
                    label=self.node_type_names[node_type],
                    ax=ax,
                )
        
        # Draw chargers by type (Level2, DCFast, Mixed)
        charger_nodes = [i for i in range(num_nodes) if node_types[i] == 2]
        level2_nodes = []
        dcfast_nodes = []
        mixed_nodes = []
        
        for idx in charger_nodes:
            # Extract charger node ID from label
            label = node_labels_info.get(idx, "")
            if label.startswith('C'):
                charger_id = int(label[1:])  # Remove 'C' prefix
                if charger_id in charger_details:
                    types = charger_details[charger_id].get('types', {})
                    has_level2 = 'Level2' in types
                    has_dcfast = 'DCFast' in types
                    
                    if has_level2 and has_dcfast:
                        mixed_nodes.append(idx)
                    elif has_level2:
                        level2_nodes.append(idx)
                    elif has_dcfast:
                        dcfast_nodes.append(idx)
        
        # Draw Level2 chargers (green)
        if level2_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=level2_nodes,
                node_color='green', node_size=self.node_sizes[2],
                label='Charger (Level2)', ax=ax
            )
        
        # Draw DCFast chargers (red)
        if dcfast_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=dcfast_nodes,
                node_color='red', node_size=self.node_sizes[2],
                label='Charger (DCFast)', ax=ax
            )
        
        # Draw Mixed chargers (purple)
        if mixed_nodes:
            nx.draw_networkx_nodes(
                G, pos, nodelist=mixed_nodes,
                node_color='purple', node_size=self.node_sizes[2],
                label='Charger (Mixed)', ax=ax
            )

        # Build and draw node labels with real IDs
        nx.draw_networkx_labels(G, pos, labels=node_labels_info, ax=ax, font_size=7, font_weight="bold")

        ax.set_title("Graph Network Layout", fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.axis("off")

    def _plot_node_distribution(self, data, ax):
        """Plot distribution of node types."""
        hv = self._homogeneous_view(data)
        node_types_np = np.array(hv["node_types"], dtype=int)
        unique_vals, counts = np.unique(node_types_np, return_counts=True)

        type_names = [self.node_type_names[int(t)] for t in unique_vals]
        colors = [self.node_colors[int(t)] for t in unique_vals]

        ax.bar(
            type_names,
            counts,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Node Type Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (name, count) in enumerate(zip(type_names, counts)):
            ax.text(i, count + 0.1, str(count), ha="center", fontweight="bold")

    def _plot_node_features_heatmap(self, data, ax):
        """Plot heatmap of node features."""
        hv = self._homogeneous_view(data)
        x = hv["x"]

        # Normalize for better visualization
        x_norm = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0) + 1e-8)

        im = ax.imshow(x_norm.T, cmap="YlOrRd", aspect="auto")

        ax.set_xlabel("Node Index", fontweight="bold")
        ax.set_ylabel("Feature Dimension", fontweight="bold")
        ax.set_title("Node Features Heatmap (Normalized)", fontweight="bold")

        plt.colorbar(im, ax=ax, label="Feature Value (0-1)")

    def _plot_edge_statistics(self, data, env, ax):
        """Plot edge statistics for the new GNN design."""
        self._plot_edge_type_distribution(data, env, ax)

    def plot_feature_analysis(self, data, title: str = "Node Feature Analysis"):
        """
        Create detailed feature analysis plots.

        Args:
            data: PyTorch Geometric Data object
            title: Title for the plot
        """
        hv = self._homogeneous_view(data)
        x = hv["x"]
        num_features = x.shape[1]

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # Feature statistics
        ax1 = axes[0, 0]
        means = x.mean(axis=0)
        stds = x.std(axis=0)
        ax1.errorbar(
            range(num_features), means, yerr=stds, fmt="o-", capsize=5, linewidth=2
        )
        ax1.set_xlabel("Feature Index", fontweight="bold")
        ax1.set_ylabel("Value", fontweight="bold")
        ax1.set_title("Feature Mean ± Std", fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Feature ranges
        ax2 = axes[0, 1]
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        ax2.fill_between(range(num_features), mins, maxs, alpha=0.5, color="skyblue")
        ax2.plot(range(num_features), means, "r-o", linewidth=2, label="Mean")
        ax2.set_xlabel("Feature Index", fontweight="bold")
        ax2.set_ylabel("Value", fontweight="bold")
        ax2.set_title("Feature Range", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Feature distribution (violin plot style)
        ax3 = axes[1, 0]
        feature_samples = x[:, :5]  # First 5 features
        positions = range(5)
        bp = ax3.boxplot(feature_samples, positions=positions, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
        ax3.set_xlabel("Feature Index", fontweight="bold")
        ax3.set_ylabel("Value", fontweight="bold")
        ax3.set_title("Feature Distribution (First 5)", fontweight="bold")
        ax3.grid(axis="y", alpha=0.3)

        # Correlation heatmap
        ax4 = axes[1, 1]
        # Suppress warnings for features with zero variance
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.corrcoef(x.T)
            # Replace NaN values (from zero-variance features) with 0
            corr = np.nan_to_num(corr, nan=0.0)
        im = ax4.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax4.set_xlabel("Feature Index", fontweight="bold")
        ax4.set_ylabel("Feature Index", fontweight="bold")
        ax4.set_title("Feature Correlation Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax4, label="Correlation")

        plt.tight_layout()
        return fig

    def plot_action_graph(self, data, env, title: str = "Action Graph - Feasible Actions"):
        """
        Plot action-centric view showing each truck and its feasible actions.
        
        Creates subplots for each active truck showing:
        - Truck at center
        - Next delivery node (if exists)
        - All reachable chargers with current battery
        - Edge weights showing energy/time costs
        
        Args:
            data: PyTorch Geometric Data object
            env: EventDrivenTruckEnv instance
            title: Title for the plot
        """
        # Count active trucks
        active_trucks = [t for t in env.trucks if not t.failed and not t.is_complete]
        num_trucks = len(active_trucks)
        
        if num_trucks == 0:
            print("No active trucks to visualize")
            return None
        
        # Create subplots - one per truck
        ncols = min(2, num_trucks)
        nrows = (num_trucks + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 8 * nrows))
        fig.suptitle(title, fontsize=16, fontweight="bold")
        
        # Ensure axes is always a list
        if num_trucks == 1:
            axes = [axes]
        elif nrows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each truck's action graph
        for truck_idx, truck in enumerate(active_trucks):
            ax = axes[truck_idx]
            self._plot_single_truck_action_graph(truck, env, data, ax)
        
        # Hide unused subplots
        for idx in range(num_trucks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def _plot_single_truck_action_graph(self, truck, env, data, ax):
        """Plot action graph for a single truck."""
        G = nx.DiGraph()
        pos = {}
        node_colors = []
        node_sizes = []
        node_labels = {}
        
        # Central node: truck
        truck_node = f"T{truck.truck_id}"
        G.add_node(truck_node)
        pos[truck_node] = (0, 0)  # Center position
        node_colors.append(self.node_colors[0])  # Truck color
        node_sizes.append(1200)
        node_labels[truck_node] = f"Truck {truck.truck_id}\n{truck.current_battery:.0f}kWh\n({truck.get_battery_percentage():.0f}%)"
        
        current_battery = truck.current_battery
        current_location = truck.current_node
        
        # Determine truck state
        is_charging = truck.is_charging
        is_waiting = truck.truck_id in env.charging_station.charger_waitlist.get(current_location, [])
        is_routing = truck.route_destination is not None
        
        if is_charging:
            state_str = "CHARGING"
        elif is_waiting:
            state_str = "WAITING"
        elif is_routing:
            state_str = f"ROUTING→{truck.route_destination}"
        else:
            state_str = "READY"
        
        # Add edges and nodes based on state
        edge_labels = {}
        angle_step = 360 / max(1, (1 + len(env.charging_nodes)))  # Distribute nodes in circle
        angle = 0
        
        if is_charging or is_waiting:
            # Show only current charger
            charger_node = f"C{current_location}"
            G.add_node(charger_node)
            angle_rad = np.radians(angle)
            pos[charger_node] = (2 * np.cos(angle_rad), 2 * np.sin(angle_rad))
            node_colors.append(self.node_colors[2])  # Charger color
            node_sizes.append(800)
            
            # Get charger type
            charger_details = env.transport_graph.get_charger_details()
            charger_type_str = ""
            if current_location in charger_details:
                types = charger_details[current_location].get('types', {})
                if 'Level2' in types and 'DCFast' in types:
                    charger_type_str = " Mixed"
                elif 'Level2' in types:
                    charger_type_str = " Level2"
                elif 'DCFast' in types:
                    charger_type_str = " DCFast"
            
            node_labels[charger_node] = f"Charger\n{current_location}{charger_type_str}\n(current)"
            
            G.add_edge(truck_node, charger_node)
            edge_labels[(truck_node, charger_node)] = "0 kWh\n0.0 h"
            
        elif is_routing:
            # Show destination
            dest_node = truck.route_destination
            dest_node_name = f"D{dest_node}" if dest_node not in env.charging_nodes else f"C{dest_node}"
            G.add_node(dest_node_name)
            angle_rad = np.radians(angle)
            pos[dest_node_name] = (2 * np.cos(angle_rad), 2 * np.sin(angle_rad))
            
            if dest_node in env.charging_nodes:
                node_colors.append(self.node_colors[2])  # Charger
                
                # Get charger type
                charger_details = env.transport_graph.get_charger_details()
                charger_type_str = ""
                if dest_node in charger_details:
                    types = charger_details[dest_node].get('types', {})
                    if 'Level2' in types and 'DCFast' in types:
                        charger_type_str = " Mixed"
                    elif 'Level2' in types:
                        charger_type_str = " Level2"
                    elif 'DCFast' in types:
                        charger_type_str = " DCFast"
                
                node_labels[dest_node_name] = f"Charger\n{dest_node}{charger_type_str}\n(dest)"
            else:
                node_colors.append(self.node_colors[1])  # Delivery
                node_labels[dest_node_name] = f"Delivery\n{dest_node}\n(dest)"
            
            node_sizes.append(800)
            time_remaining = max(0.0, truck.route_arrival_time - env.global_clock) if truck.route_arrival_time else 0.0
            G.add_edge(truck_node, dest_node_name)
            edge_labels[(truck_node, dest_node_name)] = f"0 kWh\n{time_remaining:.1f} h"
            
        else:
            # READY: show next delivery + reachable chargers
            next_delivery = truck.get_next_delivery_target()
            
            # Add next delivery
            if next_delivery is not None:
                delivery_node = f"D{next_delivery}"
                G.add_node(delivery_node)
                angle_rad = np.radians(angle)
                pos[delivery_node] = (2 * np.cos(angle_rad), 2 * np.sin(angle_rad))
                node_colors.append(self.node_colors[1])  # Delivery color
                node_sizes.append(900)
                
                if next_delivery == current_location:
                    energy, time = 0.0, 0.0
                    node_labels[delivery_node] = f"Next Delivery\n{next_delivery}\n(here)"
                else:
                    energy = env.transport_graph.get_path_energy(current_location, next_delivery)
                    time = env.transport_graph.get_time_distance(current_location, next_delivery)
                    node_labels[delivery_node] = f"Next Delivery\n{next_delivery}"
                
                if energy < current_battery:
                    G.add_edge(truck_node, delivery_node)
                    edge_labels[(truck_node, delivery_node)] = f"{energy:.0f} kWh\n{time:.1f} h"
                else:
                    node_labels[delivery_node] += "\n⚠️ NO ENERGY"
                
                angle += angle_step
            
            # Add reachable chargers
            reachable_chargers = []
            for charger_id in env.charging_nodes:
                if charger_id == current_location:
                    energy, time = 0.0, 0.0
                else:
                    energy = env.transport_graph.get_path_energy(current_location, charger_id)
                    time = env.transport_graph.get_time_distance(current_location, charger_id)
                
                if energy < current_battery and not np.isinf(energy):
                    reachable_chargers.append((charger_id, energy, time))
            
            # Sort by energy (closest first) and take top 10
            # Get charger details for type information
            charger_details = env.transport_graph.get_charger_details()
            
            # Sort by energy (closest first) and take top 10
            reachable_chargers.sort(key=lambda x: x[1])
            for charger_id, energy, time in reachable_chargers[:10]:
                charger_node = f"C{charger_id}"
                G.add_node(charger_node)
                angle_rad = np.radians(angle)
                pos[charger_node] = (2 * np.cos(angle_rad), 2 * np.sin(angle_rad))
                
                # Get charger type and set color accordingly
                charger_color = self.node_colors[2]  # Default blue
                charger_type_str = ""
                if charger_id in charger_details:
                    types = charger_details[charger_id].get('types', {})
                    if 'Level2' in types and 'DCFast' in types:
                        charger_color = 'purple'
                        charger_type_str = " M"  # Mixed
                    elif 'Level2' in types:
                        charger_color = 'green'
                        charger_type_str = " L2"
                    elif 'DCFast' in types:
                        charger_color = 'red'
                        charger_type_str = " DC"
                
                node_colors.append(charger_color)
                node_sizes.append(600)
                
                # Get charger info
                occupancy = len(env.charging_station.charger_occupancy.get(charger_id, []))
                capacity = env.charging_station.charger_capacity.get(charger_id, 0)
                queue = len(env.charging_station.charger_waitlist.get(charger_id, []))
                
                if charger_id == current_location:
                    node_labels[charger_node] = f"Charger {charger_id}{charger_type_str}\n(here)\n{occupancy}/{capacity}"
                else:
                    node_labels[charger_node] = f"C{charger_id}{charger_type_str}\n{occupancy}/{capacity}"
                    if queue > 0:
                        node_labels[charger_node] += f"\nQ:{queue}"
                
                G.add_edge(truck_node, charger_node)
                edge_labels[(truck_node, charger_node)] = f"{energy:.0f} kWh\n{time:.1f} h"
                
                angle += angle_step
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=8, font_weight="bold")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, 
                              arrowsize=20, arrowstyle='->', width=2, connectionstyle="arc3,rad=0.1")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
        
        # Set title with truck info
        ax.set_title(
            f"Truck {truck.truck_id} - {state_str}\n"
            f"Location: {current_location} | Battery: {current_battery:.0f}/{truck.battery_capacity:.0f} kWh | "
            f"Deliveries: {truck.current_sequence_index}/{len(truck.delivery_sequence)-1}",
            fontweight="bold",
            fontsize=10
        )
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.node_colors[0], label='Truck'),
            mpatches.Patch(color=self.node_colors[1], label='Delivery'),
            mpatches.Patch(color=self.node_colors[2], label='Charger'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def plot_graph_info_text(self, data, env, title: str = "Graph Information"):
        """
        Create a text-based information plot.

        Args:
            data: PyTorch Geometric Data object
            env: EventDrivenTruckEnv instance
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

        # Gather statistics using homogeneous view
        hv = self._homogeneous_view(data)
        node_types = hv["node_types"]
        edge_index = hv["edge_index"]
        x = hv["x"]

        info_text = f"""
{'='*60}
{title}
{'='*60}

GRAPH STRUCTURE
───────────────
• Total Nodes: {len(node_types)}
• Total Edges: {edge_index.shape[1]}
• Node Features (padded): {x.shape}
• Edge Features: 2 (energy, time)

NODE TYPES BREAKDOWN
────────────────────
"""

        for node_type in [0, 1, 2]:  # Only 3 types now
            count = sum(1 for t in node_types if t == node_type)
            info_text += f"• {self.node_type_names[node_type]}: {count}\n"

        info_text += f"""
FEATURE STATISTICS
──────────────────
• Node Feature Dimension (padded): {x.shape[1]} (type-specific padded)
• Total Node Features: {x.size:,}
• Feature Mean: {x.mean():.4f}
• Feature Std: {x.std():.4f}
• Feature Min: {x.min():.4f}
• Feature Max: {x.max():.4f}

EDGE STATISTICS
───────────────
• Total Edges: {edge_index.shape[1]}
• Avg Degree: {2 * edge_index.shape[1] / max(1, len(node_types)):.2f}
• Edge Features: 2 (energy, time)
  - Dimension 0: Energy Distance (kWh)
  - Dimension 1: Time to Traverse (hours)

ENVIRONMENT STATE
─────────────────
• Active Truck ID: {data.active_truck_id.item() if hasattr(data, 'active_truck_id') else 'N/A'}
• Global Clock: {data.global_clock.item():.2f} hours
• Num Trucks: {data.num_trucks.item() if hasattr(data, 'num_trucks') else 'N/A'}
"""

        ax.text(
            0.05,
            0.95,
            info_text,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def _extract_node_types(self, data, env) -> List[int]:
        """Extract node types in homogeneous ordering: trucks, deliveries, chargers."""
        n_truck = data['truck'].x.shape[0] if hasattr(data, '__getitem__') and 'truck' in data.node_types else 0
        n_delivery = data['delivery'].x.shape[0] if hasattr(data, '__getitem__') and 'delivery' in data.node_types else 0
        n_charger = data['charger'].x.shape[0] if hasattr(data, '__getitem__') and 'charger' in data.node_types else 0
        return [0] * n_truck + [1] * n_delivery + [2] * n_charger

    def _build_node_labels(self, data) -> Dict[int, str]:
        """Build node labels using `node_id_to_type` and homogeneous ordering."""
        labels: Dict[int, str] = {}
        # Determine counts per type
        n_truck = data['truck'].x.shape[0] if 'truck' in data.node_types else 0
        n_delivery = data['delivery'].x.shape[0] if 'delivery' in data.node_types else 0
        n_charger = data['charger'].x.shape[0] if 'charger' in data.node_types else 0
        off_truck = 0
        off_delivery = off_truck + n_truck
        off_charger = off_delivery + n_delivery

        # Build reverse map: (type, local_idx) -> node_id
        reverse_map: Dict[Tuple[str, int], int] = {}
        if hasattr(data, 'node_id_to_type') and isinstance(data.node_id_to_type, dict):
            for nid, (tstr, lidx) in data.node_id_to_type.items():
                reverse_map[(tstr, int(lidx))] = int(nid)

        # Trucks
        for i in range(n_truck):
            node_id = reverse_map.get(('truck', i), i)
            labels[off_truck + i] = f"T{node_id}"
        # Deliveries
        for i in range(n_delivery):
            node_id = reverse_map.get(('delivery', i), i)
            labels[off_delivery + i] = f"D{node_id}"
        # Chargers
        for i in range(n_charger):
            node_id = reverse_map.get(('charger', i), i)
            labels[off_charger + i] = f"C{node_id}"
        return labels

    def _extract_node_types_tensor(self, data) -> torch.Tensor:
        """Return node types tensor in homogeneous ordering."""
        types = self._extract_node_types(data, None)
        if len(types) == 0:
            return torch.zeros((0,), dtype=torch.long)
        return torch.tensor(types, dtype=torch.long)

    def _homogeneous_view(self, data):
        """Construct a homogeneous view (x, edge_index, node_types, labels) from HeteroData.

        - Node order: trucks -> deliveries -> chargers
        - x is padded to max feature dimension across node types
        - edge_index indices are shifted by type offsets
        """
        # Counts per type
        node_types_present = getattr(data, 'node_types', [])
        n_truck = data['truck'].x.shape[0] if 'truck' in node_types_present else 0
        n_delivery = data['delivery'].x.shape[0] if 'delivery' in node_types_present else 0
        n_charger = data['charger'].x.shape[0] if 'charger' in node_types_present else 0
        off_truck = 0
        off_delivery = off_truck + n_truck
        off_charger = off_delivery + n_delivery
        num_nodes = n_truck + n_delivery + n_charger

        # Features (pad to max dim)
        dims = []
        xs = []
        for t in ['truck', 'delivery', 'charger']:
            if t in node_types_present:
                xt = data[t].x.detach().cpu().numpy()
                xs.append(xt)
                dims.append(xt.shape[1] if xt.size else 0)
            else:
                xs.append(np.zeros((0, 0), dtype=np.float32))
                dims.append(0)
        max_dim = max(dims + [0])
        padded_blocks = []
        for xt in xs:
            if xt.size == 0:
                padded = np.zeros((0, max_dim), dtype=np.float32)
            else:
                pad_width = max_dim - xt.shape[1]
                if pad_width > 0:
                    padded = np.concatenate([xt, np.zeros((xt.shape[0], pad_width), dtype=xt.dtype)], axis=1)
                else:
                    padded = xt
            padded_blocks.append(padded)
        x = np.concatenate(padded_blocks, axis=0) if num_nodes > 0 else np.zeros((0, max_dim), dtype=np.float32)

        # Node types list
        node_types = [0] * n_truck + [1] * n_delivery + [2] * n_charger

        # Labels
        labels = self._build_node_labels(data)

        # Edges: concatenate all edge types with shifted indices
        edges_src = []
        edges_dst = []
        edge_attrs = []
        def shift_for(ntype: str) -> int:
            return off_truck if ntype == 'truck' else (off_delivery if ntype == 'delivery' else off_charger)

        if hasattr(data, 'edge_types'):
            for (src_t, _, dst_t) in data.edge_types:
                ei = data[(src_t, 'to', dst_t)].edge_index
                if ei is None:
                    continue
                ei = ei.detach().cpu().numpy()
                if ei.shape[1] == 0:
                    continue
                src_off = shift_for(src_t)
                dst_off = shift_for(dst_t)
                edges_src.append(ei[0] + src_off)
                edges_dst.append(ei[1] + dst_off)
                ea = data[(src_t, 'to', dst_t)].edge_attr
                if ea is not None:
                    edge_attrs.append(ea.detach().cpu().numpy())
        if edges_src:
            src = np.concatenate(edges_src, axis=0)
            dst = np.concatenate(edges_dst, axis=0)
            edge_index = np.vstack([src, dst])
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)

        if edge_attrs:
            edge_attr = np.concatenate(edge_attrs, axis=0)
        else:
            edge_attr = np.zeros((0, 2), dtype=np.float32)

        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "node_types": node_types,
            "labels": labels,
            "num_nodes": num_nodes,
        }

    def save_figure(
        self, fig: List, index: int, output_dir: str = "/home/sorfanouda/EVPR/gnn_plots"
    ):
        """Save figures to files."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        # for i, fig in enumerate(figs):
        filepath = os.path.join(output_dir, f"gnn_plot_{index}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"✓ Saved: {filepath}")

        return output_dir


def visualize_gnn_state(config_path: str, num_steps: int = 5):
    """
    Main function to visualize GNN state over multiple steps.

    Args:
        config_path: Path to environment config
        num_steps: Number of steps to visualize
    """
    print("\n" + "=" * 80)
    print("GNN State Visualization")
    print("=" * 80)

    from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy

    # Initialize
    env = EventDrivenTruckEnv(config_path, verbose=True, enable_plotting=True)
    gnn_state = GNNStateSpace(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
    )
    policy = HeuristicPolicy(verbose=False)
    visualizer = GNNVisualizer()

    # Reset environment
    obs, info = env.reset(seed=0)

    figs = []

    # Plot initial state
    print(f"\nGenerating visualization for initial state...")
    data_initial = gnn_state.get_state_GNN(env)
    # Verify charger capacity-derived features
    try:
        visualizer.verify_charger_capacity_features(data_initial, env)
    except Exception as e:
        print(f"[VERIFY] Skipped charger feature verification: {e}")
    fig1 = visualizer.plot_graph_structure(
        data_initial, env, title="Initial GNN Graph State"
    )
    figs.append(fig1)

    fig2 = visualizer.plot_feature_analysis(
        data_initial, title="Initial Node Feature Analysis"
    )
    figs.append(fig2)

    fig3 = visualizer.plot_graph_info_text(
        data_initial, env, title="Initial Graph Information"
    )
    figs.append(fig3)
    
    fig4 = visualizer.plot_action_graph(
        data_initial, env, title="Initial Action Graph - Feasible Actions"
    )
    if fig4 is not None:
        figs.append(fig4)
    
    output_dir = visualizer.save_figure(fig1, index=0)
    output_dir = visualizer.save_figure(fig2, index=-1)
    output_dir = visualizer.save_figure(fig3, index=-2)
    if fig4 is not None:
        output_dir = visualizer.save_figure(fig4, index=-3)
    
    # Close figures to free memory
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    if fig4 is not None:
        plt.close(fig4)

    input("Press Enter to continue...")
    # Run steps and visualize
    for step in range(1, 10):
        action = policy.get_action(env)
        if action is None:
            print("No valid action available")
            break

        obs, reward, done, truncated, info = env.step(action)

        print(f"\nGenerating visualization for step {step}...")
        data = gnn_state.get_state_GNN(env)

        fig = visualizer.plot_graph_structure(
            data, env, title=f"GNN Graph State - Step {step}"
        )
        figs.append(fig)
        output_dir = visualizer.save_figure(fig, index=step)
        
        # Also generate action graph
        fig_action = visualizer.plot_action_graph(
            data, env, title=f"Action Graph - Step {step}"
        )
        if fig_action is not None:
            figs.append(fig_action)
            output_dir = visualizer.save_figure(fig_action, index=5 + step + 100)
            plt.close(fig_action)
        
        # Close main figure to free memory
        plt.close(fig)
        
        # Print some stats
        print(f"  Step {step}: Nodes={data.num_nodes}, Edges={data.num_edges}, Reward={reward:.2f}")
        input("Press Enter to continue...")

        if done or truncated:
            print("Episode finished!")
            break

    # Save all figures
    print("\nSaving figures...")
    

    print(f"\n✓ Visualization complete!")
    print(f"  - Generated {len(figs)} plots")
    print(f"  - Saved to: {output_dir}")

    # Show plots
    print("\nDisplaying plots...")
    plt.show()


if __name__ == "__main__":
    config_path = "/home/sorfanouda/EVPR/EVRoutingEnv/config_files/config.yaml"
    visualize_gnn_state(config_path, num_steps=3)

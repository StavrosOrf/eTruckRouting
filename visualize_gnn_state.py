"""
Visualization script for GNN state representation.

Creates visual plots of the GNN graph structure, showing:
- Node types (Depots, Trucks, Deliveries, Chargers) and their features
- Edge connections representing transportation network
- Graph statistics and topology evolution
- Feature distributions and correlations

Multi-Depot GNN Design:
- Nodes: Multiple depot nodes (one per unique truck starting position) + trucks + remaining deliveries + chargers
- Edges: Truck↔Depot (energy, time) + Depot↔Location (energy, time) + Location↔Location (energy, time)
- Edge Features: [energy_distance (kWh), time_to_traverse (hours)]
- Dynamic: Graph shrinks as deliveries complete
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import torch

sys.path.insert(0, "/home/sorfanouda/EVPR")

from truck_env.models.gnn_state_space import GNNStateSpace
from truck_env.models.event_driven_env import EventDrivenTruckEnv


class GNNVisualizer:
    """Visualizes GNN state representations."""

    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """Initialize visualizer with figure size."""
        self.figsize = figsize
        self.node_type_names = {0: "Depot", 1: "Truck", 2: "Delivery", 3: "Charger"}
        self.node_colors = {0: "#FFD700", 1: "#FF6B6B", 2: "#4ECDC4", 3: "#45B7D1"}
        self.node_sizes = {0: 700, 1: 800, 2: 600, 3: 600}

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

    def _plot_edge_type_distribution(self, data, env, ax):
        """Plot distribution of edge types in the multi-depot design."""
        edge_index = data.edge_index.cpu().numpy()
        node_types = self._extract_node_types(data, env)

        # Count different edge types
        edge_counts = {
            "Truck↔Depot": 0,
            "Depot↔Delivery": 0,
            "Depot↔Charger": 0,
            "Charger↔Charger": 0,
            "Charger↔Delivery": 0,
            "Delivery↔Delivery": 0,
        }

        for i in range(edge_index.shape[1]):
            src_type = node_types[edge_index[0, i]]
            dst_type = node_types[edge_index[1, i]]

            # Categorize edge
            if (src_type == 0 and dst_type == 1) or (src_type == 1 and dst_type == 0):
                edge_counts["Truck↔Depot"] += 1
            elif (src_type == 0 and dst_type == 2) or (src_type == 2 and dst_type == 0):
                edge_counts["Depot↔Delivery"] += 1
            elif (src_type == 0 and dst_type == 3) or (src_type == 3 and dst_type == 0):
                edge_counts["Depot↔Charger"] += 1
            elif src_type == 3 and dst_type == 3:
                edge_counts["Charger↔Charger"] += 1
            elif (src_type == 3 and dst_type == 2) or (src_type == 2 and dst_type == 3):
                edge_counts["Charger↔Delivery"] += 1
            elif src_type == 2 and dst_type == 2:
                edge_counts["Delivery↔Delivery"] += 1

        edge_types = list(edge_counts.keys())
        counts = list(edge_counts.values())

        colors_list = plt.cm.Set3(np.linspace(0, 1, len(edge_types)))
        ax.barh(edge_types, counts, color=colors_list, edgecolor="black", linewidth=1.5)

        ax.set_xlabel("Count", fontweight="bold")
        ax.set_title("Edge Type Distribution (Multi-Depot Design)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (edge_type, count) in enumerate(zip(edge_types, counts)):
            ax.text(
                count + max(counts) * 0.01,
                i,
                str(count),
                va="center",
                fontweight="bold",
            )

    def _plot_networkx_layout(self, data, env, ax):
        """Plot graph using NetworkX layout."""
        # Convert to NetworkX graph
        edge_index = data.edge_index.cpu().numpy()
        num_nodes = data.num_nodes

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

        # Draw nodes by type
        node_types = self._extract_node_types(data, env)
        for node_type in range(4):
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

        # Build and draw node labels with real IDs
        node_labels = self._build_node_labels(data)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=7, font_weight="bold")

        ax.set_title("Graph Network Layout", fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.axis("off")

    def _plot_node_distribution(self, data, ax):
        """Plot distribution of node types."""
        node_types = self._extract_node_types_tensor(data)
        unique_types, counts = node_types.unique(return_counts=True)

        type_names = [self.node_type_names[t.item()] for t in unique_types]
        colors = [self.node_colors[t.item()] for t in unique_types]

        ax.bar(
            type_names,
            counts.numpy(),
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
        ax.set_ylabel("Count", fontweight="bold")
        ax.set_title("Node Type Distribution", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for i, (name, count) in enumerate(zip(type_names, counts.numpy())):
            ax.text(i, count + 0.1, str(count), ha="center", fontweight="bold")

    def _plot_node_features_heatmap(self, data, ax):
        """Plot heatmap of node features."""
        x = data.x.cpu().numpy()

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
        x = data.x.cpu().numpy()
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
        corr = np.corrcoef(x.T)
        im = ax4.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax4.set_xlabel("Feature Index", fontweight="bold")
        ax4.set_ylabel("Feature Index", fontweight="bold")
        ax4.set_title("Feature Correlation Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax4, label="Correlation")

        plt.tight_layout()
        return fig

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

        # Gather statistics
        node_types = self._extract_node_types(data, env)
        edge_index = data.edge_index.cpu().numpy()

        info_text = f"""
{'='*60}
{title}
{'='*60}

GRAPH STRUCTURE
───────────────
• Total Nodes: {data.num_nodes}
• Total Edges: {data.num_edges}
• Node Features: {data.x.shape}
• Edge Features: {data.edge_attr.shape if data.edge_attr is not None else 'None'}

NODE TYPES BREAKDOWN
────────────────────
"""

        for node_type in [0, 1, 2, 3]:
            count = sum(1 for t in node_types if t == node_type)
            info_text += f"• {self.node_type_names[node_type]}: {count}\n"

        info_text += f"""
FEATURE STATISTICS
──────────────────
• Node Feature Dimension: {data.x.shape[1]} (type, position, battery/occupancy, etc.)
• Total Node Features: {data.x.numel():,}
• Feature Mean: {data.x.mean():.4f}
• Feature Std: {data.x.std():.4f}
• Feature Min: {data.x.min():.4f}
• Feature Max: {data.x.max():.4f}

EDGE STATISTICS
───────────────
• Total Edges: {data.num_edges}
• Avg Degree: {2 * data.num_edges / data.num_nodes:.2f}
• Edge Features: {data.edge_attr.shape if data.edge_attr is not None else 'N/A'}
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
        """Extract node types from graph structure (first feature dimension)."""
        # Node types are stored in the first feature dimension
        node_types = data.x[:, 0].int().tolist()
        return node_types

    def _build_node_labels(self, data) -> Dict[int, str]:
        """Build node labels with real IDs from node_list metadata."""
        node_labels = {}
        
        # Check if node_list metadata is available
        if hasattr(data, 'node_list') and data.node_list is not None:
            for idx, (node_type, node_id) in enumerate(data.node_list):
                if node_type == "depot":
                    node_labels[idx] = f"D{node_id}"
                elif node_type == "truck":
                    node_labels[idx] = f"T{node_id}"
                elif node_type == "delivery":
                    node_labels[idx] = f"Del{node_id}"
                elif node_type == "charger":
                    node_labels[idx] = f"C{node_id}"
                else:
                    node_labels[idx] = str(idx)
        else:
            # Fallback to just indices if metadata not available
            for i in range(data.num_nodes):
                node_labels[i] = str(i)
        
        return node_labels

    def _extract_node_types_tensor(self, data) -> torch.Tensor:
        """Extract first feature (node type) from node features."""
        # First feature is the node type (0, 1, 2, or 3)
        return data.x[:, 0]

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

    from truck_env.models.heuristic_policy import HeuristicPolicy

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
    output_dir = visualizer.save_figure(fig1, index=0)
    output_dir = visualizer.save_figure(fig2, index=-1)
    output_dir = visualizer.save_figure(fig3, index=-2)

    # Run steps and visualize
    for step in range(1, 20):
        action = policy.get_action(env)
        if action is None:
            break

        obs, reward, done, truncated, info = env.step(action)

        print(f"\nGenerating visualization for step {step}...")
        data = gnn_state.get_state_GNN(env)

        fig = visualizer.plot_graph_structure(
            data, env, title=f"GNN Graph State - Step {step}"
        )
        # figs.append(fig)

        output_dir = visualizer.save_figure(fig, index=5)
        input("Press Enter to continue...")

        if done or truncated:
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
    config_path = "/home/sorfanouda/EVPR/truck_env/config_files/config.yaml"
    visualize_gnn_state(config_path, num_steps=3)

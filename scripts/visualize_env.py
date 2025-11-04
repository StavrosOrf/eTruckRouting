"""
Interactive visualization for HierarchicalTruckRoutingEnv with geolocated trucks and graph.
Shows trucks moving on the graph in real-time with step-by-step updates.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import networkx as nx
import numpy as np
from truck_env.truck_env import HierarchicalTruckRoutingEnv
from truck_env.utils import GRAPH_MAPPINGS
import os


class TruckEnvVisualizer:
    """Visualizer for truck routing environment with real-time updates."""
    
    def __init__(self, env, save_frames=True, output_dir="visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            env: HierarchicalTruckRoutingEnv instance
            save_frames: Whether to save individual frames
            output_dir: Directory to save visualizations
        """
        self.env = env
        self.save_frames = save_frames
        self.output_dir = output_dir
        self.step_count = 0
        
        if save_frames:
            os.makedirs(output_dir, exist_ok=True)
        
        # Try to get geographical coordinates from graph
        self.has_geo_coords = self._check_geo_coords()
        
        # Get graph layout
        self.pos = self._get_graph_layout()
        
        # Color scheme for trucks
        self.truck_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
    def _check_geo_coords(self):
        """Check if graph nodes have geographical coordinates."""
        sample_node = list(self.env.graph.nodes())[0]
        node_data = self.env.graph.nodes[sample_node]
        return 'lat' in node_data or 'latitude' in node_data or 'y' in node_data
    
    def _get_graph_layout(self):
        """Get node positions for visualization."""
        G = self.env.graph
        
        # Try to use geographical coordinates if available
        if self.has_geo_coords:
            pos = {}
            for node in G.nodes():
                node_data = G.nodes[node]
                # Try different possible coordinate field names
                lat = node_data.get('lat') or node_data.get('latitude') or node_data.get('y')
                lon = node_data.get('lon') or node_data.get('longitude') or node_data.get('x')
                
                if lat is not None and lon is not None:
                    pos[node] = (lon, lat)  # (x, y) for matplotlib
                else:
                    # Fallback to spring layout for this node
                    pos[node] = (0, 0)
            
            # If we got valid positions, use them
            if any(p != (0, 0) for p in pos.values()):
                return pos
        
        # Fallback to spring layout if no geo coords or all zeros
        print("No geographical coordinates found. Using spring layout.")
        
        # For large graphs, use a faster layout
        if G.number_of_nodes() > 100:
            # Use spring layout with fewer iterations for speed
            return nx.spring_layout(G, k=0.5, iterations=20, seed=42)
        else:
            return nx.spring_layout(G, seed=42)
    
    def visualize_step(self, step_num=None, actions=None, rewards=None, save=True):
        """
        Visualize the current state of the environment.
        
        Args:
            step_num: Current step number
            actions: Dictionary of actions taken
            rewards: Dictionary of rewards received
            save: Whether to save this frame
        """
        self.ax.clear()
        
        # Set title
        title = f"Truck Routing Environment"
        if step_num is not None:
            title += f" - Step {step_num}"
        if self.has_geo_coords:
            title += " (Geolocated)"
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Draw the graph
        self._draw_graph(step_num)
        
        # Draw trucks
        self._draw_trucks(actions, rewards)
        
        # Draw legend
        self._draw_legend()
        
        # Add grid and labels
        self.ax.grid(True, alpha=0.3)
        if self.has_geo_coords:
            self.ax.set_xlabel('Longitude', fontsize=12)
            self.ax.set_ylabel('Latitude', fontsize=12)
        else:
            self.ax.set_xlabel('X', fontsize=12)
            self.ax.set_ylabel('Y', fontsize=12)
        
        plt.tight_layout()
        
        # Save frame if requested
        if save and self.save_frames and step_num is not None:
            filename = os.path.join(self.output_dir, f"step_{step_num:04d}.png")
            plt.savefig(filename, dpi=80, bbox_inches='tight')  # Reduced DPI for speed
            print(f"💾 Saved frame: {filename}")
    
    def _draw_graph(self, step_num=None):
        """Draw the graph with nodes and edges (optimized for large graphs)."""
        G = self.env.graph
        
        # For large graphs, don't draw all edges (too slow)
        num_edges = G.number_of_edges()
        if num_edges > 1000:
            print(f"⚠️  Skipping edge rendering ({num_edges} edges - too many for visualization)")
        else:
            # Draw edges (roads) in light gray
            nx.draw_networkx_edges(
                G, self.pos,
                edge_color='lightgray',
                alpha=0.3,
                width=0.5,
                ax=self.ax
            )
        
        # Identify charging stations and regular nodes
        charger_nodes = set(self.env.charger_configs.keys())
        regular_nodes = [n for n in G.nodes() if n not in charger_nodes]
        
        # For large graphs, only draw a subset of regular nodes
        num_regular = len(regular_nodes)
        if num_regular > 500:
            # Sample nodes to draw
            import random
            sample_size = min(100, num_regular)
            regular_nodes = random.sample(regular_nodes, sample_size)
            if step_num == 0:  # Only print once
                print(f"⚠️  Drawing {sample_size}/{num_regular} regular nodes for performance")
        
        # Draw regular nodes (small, light blue)
        if regular_nodes:
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=regular_nodes,
                node_color='lightblue',
                node_size=30,
                alpha=0.6,
                ax=self.ax
            )
        
        # Draw charging stations (larger, yellow diamonds)
        if charger_nodes:
            nx.draw_networkx_nodes(
                G, self.pos,
                nodelist=list(charger_nodes),
                node_color='gold',
                node_size=150,
                node_shape='D',
                alpha=0.8,
                ax=self.ax,
                edgecolors='orange',
                linewidths=2
            )
    
    def _draw_trucks(self, actions=None, rewards=None):
        """Draw trucks on the graph."""
        for i, truck in enumerate(self.env.trucks):
            color = self.truck_colors[i % len(self.truck_colors)]
            current_node = truck['current_node']
            dest_node = truck['destination_node']
            
            # Get position
            if current_node in self.pos:
                x, y = self.pos[current_node]
                
                # Draw truck as a large marker
                truck_done = current_node == dest_node or truck['current_battery'] <= 0
                marker = 'v' if truck_done else 'o'  # Triangle if done, circle if active
                size = 400 if truck_done else 300
                
                self.ax.scatter(
                    x, y,
                    s=size,
                    c=color,
                    marker=marker,
                    edgecolors='black',
                    linewidths=2,
                    zorder=10,
                    alpha=0.9,
                    label=f'Truck {i}'
                )
                
                # Add truck label with battery level
                battery_pct = 100 * truck['current_battery'] / truck['battery_capacity']
                label_text = f"T{i}\n{battery_pct:.0f}%"
                
                # Add charging indicator
                if truck['is_charging']:
                    label_text += "\n⚡"
                
                self.ax.text(
                    x, y + 0.02,  # Slightly above truck
                    label_text,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black'),
                    zorder=11
                )
                
                # Draw destination marker
                if dest_node in self.pos and dest_node != current_node:
                    dest_x, dest_y = self.pos[dest_node]
                    self.ax.scatter(
                        dest_x, dest_y,
                        s=200,
                        c=color,
                        marker='*',
                        edgecolors='black',
                        linewidths=1.5,
                        zorder=9,
                        alpha=0.6
                    )
                    
                    # Draw line from truck to destination
                    self.ax.plot(
                        [x, dest_x], [y, dest_y],
                        color=color,
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.4,
                        zorder=1
                    )
                
                # Display action and reward if provided
                if actions or rewards:
                    info_text = []
                    high_agent = f"truck_{i}_route_planner"
                    low_agent = f"truck_{i}_charge_manager"
                    
                    if actions:
                        if high_agent in actions:
                            info_text.append(f"Route: {actions[high_agent]}")
                        if low_agent in actions:
                            action_names = ["Wait", "Start⚡", "Stop⚡", "Wait⚡"]
                            action_val = actions[low_agent]
                            if action_val < len(action_names):
                                info_text.append(f"{action_names[action_val]}")
                    
                    if rewards:
                        total_reward = 0
                        if high_agent in rewards:
                            total_reward += rewards[high_agent]
                        if low_agent in rewards:
                            total_reward += rewards[low_agent]
                        if abs(total_reward) > 0.01:
                            info_text.append(f"R: {total_reward:.1f}")
                    
                    if info_text:
                        self.ax.text(
                            x, y - 0.02,  # Slightly below truck
                            "\n".join(info_text),
                            fontsize=7,
                            ha='center',
                            va='top',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                            zorder=11
                        )
    
    def _draw_legend(self):
        """Draw legend explaining the visualization."""
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Regular Nodes'),
            mpatches.Patch(color='gold', label='Charging Stations'),
        ]
        
        # Add truck legends
        for i, truck in enumerate(self.env.trucks):
            color = self.truck_colors[i % len(self.truck_colors)]
            battery_pct = 100 * truck['current_battery'] / truck['battery_capacity']
            dist = truck['total_distance']
            status = "✓" if truck['current_node'] == truck['destination_node'] else "→"
            label = f"Truck {i} ({battery_pct:.0f}%, {dist:.0f}km) {status}"
            legend_elements.append(mpatches.Patch(color=color, label=label))
        
        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=9,
            framealpha=0.9
        )
    
    def run_episode(self, max_steps=50, delay=0.5, random_actions=True):
        """
        Run an episode with visualization.
        
        Args:
            max_steps: Maximum number of steps
            delay: Delay between steps (seconds)
            random_actions: If True, use random actions; otherwise prompt for actions
        """
        print("\n" + "="*80)
        print("🎬 STARTING VISUALIZED EPISODE")
        print("="*80 + "\n")
        
        obs, info = self.env.reset()
        
        # Initial visualization
        self.visualize_step(step_num=0, save=True)
        
        for step in range(1, max_steps + 1):
            print(f"\n{'='*80}")
            print(f"📍 STEP {step}")
            print(f"{'='*80}")
            
            # Get actions
            if random_actions:
                actions = {
                    agent_id: self.env.get_action_space(agent_id).sample()
                    for agent_id in self.env.agents
                }
                print("🎲 Using random actions")
            else:
                actions = self._get_user_actions()
            
            # Execute step
            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            
            # Visualize
            self.visualize_step(step_num=step, actions=actions, rewards=rewards, save=True)
            
            # Print summary
            total_reward = sum(rewards.values())
            print(f"💰 Total reward: {total_reward:.2f}")
            print(f"⏱️  Global time: {self.env.global_time:.2f}")
            
            for i, truck in enumerate(self.env.trucks):
                battery_pct = 100 * truck['current_battery'] / truck['battery_capacity']
                print(f"  Truck {i}: Node {truck['current_node']}, Battery {battery_pct:.1f}%, "
                      f"Distance {truck['total_distance']:.1f}km")
            
            # Check if done
            if terminateds.get("__all__", False):
                print(f"\n✅ Episode completed at step {step}")
                break
        
        print("\n" + "="*80)
        print("🏁 EPISODE FINISHED")
        print(f"📁 Frames saved to: {self.output_dir}/")
        print("="*80 + "\n")
    
    def _get_user_actions(self):
        """Get actions from user input (for interactive mode)."""
        actions = {}
        for agent_id in self.env.agents:
            action_space = self.env.get_action_space(agent_id)
            print(f"{agent_id}: Enter action (0-{action_space.n - 1}): ", end="")
            try:
                action = int(input())
                if 0 <= action < action_space.n:
                    actions[agent_id] = action
                else:
                    print(f"Invalid action. Using random action.")
                    actions[agent_id] = action_space.sample()
            except:
                print(f"Invalid input. Using random action.")
                actions[agent_id] = action_space.sample()
        return actions


def main():
    """Main visualization demo."""
    print("\n" + "="*80)
    print("🚚 TRUCK ROUTING ENVIRONMENT VISUALIZATION")
    print("="*80 + "\n")
    
    # Create environment
    print("🏗️  Creating environment...")
    env = HierarchicalTruckRoutingEnv(config={"verbose": False, "debug": False})
    
    print(f"✅ Environment created!")
    print(f"  📊 Trucks: {env.num_trucks}")
    print(f"  📊 Nodes: {env.graph.number_of_nodes()}")
    print(f"  📊 Edges: {env.graph.number_of_edges()}")
    print(f"  📊 Chargers: {len(env.charger_configs)}")
    
    # Create visualizer
    print("\n🎨 Initializing visualizer...")
    viz = TruckEnvVisualizer(env, save_frames=True, output_dir="visualizations")
    
    print("✅ Visualizer ready!")
    print(f"  📍 Using {'geographical coordinates' if viz.has_geo_coords else 'spring layout'}")
    
    # Run episode
    print("\n" + "="*80)
    print("Starting episode visualization...")
    print("Frames will be saved to 'visualizations/' directory")
    print("="*80 + "\n")
    
    viz.run_episode(max_steps=20, delay=1.0, random_actions=True)


if __name__ == "__main__":
    main()

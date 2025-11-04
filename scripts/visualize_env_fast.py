"""
Fast visualization for HierarchicalTruckRoutingEnv showing trucks, destinations, and charging stations.
Optimized for large graphs by avoiding full graph rendering.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from truck_env.truck_env import HierarchicalTruckRoutingEnv
import os


class FastTruckVisualizer:
    """Fast visualizer for truck routing environment - shows only trucks, destinations, and chargers."""
    
    def __init__(self, env, save_frames=True, output_dir="visualizations"):
        """
        Initialize the fast visualizer.
        
        Args:
            env: HierarchicalTruckRoutingEnv instance
            save_frames: Whether to save frames to disk
            output_dir: Directory to save frames
        """
        self.env = env
        self.save_frames = save_frames
        self.output_dir = output_dir
        
        # Create output directory
        if self.save_frames:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Get node positions (use node IDs as coordinates for simplicity)
        self.node_positions = self._get_node_positions()
        
        # Truck colors
        self.truck_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
    def _get_node_positions(self):
        """Get simplified node positions using node IDs arranged in a grid."""
        nodes = list(self.env.graph.nodes())
        n = len(nodes)
        
        # Arrange nodes in a square grid
        grid_size = int(np.ceil(np.sqrt(n)))
        positions = {}
        
        for idx, node in enumerate(nodes):
            row = idx // grid_size
            col = idx % grid_size
            positions[node] = (col, row)
        
        return positions
    
    def visualize_step(self, step_num=None, actions=None, rewards=None, save=True):
        """
        Visualize a single step.
        
        Args:
            step_num: Current step number
            actions: Dictionary of actions taken
            rewards: Dictionary of rewards received
            save: Whether to save the frame
        """
        self.ax.clear()
        
        # Set title
        title = f"Truck Routing Environment (Fast View)"
        if step_num is not None:
            title += f" - Step {step_num}"
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Draw charging stations
        self._draw_chargers()
        
        # Draw trucks and destinations
        self._draw_trucks(actions, rewards)
        
        # Draw legend
        self._draw_legend()
        
        # Add grid and labels
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('Grid X', fontsize=12)
        self.ax.set_ylabel('Grid Y', fontsize=12)
        
        plt.tight_layout()
        
        # Save frame if requested
        if save and self.save_frames and step_num is not None:
            filename = os.path.join(self.output_dir, f"step_{step_num:04d}.png")
            plt.savefig(filename, dpi=80)
            print(f"💾 Saved frame: {filename}")
    
    def _draw_chargers(self):
        """Draw charging stations on the plot."""
        charger_nodes = list(self.env.charger_configs.keys())
        if not charger_nodes:
            return
        
        # Get positions
        charger_positions = [self.node_positions.get(node, (0, 0)) for node in charger_nodes]
        if charger_positions:
            x_coords, y_coords = zip(*charger_positions)
            self.ax.scatter(
                x_coords, y_coords,
                s=200,
                c='gold',
                marker='D',
                alpha=0.8,
                edgecolors='orange',
                linewidths=2,
                label='Charging Stations',
                zorder=2
            )
    
    def _draw_trucks(self, actions=None, rewards=None):
        """Draw trucks and their destinations."""
        for i, truck in enumerate(self.env.trucks):
            color = self.truck_colors[i % len(self.truck_colors)]
            current_node = truck['current_node']
            dest_node = truck['destination_node']
            
            # Get positions
            current_pos = self.node_positions.get(current_node, (0, 0))
            dest_pos = self.node_positions.get(dest_node, (0, 0))
            
            # Determine if truck is done
            truck_done = current_node == dest_node or truck['current_battery'] <= 0
            is_charging = truck.get('is_charging', False)
            
            # Draw line from truck to destination
            if not truck_done:
                self.ax.plot(
                    [current_pos[0], dest_pos[0]],
                    [current_pos[1], dest_pos[1]],
                    color=color,
                    linestyle='--',
                    linewidth=2,
                    alpha=0.4,
                    zorder=1
                )
            
            # Draw destination
            self.ax.scatter(
                dest_pos[0], dest_pos[1],
                s=300,
                c=color,
                marker='*',
                alpha=0.6,
                edgecolors='black',
                linewidths=1,
                zorder=3
            )
            
            # Draw truck
            marker = 'v' if truck_done else 'o'
            size = 600 if is_charging else 500
            edge_color = 'yellow' if is_charging else 'black'
            edge_width = 4 if is_charging else 2
            
            self.ax.scatter(
                current_pos[0], current_pos[1],
                s=size,
                c=color,
                marker=marker,
                alpha=0.9,
                edgecolors=edge_color,
                linewidths=edge_width,
                zorder=4
            )
            
            # Add truck label with battery and distance
            battery_pct = 100 * truck['current_battery'] / truck['battery_capacity']
            dist_km = truck['total_distance']
            
            label_text = f"T{i}\n{battery_pct:.0f}%\n{dist_km:.0f}km"
            if is_charging:
                label_text = f"T{i}⚡\n{battery_pct:.0f}%\n{dist_km:.0f}km"
            
            self.ax.text(
                current_pos[0], current_pos[1] - 0.5,
                label_text,
                fontsize=9,
                ha='center',
                va='top',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                zorder=5
            )
            
            # Add action/reward info if provided
            if actions or rewards:
                info_parts = []
                high_agent = f"truck_{i}_route_planner"
                low_agent = f"truck_{i}_charge_manager"
                
                if actions:
                    if high_agent in actions:
                        next_node = actions[high_agent]
                        info_parts.append(f"→{next_node}")
                    if low_agent in actions:
                        action_names = ["Wait", "Start⚡", "Stop⚡", "Wait⚡"]
                        action_val = actions[low_agent]
                        if action_val < len(action_names):
                            info_parts.append(action_names[action_val])
                
                if rewards and info_parts:
                    total_r = rewards.get(high_agent, 0) + rewards.get(low_agent, 0)
                    if abs(total_r) > 0.01:
                        info_parts.append(f"R:{total_r:.1f}")
                
                if info_parts:
                    self.ax.text(
                        current_pos[0], current_pos[1] + 0.5,
                        " | ".join(info_parts),
                        fontsize=7,
                        ha='center',
                        va='bottom',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.9),
                        zorder=5
                    )
    
    def _draw_legend(self):
        """Draw legend showing truck status."""
        legend_elements = [
            mpatches.Patch(color='gold', label='Charging Stations'),
        ]
        
        # Add truck info
        for i, truck in enumerate(self.env.trucks):
            color = self.truck_colors[i % len(self.truck_colors)]
            battery_pct = 100 * truck['current_battery'] / truck['battery_capacity']
            dist_km = truck['total_distance']
            
            status = "✓" if truck['current_node'] == truck['destination_node'] else "→"
            label = f"Truck {i}: {battery_pct:.0f}% bat, {dist_km:.0f}km {status}"
            legend_elements.append(mpatches.Patch(color=color, label=label))
        
        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=9,
            framealpha=0.95
        )
    
    def run_episode(self, max_steps=50, random_actions=True):
        """
        Run an episode with visualization.
        
        Args:
            max_steps: Maximum number of steps
            random_actions: If True, use random actions
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
                actions = {
                    agent_id: self.env.get_action_space(agent_id).sample()
                    for agent_id in self.env.agents
                }
            
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
                status = "⚡" if truck.get('is_charging', False) else ""
                print(f"  Truck {i}{status}: Node {truck['current_node']}, "
                      f"Battery {battery_pct:.1f}%, Distance {truck['total_distance']:.1f}km")
            
            # Check if done
            if terminateds.get("__all__", False):
                print(f"\n✅ Episode completed at step {step}")
                break
        
        print("\n" + "="*80)
        print("🏁 EPISODE FINISHED")
        print(f"📁 Frames saved to: {self.output_dir}/")
        print("="*80 + "\n")
        
        plt.close()


def main():
    """Main function to run the visualization."""
    print("\n" + "="*80)
    print("🚚 FAST TRUCK ROUTING ENVIRONMENT VISUALIZATION")
    print("="*80 + "\n")
    
    # Create environment
    print("🏗️  Creating environment...")
    env = HierarchicalTruckRoutingEnv()
    print("✅ Environment created!")
    print(f"  📊 Trucks: {len(env.trucks)}")
    print(f"  📊 Nodes: {env.graph.number_of_nodes()}")
    print(f"  📊 Edges: {env.graph.number_of_edges()}")
    print(f"  📊 Chargers: {len(env.charger_configs)}")
    
    # Create visualizer
    print("\n🎨 Initializing fast visualizer...")
    visualizer = FastTruckVisualizer(env)
    print("✅ Visualizer ready!")
    
    print("\n" + "="*80)
    print("Starting episode visualization...")
    print("Frames will be saved to 'visualizations/' directory")
    print("="*80 + "\n")
    
    # Run episode
    visualizer.run_episode(max_steps=20, random_actions=True)


if __name__ == "__main__":
    main()

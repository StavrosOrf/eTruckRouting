"""
Simple text-based visualizer for SimpleTruckEnv
Shows truck states, actions, and rewards at each step
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_truck_env import SimpleTruckEnv, load_config
import numpy as np


class TruckVisualizer:
    """Simple text-based visualizer for truck environment."""
    
    def __init__(self, env: SimpleTruckEnv):
        self.env = env
        
    def _get_battery_bar(self, percentage: float, width: int = 20) -> str:
        """Create a text-based battery bar."""
        filled = int(percentage / 100 * width)
        empty = width - filled
        
        if percentage > 70:
            color = "🟢"
        elif percentage > 30:
            color = "🟡"
        else:
            color = "🔴"
        
        bar = "█" * filled + "░" * empty
        return f"{color} [{bar}] {percentage:5.1f}%"
    
    def _get_action_description(self, action: int) -> str:
        """Get human-readable action description."""
        if action < self.env.num_charging_nodes:
            node = self.env.charging_nodes[action]
            return f"Go to charger @ node {node}"
        elif action == self.env.num_charging_nodes:
            return "Go to next delivery"
        else:
            charge_idx = action - self.env.num_navigation_actions
            hours = charge_idx + 1
            return f"Charge for {hours}h"
    
    def _get_status_icon(self, truck_state: dict) -> str:
        """Get status icon for truck."""
        if truck_state['is_complete']:
            return "✅"
        elif truck_state['failed']:
            return "❌"
        elif truck_state['is_charging']:
            return "🔌"
        else:
            return "🚛"
    
    def print_header(self):
        """Print episode header."""
        print("\n" + "="*100)
        print(f"{'TRUCK ROUTING VISUALIZATION':^100}")
        print("="*100)
        print(f"Trucks: {self.env.num_trucks} | Max Steps: {self.env.max_steps} | " +
              f"Charging Stations: {self.env.num_charging_nodes}")
        print("="*100 + "\n")
    
    def print_initial_state(self, info: dict):
        """Print initial state of all trucks."""
        print("📋 INITIAL STATE")
        print("-" * 100)
        
        for truck_state in info['trucks']:
            truck_id = truck_state['truck_id']
            truck_type = truck_state['truck_type']
            battery = truck_state['battery_percentage']
            deliveries = truck_state['deliveries_remaining']
            start_node = truck_state['current_node']
            
            print(f"  Truck {truck_id} ({truck_type.upper()}):")
            print(f"    Start Location: Node {start_node}")
            print(f"    Battery: {self._get_battery_bar(battery)}")
            print(f"    Deliveries: {deliveries} stops")
            print(f"    Route: {truck_state['delivery_sequence']}")
            print()
    
    def print_step(self, step: int, actions: np.ndarray, info: dict, rewards: dict):
        """Print step information."""
        print(f"\n{'─'*100}")
        print(f"STEP {step}")
        print(f"{'─'*100}")
        
        # Print each truck's state and action
        for truck_idx, truck_state in enumerate(info['trucks']):
            truck_id = truck_state['truck_id']
            action = actions[truck_idx]
            reward = rewards.get(truck_id, 0.0)
            
            status = self._get_status_icon(truck_state)
            
            print(f"\n{status} TRUCK {truck_id}")
            print(f"  {'─'*95}")
            
            # Current state
            print(f"  Location:    Node {truck_state['current_node']}")
            print(f"  Battery:     {self._get_battery_bar(truck_state['battery_percentage'])}")
            print(f"  Deliveries:  {truck_state['deliveries_remaining']} remaining")
            print(f"  Time:        {truck_state['total_time']:.2f} hours")
            print(f"  Distance:    {truck_state['total_distance']:.2f} km")
            
            # Action taken
            if not truck_state['is_complete'] and not truck_state['failed']:
                action_desc = self._get_action_description(action)
                print(f"  Action:      {action_desc}")
                print(f"  Reward:      {reward:+.2f}")
            else:
                status_text = "COMPLETE" if truck_state['is_complete'] else "FAILED"
                print(f"  Status:      {status_text}")
                print(f"  Reward:      {reward:+.2f}")
    
    def print_summary(self, info: dict, total_steps: int):
        """Print episode summary."""
        print("\n" + "="*100)
        print(f"{'EPISODE SUMMARY':^100}")
        print("="*100)
        
        print(f"\nTotal Steps: {total_steps}")
        print(f"Total Reward: {info['episode_reward']:.2f}")
        print(f"All Complete: {'YES ✅' if info['all_complete'] else 'NO ❌'}")
        print(f"Any Failed: {'YES ❌' if info['any_failed'] else 'NO ✅'}")
        
        print(f"\n{'TRUCK STATISTICS':^100}")
        print("-" * 100)
        
        for truck_state in info['trucks']:
            truck_id = truck_state['truck_id']
            status = "COMPLETE ✅" if truck_state['is_complete'] else ("FAILED ❌" if truck_state['failed'] else "INCOMPLETE ⏸️")
            
            print(f"\nTruck {truck_id}: {status}")
            print(f"  Total Time:     {truck_state['total_time']:.2f} hours")
            print(f"  Total Distance: {truck_state['total_distance']:.2f} km")
            print(f"  Final Battery:  {truck_state['battery_percentage']:.1f}%")
            print(f"  Deliveries:     {len(truck_state['delivery_sequence']) - 1 - truck_state['deliveries_remaining']}/{len(truck_state['delivery_sequence']) - 1}")
            print(f"  Charging Sessions: {truck_state['num_charging_sessions']}")
            print(f"  Charging Time:  {truck_state['total_charging_time']:.2f} hours")
        
        print("\n" + "="*100 + "\n")


def run_visualization(
    num_trucks: int = 2,
    num_stops: int = 3,
    max_steps: int = 50,
    strategy: str = "random",
    seed: int = 42,
    show_every_step: bool = True
):
    """
    Run a visualization of the truck environment.
    
    Args:
        num_trucks: Number of trucks
        num_stops: Number of delivery stops per truck
        max_steps: Maximum steps
        strategy: "random", "greedy", or "smart"
        seed: Random seed
        show_every_step: If False, only show summary
    """
    # Create environment
    config = load_config()
    config['advanced']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    config['environment']['max_steps'] = max_steps
    config['environment']['verbose'] = False
    
    env = SimpleTruckEnv(config=config)
    visualizer = TruckVisualizer(env)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Print header and initial state
    visualizer.print_header()
    visualizer.print_initial_state(info)
    
    # Track individual truck rewards
    truck_step_rewards = {i: [] for i in range(num_trucks)}
    
    # Run episode
    for step in range(max_steps):
        # Select action based on strategy
        if strategy == "random":
            action = env.action_space.sample()
        elif strategy == "greedy":
            # Always go to next delivery
            action = np.array([env.num_charging_nodes] * num_trucks)
        elif strategy == "smart":
            # Smart strategy: charge when low, otherwise deliver
            action = []
            for truck_state in info['trucks']:
                battery_pct = truck_state['battery_percentage']
                at_charger = truck_state['current_node'] in env.charging_nodes
                
                if battery_pct < 20.0:
                    # Critical battery - go to charger
                    truck_action = 0  # First charger
                elif battery_pct < 50.0 and at_charger:
                    # At charger with medium battery - charge for 2h
                    truck_action = env.num_navigation_actions + 1
                else:
                    # Go to delivery
                    truck_action = env.num_charging_nodes
                
                action.append(truck_action)
            action = np.array(action)
        else:
            action = env.action_space.sample()
        
        # Store previous info to calculate per-truck rewards
        prev_info = info
        
        # Execute step
        obs, total_reward, terminated, truncated, info = env.step(action)
        
        # Calculate per-truck rewards (approximation based on state changes)
        truck_rewards = {}
        for truck_idx, truck_state in enumerate(info['trucks']):
            truck_id = truck_state['truck_id']
            prev_truck = prev_info['trucks'][truck_idx]
            
            # Estimate reward based on time difference
            time_diff = truck_state['total_time'] - prev_truck['total_time']
            base_reward = -time_diff  # Time penalty
            
            # Add delivery bonus if deliveries decreased
            if truck_state['deliveries_remaining'] < prev_truck['deliveries_remaining']:
                base_reward += config.get('rewards', {}).get('delivery_bonus', 50.0)
            
            truck_rewards[truck_id] = base_reward
            truck_step_rewards[truck_id].append(base_reward)
        
        # Print step if enabled
        if show_every_step:
            visualizer.print_step(step + 1, action, info, truck_rewards)
        elif step % 10 == 0:
            # Print every 10 steps if not showing all
            print(f"Step {step + 1}/{max_steps} - Active: {sum(1 for t in info['trucks'] if not t['is_complete'] and not t['failed'])}/{num_trucks}")
        
        # Check termination
        if terminated or truncated:
            break
    
    # Print summary
    visualizer.print_summary(info, step + 1)
    
    env.close()
    
    return info


def demo_random():
    """Demo with random actions."""
    print("\n" + "🎲 " * 25)
    print("DEMO 1: Random Strategy")
    print("🎲 " * 25)
    run_visualization(num_trucks=2, num_stops=2, max_steps=30, strategy="random", seed=42)


def demo_greedy():
    """Demo with greedy strategy (always deliver)."""
    print("\n" + "🎯 " * 25)
    print("DEMO 2: Greedy Strategy (Always Deliver)")
    print("🎯 " * 25)
    run_visualization(num_trucks=2, num_stops=2, max_steps=30, strategy="greedy", seed=42)


def demo_smart():
    """Demo with smart strategy."""
    print("\n" + "🧠 " * 25)
    print("DEMO 3: Smart Strategy (Charge when needed)")
    print("🧠 " * 25)
    run_visualization(num_trucks=3, num_stops=3, max_steps=40, strategy="smart", seed=42)


def demo_custom():
    """Demo with custom parameters."""
    print("\n" + "⚙️ " * 25)
    print("DEMO 4: Custom Scenario")
    print("⚙️ " * 25)
    
    # Get user input or use defaults
    num_trucks = 2
    num_stops = 3
    max_steps = 50
    strategy = "smart"
    
    run_visualization(
        num_trucks=num_trucks,
        num_stops=num_stops,
        max_steps=max_steps,
        strategy=strategy,
        seed=123
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize SimpleTruckEnv episodes")
    parser.add_argument("--trucks", type=int, default=10, help="Number of trucks")
    parser.add_argument("--stops", type=int, default=3, help="Number of delivery stops per truck")
    parser.add_argument("--steps", type=int, default=10, help="Maximum steps")
    parser.add_argument("--strategy", choices=["random", "greedy", "smart"], default="smart", 
                        help="Action selection strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo", choices=["random", "greedy", "smart", "all"], 
                        help="Run predefined demo")
    parser.add_argument("--summary-only", action="store_true", 
                        help="Only show summary, not every step")
    
    args = parser.parse_args()
    
    if args.demo:
        if args.demo == "random":
            demo_random()
        elif args.demo == "greedy":
            demo_greedy()
        elif args.demo == "smart":
            demo_smart()
        elif args.demo == "all":
            demo_random()
            demo_greedy()
            demo_smart()
    else:
        print("\n" + "🚛 " * 25)
        print("SimpleTruckEnv Visualization")
        print("🚛 " * 25)
        
        run_visualization(
            num_trucks=args.trucks,
            num_stops=args.stops,
            max_steps=args.steps,
            strategy=args.strategy,
            seed=args.seed,
            show_every_step=not args.summary_only
        )

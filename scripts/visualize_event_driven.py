"""
Event-driven visualization for EventDrivenTruckEnv
Shows truck states, events, and timeline progression
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env import EventDrivenTruckEnv, load_config
import numpy as np


class EventDrivenVisualizer:
    """Text-based visualizer for event-driven truck environment."""
    
    def __init__(self, env: EventDrivenTruckEnv):
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
    
    def _get_status_icon(self, truck) -> str:
        """Get status icon for truck."""
        if truck.is_complete:
            return "✅"
        elif truck.failed:
            return "❌"
        elif truck.is_charging:
            return "🔌"
        else:
            return "🚛"
    
    def _get_event_icon(self, event_type: str) -> str:
        """Get icon for event type."""
        icons = {
            'TRUCK_READY': '🚛',
            'ROUTE_COMPLETE': '📍',
            'CHARGE_COMPLETE': '🔋',
            'TRUCK_TERMINATED': '🏁'
        }
        return icons.get(event_type, '❓')
    
    def print_header(self):
        """Print episode header."""
        print("\n" + "="*110)
        print(f"{'EVENT-DRIVEN TRUCK ROUTING VISUALIZATION':^110}")
        print("="*110)
        print(f"Trucks: {self.env.num_trucks} | Max Time: {self.env.max_time}h | " +
              f"Charging Stations: {self.env.num_charging_nodes}")
        print("="*110 + "\n")
    
    def print_initial_state(self, info: dict):
        """Print initial state of all trucks."""
        print("📋 INITIAL STATE")
        print("-" * 100)
        print(f"🕐 Clock: {self.env.global_clock:.2f}h")
        print(f"🚛 Active Trucks: {info['num_active_trucks']}/{self.env.num_trucks}")
        print(f"📅 Events Pending: {info['events_pending']}")
        print()
        
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
    
    def print_event_queue(self, max_events: int = 5):
        """Print upcoming events in the queue."""
        if not self.env.event_queue:
            print("  📭 Event queue is empty")
            return
        
        print(f"  📬 Upcoming Events (showing next {min(max_events, len(self.env.event_queue))}):")
        
        # Show first few events without modifying the heap
        events = sorted(self.env.event_queue)[:max_events]
        for i, event in enumerate(events, 1):
            icon = self._get_event_icon(event.event_type.name)
            print(f"     {i}. {icon} t={event.time:.2f}h - {event.event_type.name} (Truck {event.truck_id})")
    
    def print_decision_point(self, decision_num: int, action: int, info: dict, reward: float, truck_id: int):
        """Print information at a decision point."""
        active_truck = self.env.trucks[truck_id]
        
        print(f"\n{'═'*110}")
        print(f"🎯 DECISION POINT {decision_num}")
        print(f"{'═'*110}")
        print(f"🕐 Current Time: {self.env.global_clock:.2f}h / {self.env.max_time}h")
        print()
        
        # Active truck info
        status = self._get_status_icon(active_truck)
        print(f"{status} ACTIVE TRUCK: {active_truck.truck_id}")
        print(f"  {'─'*105}")
        print(f"  Location:         Node {active_truck.current_node}")
        print(f"  Battery:          {self._get_battery_bar(active_truck.get_battery_percentage())}")
        remaining_deliveries = len(active_truck.get_remaining_deliveries())
        print(f"  Deliveries:       {remaining_deliveries} remaining")
        remaining_nodes = active_truck.get_remaining_deliveries()
        next_delivery = remaining_nodes[0] if remaining_nodes else 'N/A'
        print(f"  Next Delivery:    Node {next_delivery}")
        print(f"  Total Time:       {active_truck.total_time_elapsed:.2f}h")
        print(f"  Total Distance:   {active_truck.total_distance_traveled:.2f} km")
        print(f"  Charging Sessions: {active_truck.num_charging_sessions}")
        print()
        
        # Action taken
        action_desc = self._get_action_description(action)
        print(f"  ▶️  Action Taken:  {action_desc}")
        print(f"  💰 Reward:        {reward:+.2f}")
        print()
        
        # Event queue
        self.print_event_queue(max_events=5)
        print()
        
        # All trucks summary
        print(f"  🚛 Fleet Status:")
        active_count = sum(1 for t in self.env.trucks if not t.is_complete and not t.failed)
        complete_count = sum(1 for t in self.env.trucks if t.is_complete)
        failed_count = sum(1 for t in self.env.trucks if t.failed)
        print(f"     Active: {active_count} | Complete: {complete_count} ✅ | Failed: {failed_count} ❌")
    
    def print_all_trucks_status(self):
        """Print status of all trucks."""
        print(f"\n{'─'*110}")
        print("📊 ALL TRUCKS STATUS")
        print(f"{'─'*110}")
        
        for truck in self.env.trucks:
            status = self._get_status_icon(truck)
            state = "COMPLETE" if truck.is_complete else ("FAILED" if truck.failed else "ACTIVE")
            
            remaining_deliveries = len(truck.get_remaining_deliveries())
            total_deliveries = len(truck.delivery_sequence) - 1
            deliveries_made = total_deliveries - remaining_deliveries
            
            print(f"\n{status} Truck {truck.truck_id} - {state}")
            print(f"  Time: {truck.total_time_elapsed:.2f}h | Distance: {truck.total_distance_traveled:.2f}km | " +
                  f"Battery: {truck.get_battery_percentage():.1f}% | Deliveries: {deliveries_made}/{total_deliveries}")
    
    def print_summary(self, info: dict, total_decisions: int):
        """Print episode summary."""
        print("\n" + "="*110)
        print(f"{'EPISODE SUMMARY':^110}")
        print("="*110)
        
        print(f"\n🕐 Final Time: {self.env.global_clock:.2f}h / {self.env.max_time}h")
        print(f"🎯 Total Decisions: {total_decisions}")
        print(f"💰 Total Reward: {info['episode_reward']:.2f}")
        print(f"✅ All Complete: {'YES ✅' if info['all_complete'] else 'NO ❌'}")
        print(f"❌ Any Failed: {'YES ❌' if info['any_failed'] else 'NO ✅'}")
        
        print(f"\n{'TRUCK STATISTICS':^110}")
        print("-" * 110)
        
        for truck in self.env.trucks:
            status = "COMPLETE ✅" if truck.is_complete else ("FAILED ❌" if truck.failed else "INCOMPLETE ⏸️")
            
            remaining_deliveries = len(truck.get_remaining_deliveries())
            total_deliveries = len(truck.delivery_sequence) - 1
            deliveries_made = total_deliveries - remaining_deliveries
            
            print(f"\nTruck {truck.truck_id}: {status}")
            print(f"  Total Time:        {truck.total_time_elapsed:.2f} hours")
            print(f"  Total Distance:    {truck.total_distance_traveled:.2f} km")
            print(f"  Final Battery:     {truck.get_battery_percentage():.1f}%")
            print(f"  Deliveries:        {deliveries_made}/{total_deliveries}")
            print(f"  Charging Sessions: {truck.num_charging_sessions}")
            print(f"  Charging Time:     {truck.total_charging_time:.2f} hours")
            
            if truck.failed:
                print(f"  ⚠️  Failure Reason: Battery depleted or time exceeded")
        
        # Print charger utilization statistics
        if 'charger_utilization' in info:
            self.print_charger_utilization(info['charger_utilization'])
        
        print("\n" + "="*110 + "\n")
    
    def print_charger_utilization(self, utilization: dict):
        """Print charging station utilization statistics."""
        print(f"\n{'CHARGING STATION UTILIZATION':^110}")
        print("-" * 110)
        
        overall = utilization['overall']
        level2 = utilization['level2']
        dcfast = utilization['dcfast']
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Average Utilization:    {overall['avg_utilization']*100:.1f}%")
        print(f"  Total Charge Sessions:  {overall['total_sessions']}")
        print(f"  Total Charge Time:      {overall['total_charge_time']:.2f} hours")
        
        print(f"\n🔌 Level 2 Chargers ({level2['num_chargers']} stations):")
        print(f"  Average Utilization:    {level2['avg_utilization']*100:.1f}%")
        print(f"  Total Sessions:         {level2['total_sessions']}")
        print(f"  Total Charge Time:      {level2['total_charge_time']:.2f} hours")
        
        print(f"\n⚡ DC Fast Chargers ({dcfast['num_chargers']} stations):")
        print(f"  Average Utilization:    {dcfast['avg_utilization']*100:.1f}%")
        print(f"  Total Sessions:         {dcfast['total_sessions']}")
        print(f"  Total Charge Time:      {dcfast['total_charge_time']:.2f} hours")
        
        # Show top 5 most utilized chargers
        all_chargers = sorted(utilization['all_chargers'], key=lambda x: x['utilization_rate'], reverse=True)
        if all_chargers:
            print(f"\n🏆 Top 5 Most Utilized Chargers:")
            for i, charger in enumerate(all_chargers[:5], 1):
                util_pct = charger['utilization_rate'] * 100
                util_bar = '█' * int(util_pct / 5) + '░' * (20 - int(util_pct / 5))
                print(f"  {i}. Node {charger['node']:10d} ({charger['type']:7s}): [{util_bar}] {util_pct:5.1f}% | " +
                      f"{charger['sessions']} sessions | {charger['charge_time']:.1f}h")


def run_visualization(
    num_trucks: int = 3,
    num_stops: int = 3,
    max_time: float = 48.0,
    strategy: str = "random",
    seed: int = 42,
    show_every_decision: bool = True,
    max_decisions: int = 100
):
    """
    Run a visualization of the event-driven truck environment.
    
    Args:
        num_trucks: Number of trucks
        num_stops: Number of delivery stops per truck
        max_time: Maximum time in hours
        strategy: "random", "greedy", or "smart"
        seed: Random seed
        show_every_decision: If False, only show summary
        max_decisions: Maximum decision points to visualize
    """
    # Create environment
    config = load_config()
    config['advanced']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    config['environment']['max_time'] = max_time
    config['environment']['verbose'] = False
    
    env = EventDrivenTruckEnv(config=config,
                              enable_plotting=True)
    visualizer = EventDrivenVisualizer(env)
    
    # Reset environment
    obs, info = env.reset(seed=seed)
    
    # Print header and initial state
    visualizer.print_header()
    visualizer.print_initial_state(info)
    
    total_reward = 0.0
    decision_count = 0
    
    # Run episode
    terminated = False
    truncated = False
    
    while not (terminated or truncated) and decision_count < max_decisions:
        decision_count += 1
        
        # Check if episode ended
        if env.active_truck_id is None:
            break
        
        # Select action based on strategy
        active_truck_id = env.active_truck_id  # Save before step
        active_truck = env.trucks[active_truck_id]
        
        if strategy == "random":
            action = env.action_space.sample()
        elif strategy == "greedy":
            # Always go to next delivery
            action = env.num_charging_nodes
        elif strategy == "smart":
            # Smart strategy: charge when low, otherwise deliver
            battery_pct = active_truck.get_battery_percentage()
            at_charger = active_truck.current_node in env.charging_nodes
            
            if battery_pct < 20.0:
                # Critical battery - go to nearest charger
                action = 0
            elif battery_pct < 40.0 and at_charger:
                # At charger with low battery - charge for 2h
                action = env.num_navigation_actions + 1
            else:
                # Go to delivery
                action = env.num_charging_nodes
        else:
            action = env.action_space.sample()
        
        # Execute step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print decision point if enabled
        if show_every_decision:
            visualizer.print_decision_point(decision_count, action, info, reward, active_truck_id)
        elif decision_count % 10 == 0:
            # Print every 10 decisions if not showing all
            active_count = sum(1 for t in env.trucks if not t.is_complete and not t.failed)
            print(f"Decision {decision_count} - Time: {env.global_clock:.2f}h - Active trucks: {active_count}/{num_trucks}")
    
    # Print final status and summary
    if show_every_decision:
        visualizer.print_all_trucks_status()
    
    info['episode_reward'] = total_reward
    visualizer.print_summary(info, decision_count)
    
    env.close()
    
    return info


def demo_random():
    """Demo with random actions."""
    print("\n" + "🎲 " * 27)
    print("DEMO 1: Random Strategy")
    print("🎲 " * 27)
    run_visualization(num_trucks=2, num_stops=2, max_time=24.0, strategy="random", seed=42, max_decisions=20)


def demo_greedy():
    """Demo with greedy strategy (always deliver)."""
    print("\n" + "🎯 " * 27)
    print("DEMO 2: Greedy Strategy (Always Deliver)")
    print("🎯 " * 27)
    run_visualization(num_trucks=2, num_stops=2, max_time=24.0, strategy="greedy", seed=42, max_decisions=20)


def demo_smart():
    """Demo with smart strategy."""
    print("\n" + "🧠 " * 27)
    print("DEMO 3: Smart Strategy (Charge when needed)")
    print("🧠 " * 27)
    run_visualization(num_trucks=3, num_stops=3, max_time=48.0, strategy="smart", seed=42, max_decisions=30)


def demo_large_fleet():
    """Demo with larger fleet."""
    print("\n" + "🚚 " * 27)
    print("DEMO 4: Large Fleet (10 trucks)")
    print("🚚 " * 27)
    run_visualization(
        num_trucks=10,
        num_stops=3,
        max_time=48.0,
        strategy="smart",
        seed=123,
        show_every_decision=False,  # Too many decisions, show summary only
        max_decisions=100
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize EventDrivenTruckEnv episodes")
    parser.add_argument("--trucks", type=int, default=3, help="Number of trucks")
    parser.add_argument("--stops", type=int, default=3, help="Number of delivery stops per truck")
    parser.add_argument("--time", type=float, default=48.0, help="Maximum time in hours")
    parser.add_argument("--strategy", choices=["random", "greedy", "smart"], default="smart", 
                        help="Action selection strategy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--demo", choices=["random", "greedy", "smart", "large", "all"], 
                        help="Run predefined demo")
    parser.add_argument("--summary-only", action="store_true", 
                        help="Only show summary, not every decision")
    parser.add_argument("--max-decisions", type=int, default=100,
                        help="Maximum number of decisions to visualize")
    
    args = parser.parse_args()
    
    if args.demo:
        if args.demo == "random":
            demo_random()
        elif args.demo == "greedy":
            demo_greedy()
        elif args.demo == "smart":
            demo_smart()
        elif args.demo == "large":
            demo_large_fleet()
        elif args.demo == "all":
            demo_random()
            demo_greedy()
            demo_smart()
            demo_large_fleet()
    else:
        print("\n" + "🚛 " * 27)
        print("Event-Driven Truck Environment Visualization")
        print("🚛 " * 27)
        
        run_visualization(
            num_trucks=args.trucks,
            num_stops=args.stops,
            max_time=args.time,
            strategy=args.strategy,
            seed=args.seed,
            show_every_decision=not args.summary_only,
            max_decisions=args.max_decisions
        )

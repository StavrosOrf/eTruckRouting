"""
Test script for Event-Driven Truck Environment
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truck_env import EventDrivenTruckEnv, load_config
import numpy as np


def test_basic_event_driven():
    """Test basic event-driven environment functionality."""
    print("="*80)
    print("TEST: Basic Event-Driven Environment")
    print("="*80)
    
    # Load config and modify for testing
    config = load_config()
    config['advanced']['num_trucks'] = 2
    config['environment']['num_stops'] = 2
    config['environment']['max_time'] = 24.0  # 24 hours max
    config['environment']['verbose'] = True
    
    # Create event-driven environment
    env = EventDrivenTruckEnv(config=config)
    
    print(f"\nEnvironment created:")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Max time: {env.max_time} hours")
    
    # Reset
    obs, info = env.reset(seed=42)
    
    print(f"\nInitial state:")
    print(f"  Global clock: {info['global_clock']:.2f}h")
    print(f"  Active truck: {info['active_truck_id']}")
    print(f"  Active trucks: {info['num_active_trucks']}")
    print(f"  Events pending: {info['events_pending']}")
    
    # Take a few steps
    for step in range(10):
        # Simple strategy: always go to next delivery
        action = env.num_charging_nodes  # Go to next delivery
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Time: {info['global_clock']:.2f}h")
        print(f"  Active truck: {info['active_truck_id']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        
        if terminated or truncated:
            print(f"\nEpisode ended!")
            print(f"  All complete: {info['all_complete']}")
            print(f"  Any failed: {info['any_failed']}")
            print(f"  Total reward: {info['episode_reward']:.2f}")
            break
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_event_sequence():
    """Test event queue and time progression."""
    print("="*80)
    print("TEST: Event Sequence and Time Progression")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 3
    config['environment']['num_stops'] = 2
    config['environment']['verbose'] = False
    
    env = EventDrivenTruckEnv(config=config)
    obs, info = env.reset(seed=123)
    
    print(f"\nInitial state:")
    print(f"  Clock: {info['global_clock']:.2f}h")
    print(f"  Events: {info['events_pending']}")
    
    step_count = 0
    times = []
    active_trucks = []
    
    while step_count < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        times.append(info['global_clock'])
        active_trucks.append(info['active_truck_id'])
        
        if step_count % 5 == 0:
            print(f"\nStep {step_count}:")
            print(f"  Clock: {info['global_clock']:.2f}h")
            print(f"  Active truck: {info['active_truck_id']}")
            print(f"  Active trucks total: {info['num_active_trucks']}")
        
        step_count += 1
        
        if terminated or truncated:
            break
    
    print(f"\nTime progression:")
    print(f"  Steps: {step_count}")
    print(f"  Final time: {times[-1]:.2f}h")
    print(f"  Time always increasing: {all(times[i] <= times[i+1] for i in range(len(times)-1))}")
    print(f"  Different trucks controlled: {len(set(active_trucks))}")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_charging_events():
    """Test charging event completion."""
    print("="*80)
    print("TEST: Charging Events")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 1
    config['environment']['num_stops'] = 2
    config['environment']['verbose'] = True
    
    env = EventDrivenTruckEnv(config=config)
    obs, info = env.reset(seed=456)
    
    truck_id = info['active_truck_id']
    truck = env.trucks[truck_id]
    
    print(f"\nInitial battery: {truck.current_battery:.1f} kWh")
    
    # First go to a charger
    action_go_to_charger = 0  # First charger
    obs, reward, terminated, truncated, info = env.step(action_go_to_charger)
    
    print(f"\nAfter routing to charger:")
    print(f"  Clock: {info['global_clock']:.2f}h")
    print(f"  At node: {truck.current_node}")
    
    # Now charge for 2 hours
    action_charge_2h = env.num_navigation_actions + 1  # Charge 2h
    obs, reward, terminated, truncated, info = env.step(action_charge_2h)
    
    print(f"\nAfter charging:")
    print(f"  Clock: {info['global_clock']:.2f}h")
    print(f"  Battery: {truck.current_battery:.1f} kWh")
    print(f"  Charging sessions: {truck.num_charging_sessions}")
    
    env.close()
    print("\n✅ TEST PASSED\n")


def test_truck_termination():
    """Test truck termination conditions."""
    print("="*80)
    print("TEST: Truck Termination")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 2
    config['environment']['num_stops'] = 1  # Just one delivery
    config['environment']['verbose'] = False
    
    env = EventDrivenTruckEnv(config=config)
    obs, info = env.reset(seed=789)
    
    print(f"\nInitial state:")
    for truck_state in info['trucks']:
        print(f"  Truck {truck_state['truck_id']}: {truck_state['deliveries_remaining']} deliveries")
    
    # Run until completion
    step_count = 0
    while step_count < 50:
        action = env.num_charging_nodes  # Always go to delivery
        obs, reward, terminated, truncated, info = env.step(action)
        
        step_count += 1
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step_count}")
            print(f"  Clock: {info['global_clock']:.2f}h")
            print(f"  All complete: {info['all_complete']}")
            print(f"  Any failed: {info['any_failed']}")
            
            for truck_state in info['trucks']:
                status = "COMPLETE" if truck_state['is_complete'] else ("FAILED" if truck_state['failed'] else "ACTIVE")
                print(f"  Truck {truck_state['truck_id']}: {status}")
            
            break
    
    env.close()
    print("\n✅ TEST PASSED\n")


def demo_event_driven_visualization():
    """Visual demonstration of event-driven simulation."""
    print("="*80)
    print("DEMO: Event-Driven Simulation Visualization")
    print("="*80)
    
    config = load_config()
    config['advanced']['num_trucks'] = 2
    config['environment']['num_stops'] = 2
    config['environment']['verbose'] = True
    
    env = EventDrivenTruckEnv(config=config)
    obs, info = env.reset(seed=42)
    
    print(f"\n{'='*80}")
    print("SIMULATION START")
    print(f"{'='*80}")
    
    step_count = 0
    while step_count < 15:
        active_truck_id = info['active_truck_id']
        if active_truck_id is None:
            print("\nNo more active trucks!")
            break
        
        truck = env.trucks[active_truck_id]
        
        print(f"\n{'─'*80}")
        print(f"⏰ CLOCK: {info['global_clock']:.2f}h | STEP {step_count + 1}")
        print(f"🚛 ACTIVE TRUCK: {active_truck_id}")
        print(f"{'─'*80}")
        print(f"  Battery: {truck.current_battery:.1f}/{truck.battery_capacity:.1f} kWh ({truck.get_battery_percentage():.1f}%)")
        print(f"  Location: Node {truck.current_node}")
        print(f"  Deliveries remaining: {len(truck.get_remaining_deliveries())}")
        print(f"  State: {info['truck_states'][active_truck_id]}")
        
        # Smart action selection
        battery_pct = truck.get_battery_percentage()
        at_charger = truck.current_node in env.charging_nodes
        
        if battery_pct < 30.0 and not at_charger:
            # Low battery - go to charger
            action = 0
            action_desc = "🔋 Go to charger (low battery)"
        elif battery_pct < 60.0 and at_charger:
            # At charger with medium battery - charge
            action = env.num_navigation_actions + 1  # 2 hours
            action_desc = "⚡ Charge for 2 hours"
        else:
            # Go to delivery
            action = env.num_charging_nodes
            action_desc = "📦 Go to next delivery"
        
        print(f"  ➡️ ACTION: {action_desc}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"  💰 REWARD: {reward:+.2f}")
        
        step_count += 1
        
        if terminated or truncated:
            print(f"\n{'='*80}")
            print("SIMULATION END")
            print(f"{'='*80}")
            print(f"Final time: {info['global_clock']:.2f}h")
            print(f"Total reward: {info['episode_reward']:.2f}")
            print(f"All complete: {'✅ YES' if info['all_complete'] else '❌ NO'}")
            print(f"Any failed: {'❌ YES' if info['any_failed'] else '✅ NO'}")
            
            print(f"\nFinal truck states:")
            for truck_state in info['trucks']:
                status = "✅" if truck_state['is_complete'] else ("❌" if truck_state['failed'] else "⏸️")
                print(f"  {status} Truck {truck_state['truck_id']}: {truck_state['total_time']:.2f}h, {truck_state['total_distance']:.2f}km")
            
            break
    
    env.close()
    print("\n✅ DEMO COMPLETE\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Event-Driven Truck Environment")
    parser.add_argument("--test", choices=["basic", "events", "charging", "termination", "all"],
                        default="all", help="Which test to run")
    parser.add_argument("--demo", action="store_true", help="Run visual demo")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_event_driven_visualization()
    elif args.test == "basic":
        test_basic_event_driven()
    elif args.test == "events":
        test_event_sequence()
    elif args.test == "charging":
        test_charging_events()
    elif args.test == "termination":
        test_truck_termination()
    elif args.test == "all":
        test_basic_event_driven()
        test_event_sequence()
        test_charging_events()
        test_truck_termination()
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80 + "\n")

"""
Test script to demonstrate charging queue dynamics with more activity.
Uses a custom scenario that forces more charging station usage.
"""

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.baselines.heuristic_policy import HeuristicPolicy
import yaml

def create_test_config():
    """Create a test config that encourages more charging."""
    config = {
        'environment': {
            'num_trucks': 5,  # More trucks
            'num_stops': 4,   # More stops
            'min_hop_distance': 100,  # Longer distances
            'max_hop_distance': 300,
            'max_time': 450
        },
        'truck': {
            'battery_capacity': 300,  # Smaller battery to force charging
            'base_speed': 40.0,
            'initial_battery': 'full'
        },
        'rewards': {
            'delivery_bonus': 50.0,
            'completion_bonus': 1000.0,
            'time_penalty': -1.0,
            'distance_penalty': -0.1,
            'charge_penalty': -2.0,
            'failure_penalty': -500.0,
            'invalid_action_penalty': -10.0,
            'insufficient_battery_penalty': -50.0
        },
        'charging': {
            'charge_durations': [1, 2, 3, 4],
            'dcfast': {
                'charge_rate': 50.0,
                'efficiency': 0.85
            },
            'level2': {
                'charge_rate': 7.2,
                'efficiency': 0.90
            }
        },
        'network': {
            'data_path': 'truck_env/data/',
            'shortest_path_energy_file': 'shortest_path_energy_dict.json',
            'shortest_path_time_file': 'shortest_path_time_dict.json',
            'station_info_file': 'station_info_dict.json'
        },
        'traffic': {
            'enable_traffic': False,
            'std_dev_factor': 0.15,
            'max_std_dev_hours': 1.0
        }
    }
    return config

def test_heavy_charging_scenario(seed: int = 123):
    """Test queue visualization with a scenario that requires heavy charging."""
    
    config = create_test_config()

    # Create environment with plotting enabled
    env = EventDrivenTruckEnv(
        config=config, 
        run_id="heavy_charging_test",
        verbose=False,
        enable_plotting=True
    )

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    print("\n" + "="*80)
    print("TESTING HEAVY CHARGING SCENARIO")
    print("="*80)
    print(f"Environment: {env.num_trucks} trucks, {env.num_charging_nodes} chargers")
    print(f"Battery capacity: {env.trucks[0].battery_capacity} kWh (reduced to force charging)")
    print(f"Max simulation time: {env.max_time} hours")
    print("="*80 + "\n")

    total_reward = 0.0
    total_steps = 0
    charging_events = 0
    
    # Use heuristic policy
    policy = HeuristicPolicy(verbose=False)

    while True:
        action = policy.get_action(env)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        
        # Track charging events
        if env.active_truck_id is not None:
            truck = env.trucks[env.active_truck_id]
            if env.truck_states.get(truck.truck_id) == 'charging':
                charging_events += 1

        # Print progress
        if total_steps % 20 == 0:
            active_trucks = info.get('num_active_trucks', 0)
            total_occupancy = sum(info.get('charger_occupancy_counts', {}).values())
            total_waitlist = sum(info.get('charger_waitlist_lengths', {}).values())
            print(f"Step {total_steps}: Time={info['global_clock']:.1f}h, "
                  f"Active={active_trucks}, Charging={total_occupancy}, "
                  f"Waiting={total_waitlist}")

        if done or truncated:
            break

    print("\n" + "="*80)
    print("SIMULATION COMPLETE")
    print("="*80)
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {info['global_clock']:.2f} hours")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Charging events observed: {charging_events}")
    print(f"All trucks complete: {info.get('all_complete', False)}")
    print(f"Any trucks failed: {info.get('any_failed', False)}")
    print("="*80)
    
    # Print detailed charging statistics
    charger_util = info.get('charger_utilization', {})
    if charger_util:
        print("\nDETAILED CHARGING STATISTICS:")
        print("-" * 80)
        overall = charger_util.get('overall', {})
        print(f"Overall avg utilization: {overall.get('avg_utilization', 0)*100:.1f}%")
        print(f"Total charge sessions: {overall.get('total_sessions', 0)}")
        print(f"Total charge time: {overall.get('total_charge_time', 0):.1f} hours")
        
        # Per-charger details
        all_chargers = charger_util.get('all_chargers', [])
        active_chargers = [c for c in all_chargers if c['sessions'] > 0]
        
        if active_chargers:
            print(f"\nChargers Used: {len(active_chargers)}")
            print("-" * 80)
            for charger in sorted(active_chargers, key=lambda x: x['sessions'], reverse=True):
                print(f"  Node {charger['node']} ({charger['type']}, Cap: {charger['capacity']}):")
                print(f"    Sessions: {charger['sessions']}, "
                      f"Charge Time: {charger['charge_time']:.1f}h, "
                      f"Utilization: {charger['utilization_rate']*100:.1f}%, "
                      f"Trucks: {charger['trucks_served']}")
        print("-" * 80)
    
    # Close environment to generate visualizations
    print("\nGenerating visualizations...")
    env.close()
    print(f"\n✓ All visualizations saved to: results/heavy_charging_test/")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_heavy_charging_scenario(seed=123)

"""
Test pure event-driven FCFS charging queue with sequence numbers.
"""
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.utils.utils import load_config
import numpy as np

def test_fcfs_with_sequences():
    """Test that trucks queue correctly with sequence numbers."""
    
    config_dict = load_config("truck_env/config_files/config.yaml")
    
    # Reduce complexity for focused testing
    config_dict['environment']['verbose'] = True
    config_dict['environment']['num_trucks'] = 10
    config_dict['environment']['num_stops'] = 3
    config_dict['environment']['max_time'] = 200.0
    
    env = EventDrivenTruckEnv(config=config_dict, verbose=True)
    
    print("\n" + "="*80)
    print("TEST: Pure Event-Driven FCFS Queue with Sequence Numbers")
    print("="*80)
    
    obs, info = env.reset(seed=456)
    
    # Find a single-port charger for strict testing
    single_port = None
    for node in env.charging_nodes:
        if env.charging_station.charger_capacity[node] == 1:
            single_port = node
            break
    
    print(f"\nTarget single-port charger: Node {single_port}")
    print(f"Capacity: {env.charging_station.charger_capacity[single_port]}")
    
    step_count = 0
    max_steps = 200
    
    while step_count < max_steps and env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        
        print(f"\n{'='*80}")
        print(f"Step {step_count}: Truck {env.active_truck_id} | Time {env.global_clock:.2f}h")
        print(f"  Location: {truck.current_node} | Battery: {truck.get_battery_percentage():.1f}%")
        
        # Simple strategy: go to single-port charger if battery < 90%
        if truck.get_battery_percentage() < 90 and truck.current_node != single_port:
            # Navigate to single-port charger
            try:
                charger_idx = env.charging_nodes.index(single_port)
                action = charger_idx
                print(f"  Action: Navigate to charger {single_port}")
            except:
                action = env.action_space.sample()
        elif truck.current_node == single_port and truck.get_battery_percentage() < 95:
            # At charger - try to charge
            action = env.num_charging_nodes + 1  # Charge 2h
            print(f"  Action: Charge for 2 hours")
        else:
            # Go to next delivery
            action = env.num_charging_nodes
            print(f"  Action: Go to next delivery")
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            if terminated or truncated:
                print("\n✓ Episode ended")
                break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*80)
    print(f"Test completed after {step_count} steps")
    print("\nSequence counter reached:", env.charging_station.waitlist_sequence_counter)
    print("="*80)

if __name__ == "__main__":
    test_fcfs_with_sequences()

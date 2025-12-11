#!/usr/bin/env python3
"""Unit tests for charging duration validation logic."""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_detour import GNNStateSpaceDetourBased
from EVRoutingEnv.utils.utils import load_config


def test_charging_validation_scenario():
    """
    Test the exact scenario from the failing episode:
    - At charger 11 with 116.1 kWh battery
    - Next delivery 163 needs 161.7 kWh to reach
    - Validate which charging durations should be feasible
    """
    print("="*70)
    print("UNIT TEST: Charging Validation at Charger 11")
    print("="*70)
    
    # Load environment
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 5
    
    env = EventDrivenTruckEnv(config, verbose=False)
    env.reset(seed=27)  # The seed that caused the failure
    
    # Create GNN state space with verbose output
    gnn = GNNStateSpaceDetourBased(
        num_trucks=1,
        num_stops=5,
        max_time=env.max_time,
        num_charging_nodes=len(env.charging_nodes),
        device='cpu',
        verbose=True
    )
    
    # Simulate to the failing state: take actions until at charger 11 with ~116 kWh
    # Based on action history: step 0-5
    actions_to_take = [
        16,  # Step 0: ROUTE to charger 149
        35,  # Step 1: CHARGE for 10.0h at node 149
        25,  # Step 2: ROUTE to delivery 253
        25,  # Step 3: ROUTE to delivery 222
        25,  # Step 4: ROUTE to delivery 5
        0,   # Step 5: ROUTE to charger 11
    ]
    
    for step, action in enumerate(actions_to_take):
        if env.active_truck_id is None:
            break
        
        data = gnn.get_state_GNN(env)
        truck = env.trucks[0]
        print(f"\nStep {step}: Location={truck.current_node}, Battery={truck.current_battery:.1f} kWh, "
              f"at_charger={truck.current_node in env.charging_nodes}, must_leave={truck.must_leave_charger}")
        
        if action >= len(data.action_to_node_map):
            print(f"  Action {action} out of range, skipping")
            break
            
        action_node_id, is_charging = data.action_to_node_map[action]
        if is_charging:
            charge_dur = data.action_charge_durations[action].item()
            print(f"  Taking action {action}: CHARGE {charge_dur:.1f}h")
        else:
            node_type = "charger" if action_node_id in env.charging_nodes else "delivery"
            print(f"  Taking action {action}: ROUTE to {node_type} {action_node_id}")
        
        _, reward, done, trunc, _ = env.step(action)
        
        if done or trunc:
            break
    
    # Now we should be at charger 11 with low battery
    truck = env.trucks[0]
    print(f"\n{'='*70}")
    print(f"CURRENT STATE:")
    print(f"  Location: {truck.current_node}")
    print(f"  Battery: {truck.current_battery:.1f} kWh / {truck.battery_capacity:.1f} kWh")
    print(f"  At charger: {truck.current_node in env.charging_nodes}")
    print(f"  Must leave: {truck.must_leave_charger}")
    print(f"  Next delivery: {truck.get_next_delivery_target()}")
    print(f"  Remaining deliveries: {truck.get_remaining_deliveries()}")
    print(f"{'='*70}")
    
    # Get the GNN state to see which charging actions are feasible
    print(f"\nGetting GNN state with verbose validation...")
    data = gnn.get_state_GNN(env)
    
    # Check feasible actions
    feasible_actions = [i for i, mask in enumerate(data.feasible_action_mask) if mask]
    charging_actions = [i for i in feasible_actions 
                       if data.action_to_node_map[i][1]]  # is_charging = True
    
    print(f"\n{'='*70}")
    print(f"FEASIBLE CHARGING ACTIONS:")
    print(f"{'='*70}")
    
    if charging_actions:
        for action_idx in charging_actions:
            charge_dur = data.action_charge_durations[action_idx].item()
            print(f"  Action {action_idx}: CHARGE {charge_dur:.1f}h")
    else:
        print(f"  NO CHARGING ACTIONS FEASIBLE")
    
    # Manual validation check
    print(f"\n{'='*70}")
    print(f"MANUAL VALIDATION:")
    print(f"{'='*70}")
    
    current_battery = truck.current_battery
    battery_capacity = truck.battery_capacity
    next_delivery = truck.get_next_delivery_target()
    
    if next_delivery is not None:
        energy_to_delivery = env.transport_graph.get_path_energy(truck.current_node, next_delivery)
        energy_safety_factor = 1.2  # Default safety factor
        max_energy_to_delivery = energy_to_delivery * energy_safety_factor
        
        print(f"  Energy to next delivery {next_delivery}: {energy_to_delivery:.1f} kWh")
        print(f"  With safety factor: {max_energy_to_delivery:.1f} kWh")
        print(f"  Current battery: {current_battery:.1f} kWh")
        print(f"  Can reach now: {current_battery >= max_energy_to_delivery}")
        
        # Check each charging duration
        charger_type = env.charging_station.charger_type.get(truck.current_node, "DCFast")
        charging_config = env.config["charging"]
        
        if charger_type == "DCFast":
            charger_config_type = charging_config["dcfast"]
        else:
            charger_config_type = charging_config["level2"]
        
        print(f"\n  Charger type: {charger_type}")
        print(f"  Testing each charging duration:")
        
        charge_durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for charge_duration in charge_durations:
            initial_soc = max(0.0, min(1.0, current_battery / battery_capacity))
            charger_config_with_curve = charger_config_type.copy()
            charger_config_with_curve["use_realistic_curve"] = charging_config.get("use_realistic_curve", False)
            
            charge_amount, _ = env.charging_curve_model.calculate_charge(
                initial_soc=initial_soc,
                charge_hours=charge_duration,
                battery_capacity=battery_capacity,
                charger_config=charger_config_with_curve,
                charger_type=charger_type
            )
            
            battery_after_charging = min(battery_capacity, current_battery + charge_amount)
            can_reach_delivery = battery_after_charging >= max_energy_to_delivery
            
            # Check if can reach a charger from delivery
            battery_at_delivery = battery_after_charging - max_energy_to_delivery
            can_reach_any_charger = False
            
            if can_reach_delivery:
                for charger_id in env.charging_nodes:
                    energy_del_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
                    if not np.isinf(energy_del_to_charger):
                        max_energy_to_charger = energy_del_to_charger * energy_safety_factor
                        if battery_at_delivery >= max_energy_to_charger:
                            can_reach_any_charger = True
                            break
            
            print(f"    {charge_duration:2d}h: +{charge_amount:5.1f} kWh → {battery_after_charging:6.1f} kWh | "
                  f"Reach delivery: {can_reach_delivery}, Reach charger from delivery: {can_reach_any_charger}")


if __name__ == "__main__":
    test_charging_validation_scenario()

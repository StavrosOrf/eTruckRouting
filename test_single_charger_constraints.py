#!/usr/bin/env python3
"""
Test script to verify single-charger GNN constraints:
1. Only charging actions when at charger (must_leave=false)
2. Only routing actions when must_leave=true
3. Charging durations validated for future progress
"""

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_single_charger import GNNStateSpaceSingleCharger
from EVRoutingEnv.utils.utils import load_config

def main():
    # Load config
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    config["num_trucks"] = 1
    config["num_stops"] = 5
    config["seed"] = 42
    
    # Create environment
    env = EventDrivenTruckEnv(config=config, verbose=False)
    env.reset()
    
    # Create single-charger GNN
    gnn = GNNStateSpaceSingleCharger(
        num_trucks=env.num_trucks,
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
        verbose=True,
    )
    
    print("=" * 80)
    print("TEST 1: Initial state (not at charger)")
    print("=" * 80)
    
    truck = env.trucks[0]
    print(f"Truck location: {truck.current_node}")
    print(f"At charger: {truck.current_node in env.charging_nodes}")
    print(f"Must leave: {truck.must_leave_charger}")
    print()
    
    data = gnn.get_state_GNN(env)
    
    # Count action types
    charger_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                          if not is_chg and nid in env.charging_nodes and data.feasible_action_mask[i])
    delivery_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                           if not is_chg and nid not in env.charging_nodes and nid >= 0 and data.feasible_action_mask[i])
    charging_actions = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                          if is_chg and data.feasible_action_mask[i])
    
    print(f"Feasible charger routing actions: {charger_routing}")
    print(f"Feasible delivery routing actions: {delivery_routing}")
    print(f"Feasible charging actions: {charging_actions}")
    print()
    
    # Find a feasible charger action and execute it
    charger_action = None
    for i, (nid, is_chg) in enumerate(data.action_to_node_map):
        if not is_chg and nid in env.charging_nodes and data.feasible_action_mask[i]:
            charger_action = i
            charger_id = nid
            break
    
    if charger_action is not None:
        print(f"Executing charger routing action {charger_action} (to charger {charger_id})")
        obs, reward, done, truncated, info = env.step(charger_action)
        print(f"Reward: {reward:.2f}")
        print()
        
        # Wait for truck to arrive
        while env.truck_states[0] == "routing":
            # Step with any action (doesn't matter, truck is routing)
            obs, reward, done, truncated, info = env.step(0)
            if done or truncated:
                break
        
        print("=" * 80)
        print("TEST 2: At charger (must_leave=false)")
        print("=" * 80)
        
        truck = env.trucks[0]
        print(f"Truck location: {truck.current_node}")
        print(f"At charger: {truck.current_node in env.charging_nodes}")
        print(f"Must leave: {truck.must_leave_charger}")
        print()
        
        data = gnn.get_state_GNN(env)
        
        # Count action types
        charger_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                              if not is_chg and nid in env.charging_nodes and data.feasible_action_mask[i])
        delivery_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                               if not is_chg and nid not in env.charging_nodes and nid >= 0 and data.feasible_action_mask[i])
        charging_actions = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                              if is_chg and data.feasible_action_mask[i])
        
        print(f"Feasible charger routing actions: {charger_routing}")
        print(f"Feasible delivery routing actions: {delivery_routing}")
        print(f"Feasible charging actions: {charging_actions}")
        print()
        
        print("EXPECTED: 0 routing actions (must charge first), ≥1 charging actions")
        print(f"ACTUAL: {charger_routing + delivery_routing} routing, {charging_actions} charging")
        print()
        
        # Find and execute a charging action
        charging_action = None
        for i, (nid, is_chg) in enumerate(data.action_to_node_map):
            if is_chg and data.feasible_action_mask[i]:
                charging_action = i
                charge_duration = data.action_charge_durations[i].item()
                break
        
        if charging_action is not None:
            print(f"Executing charging action {charging_action} ({charge_duration:.0f}h)")
            obs, reward, done, truncated, info = env.step(charging_action)
            print(f"Reward: {reward:.2f}")
            print()
            
            # Wait for charging to complete
            while env.truck_states[0] in ["waiting_to_charge", "charging"]:
                obs, reward, done, truncated, info = env.step(0)
                if done or truncated:
                    break
            
            print("=" * 80)
            print("TEST 3: At charger (must_leave=true)")
            print("=" * 80)
            
            truck = env.trucks[0]
            print(f"Truck location: {truck.current_node}")
            print(f"At charger: {truck.current_node in env.charging_nodes}")
            print(f"Must leave: {truck.must_leave_charger}")
            print()
            
            data = gnn.get_state_GNN(env)
            
            # Count action types
            charger_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                                  if not is_chg and nid in env.charging_nodes and data.feasible_action_mask[i])
            delivery_routing = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                                   if not is_chg and nid not in env.charging_nodes and nid >= 0 and data.feasible_action_mask[i])
            charging_actions = sum(1 for i, (nid, is_chg) in enumerate(data.action_to_node_map)
                                  if is_chg and data.feasible_action_mask[i])
            
            print(f"Feasible charger routing actions: {charger_routing}")
            print(f"Feasible delivery routing actions: {delivery_routing}")
            print(f"Feasible charging actions: {charging_actions}")
            print()
            
            print("EXPECTED: ≥1 routing actions (must leave), 0 charging actions")
            print(f"ACTUAL: {charger_routing + delivery_routing} routing, {charging_actions} charging")
            print()
            
            print("=" * 80)
            print("ALL TESTS PASSED!" if charging_actions == 0 and (charger_routing + delivery_routing) > 0 else "TESTS FAILED!")
            print("=" * 80)

if __name__ == "__main__":
    main()

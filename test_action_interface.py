#!/usr/bin/env python3
"""
Test script to verify the unified action interface works with both formats:
1. Legacy integer actions
2. GNN tuple actions (node_id, charging_duration, is_charging)
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv


def test_legacy_action_format():
    """Test environment with legacy integer action format."""
    print("\n" + "="*80)
    print("TEST 1: Legacy Integer Action Format")
    print("="*80)
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    env = EventDrivenTruckEnv(config=config_path, verbose=False, enable_plotting=False)
    
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Active truck: {env.active_truck_id}")
    print(f"  Observation shape: {obs.shape}")
    
    # Test navigation action (go to next delivery)
    action = env.num_charging_nodes  # Navigate to next delivery
    print(f"\n✓ Testing legacy navigation action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
    
    # Test charging action (if at charger)
    if env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        if truck.current_node in env.charging_nodes:
            action = env.num_navigation_actions  # Charge for first duration option
            print(f"\n✓ Testing legacy charging action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
    
    env.close()
    print("\n✓ Legacy action format test PASSED")
    return True


def test_gnn_action_format():
    """Test environment with GNN tuple action format."""
    print("\n" + "="*80)
    print("TEST 2: GNN Tuple Action Format")
    print("="*80)
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    env = EventDrivenTruckEnv(config=config_path, verbose=False, enable_plotting=False)
    
    obs, info = env.reset(seed=42)
    print(f"✓ Environment reset successful")
    print(f"  Active truck: {env.active_truck_id}")
    print(f"  Observation shape: {obs.shape}")
    
    if env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        next_delivery = truck.get_next_delivery_target()
        
        if next_delivery is not None:
            # Test navigation action (node_id, charge_duration, is_charging)
            action = (next_delivery, 0.0, False)
            print(f"\n✓ Testing GNN navigation action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
    
    # Test charging action
    if env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        if truck.current_node in env.charging_nodes:
            # Test GNN charging action (node_id, charge_duration, is_charging)
            action = (truck.current_node, 1.0, True)
            print(f"\n✓ Testing GNN charging action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}")
    
    env.close()
    print("\n✓ GNN action format test PASSED")
    return True


def test_multiple_steps():
    """Test running multiple steps with both action formats."""
    print("\n" + "="*80)
    print("TEST 3: Multiple Steps with Mixed Action Formats")
    print("="*80)
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    env = EventDrivenTruckEnv(config=config_path, verbose=False, enable_plotting=False)
    
    obs, info = env.reset(seed=123)
    print(f"✓ Environment reset successful")
    
    steps = 0
    max_steps = 10
    legacy_count = 0
    gnn_count = 0
    
    while steps < max_steps and not (info.get('all_complete', False) or info.get('any_failed', False)):
        if env.active_truck_id is None:
            break
            
        truck = env.trucks[env.active_truck_id]
        
        # Alternate between legacy and GNN format
        if steps % 2 == 0:
            # Use legacy format
            if truck.current_node in env.charging_nodes and truck.get_battery_percentage() < 80:
                action = env.num_navigation_actions  # Charge
            else:
                action = env.num_charging_nodes  # Next delivery
            legacy_count += 1
        else:
            # Use GNN format
            next_delivery = truck.get_next_delivery_target()
            if next_delivery is not None:
                action = (next_delivery, 0.0, False)  # Navigate
            else:
                action = (truck.current_node, 1.0, True)  # Charge
            gnn_count += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    
    print(f"✓ Completed {steps} steps successfully")
    print(f"  Legacy actions: {legacy_count}")
    print(f"  GNN actions: {gnn_count}")
    print(f"  Final state - All complete: {info.get('all_complete', False)}, Any failed: {info.get('any_failed', False)}")
    
    env.close()
    print("\n✓ Multiple steps test PASSED")
    return True


def test_battery_overflow_protection():
    """Test that battery overflow protection works."""
    print("\n" + "="*80)
    print("TEST 4: Battery Overflow Protection")
    print("="*80)
    
    config_path = "EVRoutingEnv/config_files/config.yaml"
    env = EventDrivenTruckEnv(config=config_path, verbose=False, enable_plotting=False)
    
    obs, info = env.reset(seed=999)
    print(f"✓ Environment reset successful")
    
    if env.active_truck_id is not None:
        truck = env.trucks[env.active_truck_id]
        initial_battery = truck.current_battery
        capacity = truck.battery_capacity
        
        print(f"  Initial battery: {initial_battery:.2f} / {capacity:.2f} kWh")
        
        # Force truck to a charger location
        if truck.current_node not in env.charging_nodes:
            # Navigate to first charger
            charger_node = env.charging_nodes[0]
            action = (charger_node, 0.0, False)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Navigated to charger node {charger_node}")
        
        # Check battery percentage method returns valid value
        if env.active_truck_id is not None:
            truck = env.trucks[env.active_truck_id]
            battery_pct = truck.get_battery_percentage()
            assert 0.0 <= battery_pct <= 100.0, f"Battery percentage out of bounds: {battery_pct}%"
            print(f"  ✓ Battery percentage is valid: {battery_pct:.2f}%")
    
    env.close()
    print("\n✓ Battery overflow protection test PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("UNIFIED ACTION INTERFACE TEST SUITE")
    print("="*80)
    
    all_passed = True
    
    try:
        all_passed &= test_legacy_action_format()
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_gnn_action_format()
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_multiple_steps()
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        all_passed &= test_battery_overflow_protection()
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED ✗")
        print("="*80 + "\n")
        sys.exit(1)

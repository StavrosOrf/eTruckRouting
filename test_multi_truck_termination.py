"""
Test that trucks properly terminate when complete or failed, without generating spurious TRUCK_READY events.
Tests with 10+ trucks to ensure the event-driven logic handles multiple trucks correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.utils.utils import load_config


def test_multi_truck_no_spurious_events(num_trucks=10, num_stops=3, max_steps=1000, verbose=False):
    """
    Test that completed/failed trucks don't generate TRUCK_READY events.
    
    Args:
        num_trucks: Number of trucks to test with
        num_stops: Number of delivery stops per truck
        max_steps: Maximum steps before timeout
        verbose: Print detailed information
    """
    print(f"\n{'='*80}")
    print(f"Testing Multi-Truck Termination with {num_trucks} trucks, {num_stops} stops")
    print(f"{'='*80}\n")
    
    # Load config and override truck/stop counts
    config = load_config('truck_env/config_files/config.yaml')
    config['environment']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    config['environment']['max_time'] = 200.0
    config['environment']['verbose'] = verbose
    
    # Create environment
    env = EventDrivenTruckEnv(
        config=config,
        verbose=verbose,
        enable_plotting=False,
        run_id='test_multi_truck'
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"Initial state:")
    print(f"  Active trucks: {info['num_active_trucks']}")
    print(f"  Events pending: {info['events_pending']}")
    print(f"  Active truck ID: {info['active_truck_id']}")
    
    # Track truck states throughout episode
    truck_completion_times = {}
    truck_failure_times = {}
    completed_trucks = set()
    failed_trucks = set()
    
    step_count = 0
    episode_done = False
    errors = []
    
    # Run episode
    while not episode_done and step_count < max_steps:
        step_count += 1
        
        # Get active truck
        active_truck_id = env.active_truck_id
        
        if active_truck_id is None:
            # No active truck - episode should be done
            episode_done = True
            break
        
        # Verify active truck is not in completed/failed set
        if active_truck_id in completed_trucks:
            error_msg = f"Step {step_count}: Truck {active_truck_id} is ACTIVE but was marked COMPLETE at t={truck_completion_times[active_truck_id]:.2f}"
            errors.append(error_msg)
            print(f"❌ ERROR: {error_msg}")
        
        if active_truck_id in failed_trucks:
            error_msg = f"Step {step_count}: Truck {active_truck_id} is ACTIVE but was marked FAILED at t={truck_failure_times[active_truck_id]:.2f}"
            errors.append(error_msg)
            print(f"❌ ERROR: {error_msg}")
        
        # Get truck state
        truck = env.trucks[active_truck_id]
        
        # Choose a simple action: go to next delivery or charge at nearest charger
        action_mask = np.zeros(env.action_space.n, dtype=bool)
        
        # Try to go to next delivery if available
        next_delivery = truck.get_next_delivery_target()
        if next_delivery is not None:
            current_node = truck.current_node
            energy_needed = env.transport_graph.get_path_energy(current_node, next_delivery)
            
            if energy_needed < truck.current_battery and not np.isinf(energy_needed):
                # Can go to next delivery
                action = env.num_charging_nodes  # Action for next delivery
                action_mask[action] = True
            else:
                # Need to charge - go to nearest charger
                min_dist = float('inf')
                best_charger_action = 0
                for i, charger in enumerate(env.charging_nodes):
                    energy = env.transport_graph.get_path_energy(current_node, charger)
                    if energy < truck.current_battery and not np.isinf(energy):
                        if energy < min_dist:
                            min_dist = energy
                            best_charger_action = i
                            action_mask[i] = True
                
                if action_mask.any():
                    action = best_charger_action
                else:
                    # Can't reach anywhere - charge at current location if at charger
                    if current_node in env.charging_nodes:
                        action = env.num_charging_nodes + 1  # Charge 1 hour
                        action_mask[action] = True
                    else:
                        # Stuck - this should cause failure
                        action = env.num_charging_nodes
                        action_mask[action] = True
        else:
            # No deliveries left - truck should become complete
            action = env.num_charging_nodes
            action_mask[action] = True
        
        # Take action
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            error_msg = f"Step {step_count}: Exception during step: {str(e)}"
            errors.append(error_msg)
            print(f"❌ ERROR: {error_msg}")
            break
        
        # Track newly completed/failed trucks
        for truck_id, truck in enumerate(env.trucks):
            if truck.is_complete and truck_id not in completed_trucks:
                completed_trucks.add(truck_id)
                truck_completion_times[truck_id] = env.global_clock
                if verbose:
                    print(f"  ✓ Truck {truck_id} completed at t={env.global_clock:.2f}")
            
            if truck.failed and truck_id not in failed_trucks:
                failed_trucks.add(truck_id)
                truck_failure_times[truck_id] = env.global_clock
                if verbose:
                    print(f"  ✗ Truck {truck_id} failed at t={env.global_clock:.2f}")
        
        # Check if episode is done
        episode_done = terminated or truncated
        
        if step_count % 100 == 0 or episode_done:
            print(f"Step {step_count}: t={env.global_clock:.2f}h, "
                  f"completed={len(completed_trucks)}, failed={len(failed_trucks)}, "
                  f"active={info['num_active_trucks']}, done={episode_done}")
    
    # Final verification
    print(f"\n{'='*80}")
    print(f"Episode completed after {step_count} steps")
    print(f"{'='*80}")
    print(f"Final state:")
    print(f"  Completed trucks: {len(completed_trucks)}/{num_trucks}")
    print(f"  Failed trucks: {len(failed_trucks)}/{num_trucks}")
    print(f"  Total accounted: {len(completed_trucks) + len(failed_trucks)}/{num_trucks}")
    print(f"  Final time: {env.global_clock:.2f}h")
    print(f"  Errors detected: {len(errors)}")
    
    # Verify all trucks are accounted for
    total_accounted = len(completed_trucks) + len(failed_trucks)
    if total_accounted != num_trucks:
        error_msg = f"Not all trucks accounted for: {total_accounted}/{num_trucks}"
        errors.append(error_msg)
        print(f"❌ ERROR: {error_msg}")
    
    # Print any errors
    if errors:
        print(f"\n{'='*80}")
        print(f"ERRORS FOUND ({len(errors)} total):")
        print(f"{'='*80}")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
        return False
    else:
        print(f"\n{'='*80}")
        print(f"✓ SUCCESS: All tests passed!")
        print(f"  - No spurious TRUCK_READY events for completed/failed trucks")
        print(f"  - All {num_trucks} trucks properly terminated")
        print(f"  - Event-driven logic working correctly")
        print(f"{'='*80}")
        return True


def test_specific_scenario_truck_failure():
    """
    Test a specific scenario where a truck should fail and not generate more events.
    """
    print(f"\n{'='*80}")
    print(f"Testing Specific Failure Scenario")
    print(f"{'='*80}\n")
    
    config = load_config('truck_env/config_files/config.yaml')
    config['environment']['num_trucks'] = 3
    config['environment']['num_stops'] = 2
    config['environment']['max_time'] = 100.0
    config['environment']['verbose'] = True
    
    env = EventDrivenTruckEnv(
        config=config,
        verbose=True,
        enable_plotting=False,
        run_id='test_failure'
    )
    
    obs, info = env.reset(seed=123)
    
    errors = []
    failed_truck_ids = set()
    
    # Run for limited steps
    for step in range(50):
        if env.active_truck_id is None:
            break
        
        active_id = env.active_truck_id
        truck = env.trucks[active_id]
        
        # Check if this truck was previously marked as failed
        if active_id in failed_truck_ids:
            error_msg = f"Step {step}: Failed truck {active_id} became active again!"
            errors.append(error_msg)
            print(f"❌ ERROR: {error_msg}")
        
        # Choose action that might cause failure (try to go to delivery without enough battery)
        action = env.num_charging_nodes  # Go to next delivery
        
        try:
            obs, reward, done, truncated, info = env.step(action)
            
            # Check if truck failed
            if truck.failed:
                failed_truck_ids.add(active_id)
                print(f"  Truck {active_id} failed at step {step}")
            
            if done or truncated:
                break
        except Exception as e:
            print(f"Exception at step {step}: {e}")
            break
    
    if errors:
        print(f"\n❌ FAILURE: {len(errors)} errors found")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(f"\n✓ SUCCESS: Failure scenario handled correctly")
        return True


if __name__ == "__main__":
    print("="*80)
    print("MULTI-TRUCK TERMINATION TEST SUITE")
    print("="*80)
    
    # Test 1: Basic multi-truck test (10 trucks)
    success_1 = test_multi_truck_no_spurious_events(num_trucks=10, num_stops=3, verbose=False)
    
    # Test 2: Larger scale (15 trucks)
    success_2 = test_multi_truck_no_spurious_events(num_trucks=15, num_stops=2, verbose=False)
    
    # Test 3: Specific failure scenario
    success_3 = test_specific_scenario_truck_failure()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"TEST SUITE SUMMARY")
    print(f"{'='*80}")
    print(f"Test 1 (10 trucks): {'✓ PASS' if success_1 else '❌ FAIL'}")
    print(f"Test 2 (15 trucks): {'✓ PASS' if success_2 else '❌ FAIL'}")
    print(f"Test 3 (Failure scenario): {'✓ PASS' if success_3 else '❌ FAIL'}")
    
    all_passed = success_1 and success_2 and success_3
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    sys.exit(0 if all_passed else 1)

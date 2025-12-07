"""
Test charging duration calculation correctness.
Verify that charge_amount and charge_duration are consistent.
"""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv

def test_charging_duration_consistency():
    """Test that charging duration matches actual charge delivered."""
    
    # Create environment with config file
    env = EventDrivenTruckEnv(
        config="EVRoutingEnv/config_files/config.yaml",
        verbose=False
    )
    
    # Reset and get initial state
    obs, info = env.reset()
    
    print("Testing Charging Duration Consistency")
    print("=" * 80)
    
    test_cases = []
    
    for episode in range(20):
        obs, info = env.reset(seed=42 + episode)
        
        for step in range(200):
            # Get feasible actions
            action_mask = env.mask_fn()
            feasible_actions = np.where(action_mask)[0]
            
            if len(feasible_actions) == 0:
                break
            
            # Take random feasible action
            action = int(np.random.choice(feasible_actions))
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check for charging completion events in the queue
            for event in env.event_queue:
                if event.data.get("reason") == "charge_complete":
                    charge_amount = event.data["charge_amount"]
                    charge_duration = event.data["charge_duration"]
                    truck_id = event.truck_id
                    
                    # Get charger config
                    charger_node = event.data.get("charger_node")
                    if charger_node is None:
                        continue
                        
                    charger_type = env.charging_station.charger_type[charger_node]
                    charging_config = env.config["charging"]
                    
                    if charger_type == "DCFast":
                        charger_config = charging_config["dcfast"]
                    else:
                        charger_config = charging_config["level2"]
                    
                    peak_power = charger_config["charge_rate"]
                    efficiency = charger_config["efficiency"]
                    use_realistic = charging_config["use_realistic_curve"]
                    
                    # For CCCV charging, we can verify that duration and energy are consistent
                    # by re-running the charging curve calculation
                    if use_realistic and charger_type == "DCFast":
                        # Get initial SOC from event data
                        initial_soc = event.data.get("initial_soc", 0.0)
                        battery_capacity = env.trucks[truck_id].battery_capacity
                        
                        # Recalculate what the charge should be for this duration
                        recalc_charge, recalc_details = env.charging_curve_model.calculate_charge(
                            initial_soc=initial_soc,
                            charge_hours=charge_duration,
                            battery_capacity=battery_capacity,
                            charger_config=charger_config | {"use_realistic_curve": True},
                            charger_type=charger_type
                        )
                        
                        # Check consistency
                        energy_error = abs(recalc_charge - charge_amount)
                        time_error = abs(recalc_details["actual_charge_hours"] - charge_duration)
                        
                        test_case = {
                            "episode": episode,
                            "step": step,
                            "truck_id": truck_id,
                            "charger_type": charger_type + "_CCCV",
                            "charge_amount": charge_amount,
                            "charge_duration": charge_duration,
                            "peak_power": peak_power,
                            "efficiency": efficiency,
                            "recalc_charge": recalc_charge,
                            "recalc_duration": recalc_details["actual_charge_hours"],
                            "time_error": time_error,
                            "energy_error": energy_error,
                            "consistent": time_error < 0.001 and energy_error < 0.1
                        }
                        
                        test_cases.append(test_case)
                        
                        if not test_case["consistent"]:
                            print(f"\n❌ INCONSISTENCY FOUND (CCCV):")
                            print(f"  Episode {episode}, Step {step}, Truck {truck_id}")
                            print(f"  Charger: {charger_type} CCCV ({peak_power} kW, {efficiency*100}% eff)")
                            print(f"  Initial SOC: {initial_soc*100:.1f}%")
                            print(f"  Stored charge amount: {charge_amount:.4f} kWh")
                            print(f"  Stored duration: {charge_duration:.4f} hours")
                            print(f"  Recalculated charge: {recalc_charge:.4f} kWh")
                            print(f"  Recalculated duration: {recalc_details['actual_charge_hours']:.4f} hours")
                            print(f"  Energy error: {energy_error:.4f} kWh")
                            print(f"  Time error: {time_error:.6f} hours")
                    else:
                        # Linear charging: time = energy / (power * efficiency)
                        expected_time = charge_amount / (peak_power * efficiency)
                        
                        # Check consistency: time should match energy/power
                        time_error = abs(charge_duration - expected_time)
                        energy_from_time = charge_duration * peak_power * efficiency
                        energy_error = abs(energy_from_time - charge_amount)
                        
                        test_case = {
                            "episode": episode,
                            "step": step,
                            "truck_id": truck_id,
                            "charger_type": charger_type,
                            "charge_amount": charge_amount,
                            "charge_duration": charge_duration,
                            "peak_power": peak_power,
                            "efficiency": efficiency,
                            "expected_time": expected_time,
                            "time_error": time_error,
                            "energy_from_time": energy_from_time,
                            "energy_error": energy_error,
                            "consistent": time_error < 0.001 and energy_error < 0.1
                        }
                        
                        test_cases.append(test_case)
                        
                        if not test_case["consistent"]:
                            print(f"\n❌ INCONSISTENCY FOUND:")
                            print(f"  Episode {episode}, Step {step}, Truck {truck_id}")
                            print(f"  Charger: {charger_type} ({peak_power} kW, {efficiency*100}% eff)")
                            print(f"  Charge amount: {charge_amount:.4f} kWh")
                            print(f"  Charge duration: {charge_duration:.4f} hours")
                            print(f"  Expected duration: {expected_time:.4f} hours")
                            print(f"  Time error: {time_error:.6f} hours ({time_error*60:.3f} minutes)")
                            print(f"  Energy from time: {energy_from_time:.4f} kWh")
                            print(f"  Energy error: {energy_error:.4f} kWh")
            
            if terminated or truncated:
                break
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS: Tested {len(test_cases)} charging sessions")
    
    if test_cases:
        consistent_count = sum(1 for tc in test_cases if tc["consistent"])
        inconsistent_count = len(test_cases) - consistent_count
        
        print(f"  ✅ Consistent: {consistent_count}/{len(test_cases)} ({consistent_count/len(test_cases)*100:.1f}%)")
        print(f"  ❌ Inconsistent: {inconsistent_count}/{len(test_cases)} ({inconsistent_count/len(test_cases)*100:.1f}%)")
        
        if inconsistent_count > 0:
            avg_time_error = np.mean([tc["time_error"] for tc in test_cases if not tc["consistent"]])
            avg_energy_error = np.mean([tc["energy_error"] for tc in test_cases if not tc["consistent"]])
            print(f"\n  Average time error: {avg_time_error:.6f} hours ({avg_time_error*60:.3f} minutes)")
            print(f"  Average energy error: {avg_energy_error:.4f} kWh")
        
        return inconsistent_count == 0
    else:
        print("  ⚠️  No charging sessions found in test")
        return False

if __name__ == "__main__":
    success = test_charging_duration_consistency()
    
    if success:
        print("\n✅ TEST PASSED: All charging durations are consistent with charge amounts")
    else:
        print("\n❌ TEST FAILED: Found inconsistencies in charging duration calculations")
        exit(1)

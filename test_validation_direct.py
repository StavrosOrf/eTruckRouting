#!/usr/bin/env python3
"""Direct unit test for charging validation logic."""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_detour import GNNStateSpaceDetourBased
from EVRoutingEnv.utils.utils import load_config


def test_validation_logic():
    """
    Direct test of validation logic with known parameters:
    - At charger 11 with 116.1 kWh battery
    - Battery capacity: 400 kWh  
    - Next delivery: 163
    - Energy to delivery 163: ~161.7 kWh (with safety factor ~194 kWh needed)
    """
    print("="*70)
    print("DIRECT VALIDATION TEST")
    print("="*70)
    
    # Load environment just to get configuration
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 5
    
    env = EventDrivenTruckEnv(config, verbose=False)
    env.reset()
    
    # Test parameters from failing scenario
    current_location = 11  # charger
    current_battery = 116.1  # kWh
    battery_capacity = 400.0  # kWh
    next_delivery = 163
    
    # Check if charger 11 exists and get energy to delivery 163
    energy_to_delivery = env.transport_graph.get_path_energy(current_location, next_delivery)
    energy_safety_factor = 1.2
    max_energy_to_delivery = energy_to_delivery * energy_safety_factor
    
    print(f"Current location: {current_location} (charger)")
    print(f"Current battery: {current_battery:.1f} kWh")
    print(f"Battery capacity: {battery_capacity:.1f} kWh")
    print(f"Next delivery: {next_delivery}")
    print(f"Energy to delivery: {energy_to_delivery:.1f} kWh")
    print(f"With safety factor (1.2x): {max_energy_to_delivery:.1f} kWh")
    print(f"Can reach now: {current_battery >= max_energy_to_delivery}")
    print()
    
    # Get charger configuration
    charger_type = env.charging_station.charger_type.get(current_location, "DCFast")
    charging_config = env.config["charging"]
    
    if charger_type == "DCFast":
        charger_config_type = charging_config["dcfast"]
    else:
        charger_config_type = charging_config["level2"]
    
    print(f"Charger type: {charger_type}")
    print(f"Charge rate: {charger_config_type['charge_rate']} kW")
    print(f"Efficiency: {charger_config_type['efficiency']}")
    print()
    
    # Test each charging duration
    print("="*70)
    print("TESTING EACH CHARGING DURATION:")
    print("="*70)
    
    charge_durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    for charge_duration in charge_durations:
        # Calculate battery after charging
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
        
        # Check if can reach next delivery
        can_reach_delivery = battery_after_charging >= max_energy_to_delivery
        battery_at_delivery = battery_after_charging - max_energy_to_delivery if can_reach_delivery else 0
        
        # Check if can reach ANY charger from delivery
        can_reach_any_charger_from_delivery = False
        closest_charger_from_delivery = None
        min_energy_to_charger = float('inf')
        
        if can_reach_delivery:
            for charger_id in env.charging_nodes:
                energy_delivery_to_charger = env.transport_graph.get_path_energy(next_delivery, charger_id)
                
                if not np.isinf(energy_delivery_to_charger):
                    max_energy_to_charger = energy_delivery_to_charger * energy_safety_factor
                    
                    if energy_delivery_to_charger < min_energy_to_charger:
                        min_energy_to_charger = energy_delivery_to_charger
                        closest_charger_from_delivery = charger_id
                    
                    if battery_at_delivery >= max_energy_to_charger:
                        can_reach_any_charger_from_delivery = True
                        break
        
        # Determine if this duration should be valid
        should_be_valid = can_reach_delivery and can_reach_any_charger_from_delivery
        
        print(f"\n{charge_duration:2d} hours:")
        print(f"  Charge amount: +{charge_amount:5.1f} kWh")
        print(f"  Battery after: {battery_after_charging:6.1f} kWh")
        print(f"  Can reach delivery: {can_reach_delivery}")
        if can_reach_delivery:
            print(f"  Battery at delivery: {battery_at_delivery:6.1f} kWh")
            print(f"  Closest charger from delivery: {closest_charger_from_delivery} (needs {min_energy_to_charger:.1f} kWh)")
            print(f"  Can reach charger from delivery: {can_reach_any_charger_from_delivery}")
        print(f"  *** SHOULD BE VALID: {should_be_valid} ***")
    
    print("\n" + "="*70)
    print("EXPECTED RESULT:")
    print("="*70)
    print("Only durations that allow reaching delivery + charger should be valid.")
    print("If 1-hour charge is marked valid, there's a bug in the validation logic!")


if __name__ == "__main__":
    test_validation_logic()

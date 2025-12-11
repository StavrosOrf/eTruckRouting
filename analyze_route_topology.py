#!/usr/bin/env python3
"""Analyze route topology for a specific seed."""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config

def analyze_seed(seed=42):
    """Analyze delivery sequence and charging options for a specific seed."""
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 5
    
    env = EventDrivenTruckEnv(config=config, verbose=False)
    env.reset(seed=seed)
    
    truck = env.trucks[0]
    deliveries = truck.get_remaining_deliveries()
    battery_capacity = truck.battery_capacity
    current_location = truck.current_node
    
    print(f"{'='*80}")
    print(f"Route Topology Analysis - Seed {seed}")
    print(f"{'='*80}")
    print(f"Battery Capacity: {battery_capacity:.1f} kWh")
    print(f"Starting Location: {current_location}")
    print(f"Delivery Sequence: {deliveries}")
    print(f"Number of Deliveries: {len(deliveries)}")
    print(f"Number of Chargers: {len(env.charging_nodes)}")
    print(f"{'='*80}\n")
    
    # Get energy safety factor
    energy_safety_factor = 1.0
    if hasattr(env, 'traffic_config') and env.traffic_config.get('enable_traffic', False):
        if env.traffic_config.get('enable_energy_uncertainty', False):
            energy_safety_factor = env.traffic_config.get('max_energy_multiplier', 1.0)
    
    print(f"Energy Safety Factor: {energy_safety_factor}\n")
    
    # Analyze each segment
    prev_location = current_location
    for i, delivery in enumerate(deliveries):
        print(f"\n{'='*80}")
        print(f"Segment {i}: Location {prev_location} → Delivery {delivery}")
        print(f"{'='*80}")
        
        # Direct energy to delivery
        energy_direct = env.transport_graph.get_path_energy(prev_location, delivery)
        energy_with_safety = energy_direct * energy_safety_factor
        
        print(f"Direct Energy: {energy_direct:.2f} kWh (with safety: {energy_with_safety:.2f} kWh)")
        print(f"Can reach directly with full battery: {energy_with_safety <= battery_capacity}")
        
        # Find closest chargers between prev_location and delivery
        charger_options = []
        for charger_id in env.charging_nodes:
            energy_to_charger = env.transport_graph.get_path_energy(prev_location, charger_id)
            energy_charger_to_delivery = env.transport_graph.get_path_energy(charger_id, delivery)
            
            if not np.isinf(energy_to_charger) and not np.isinf(energy_charger_to_delivery):
                total_detour = energy_to_charger + energy_charger_to_delivery - energy_direct
                charger_options.append({
                    'id': charger_id,
                    'type': env.charging_station.charger_type.get(charger_id, "Unknown"),
                    'energy_to': energy_to_charger,
                    'energy_to_del': energy_charger_to_delivery,
                    'total': energy_to_charger + energy_charger_to_delivery,
                    'detour': total_detour,
                })
        
        # Sort by total distance
        charger_options.sort(key=lambda x: x['total'])
        
        print(f"\nTop 3 Chargers (by total distance):")
        for j, opt in enumerate(charger_options[:3]):
            print(f"  {j+1}. Charger {opt['id']} ({opt['type']}):")
            print(f"     → Charger: {opt['energy_to']:.2f} kWh (with safety: {opt['energy_to']*energy_safety_factor:.2f})")
            print(f"     → Delivery: {opt['energy_to_del']:.2f} kWh (with safety: {opt['energy_to_del']*energy_safety_factor:.2f})")
            print(f"     Total: {opt['total']:.2f} kWh (with safety: {opt['total']*energy_safety_factor:.2f})")
            print(f"     Detour: {opt['detour']:.2f} kWh")
            print(f"     Can reach charger from prev: {opt['energy_to']*energy_safety_factor <= battery_capacity}")
            print(f"     Can reach delivery from charger: {opt['energy_to_del']*energy_safety_factor <= battery_capacity}")
        
        if len(charger_options) == 0:
            print("  WARNING: No chargers can connect this segment!")
        
        # Check reachability from delivery to next segment
        if i < len(deliveries) - 1:
            next_delivery = deliveries[i+1]
            energy_to_next = env.transport_graph.get_path_energy(delivery, next_delivery)
            energy_to_next_safe = energy_to_next * energy_safety_factor
            
            print(f"\nFrom delivery {delivery} to next delivery {next_delivery}:")
            print(f"  Direct energy: {energy_to_next:.2f} kWh (with safety: {energy_to_next_safe:.2f} kWh)")
            print(f"  Can reach directly: {energy_to_next_safe <= battery_capacity}")
            
            # Find chargers reachable from this delivery
            reachable_chargers = []
            for charger_id in env.charging_nodes:
                energy_to_charger = env.transport_graph.get_path_energy(delivery, charger_id)
                if not np.isinf(energy_to_charger):
                    energy_safe = energy_to_charger * energy_safety_factor
                    if energy_safe <= battery_capacity:
                        reachable_chargers.append((charger_id, energy_to_charger))
            
            reachable_chargers.sort(key=lambda x: x[1])
            print(f"  Reachable chargers from delivery {delivery}: {len(reachable_chargers)}")
            if len(reachable_chargers) > 0:
                print(f"  Closest 3:")
                for k, (cid, eng) in enumerate(reachable_chargers[:3]):
                    print(f"    {k+1}. Charger {cid}: {eng:.2f} kWh (with safety: {eng*energy_safety_factor:.2f})")
            else:
                print(f"  WARNING: Cannot reach any charger from delivery {delivery}!")
        
        prev_location = delivery
    
    print(f"\n\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}")
    
    # Overall route feasibility check
    total_direct_energy = 0
    prev = current_location
    for delivery in deliveries:
        energy = env.transport_graph.get_path_energy(prev, delivery)
        total_direct_energy += energy
        prev = delivery
    
    print(f"Total direct energy (no charging): {total_direct_energy:.2f} kWh")
    print(f"Battery capacity: {battery_capacity:.1f} kWh")
    print(f"Direct route feasible: {total_direct_energy * energy_safety_factor <= battery_capacity}")
    print(f"Number of charges needed (approximate): {int(np.ceil(total_direct_energy * energy_safety_factor / battery_capacity))}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    # Analyze the seed that had issues
    print("Analyzing seed 42 (first episode):\n")
    analyze_seed(seed=42)
    
    print("\n\nAnalyzing seed 43 (second episode):\n")
    analyze_seed(seed=43)
    
    print("\n\nAnalyzing seed 44 (third episode):\n")
    analyze_seed(seed=44)

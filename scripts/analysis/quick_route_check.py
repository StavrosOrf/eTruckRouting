"""
Quick route feasibility check - fast validation of route generation.

This script quickly generates a few routes and checks for obvious issues:
- Routes with infeasible legs
- Extreme hop distances
- Energy requirements exceeding battery capacity
"""

import os
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.utils.utils import load_config


def quick_check(config_file='EVRoutingEnv/config_files/config.yaml', 
                num_samples=10, 
                num_trucks=5, 
                num_stops=5):
    """Quick check of route generation."""
    
    print(f"\n{'='*60}")
    print(f"Quick Route Generation Check")
    print(f"{'='*60}\n")
    
    config = load_config(config_file)
    config['environment']['num_trucks'] = num_trucks
    config['environment']['num_stops'] = num_stops
    
    battery_capacity = config['truck']['battery_capacity']
    
    issues = []
    all_hop_distances = []
    all_hop_energies = []
    
    for i in tqdm(range(num_samples), desc="Checking routes"):
        env = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
        env.reset(seed=42 + i)
        
        for truck_idx, truck in enumerate(env.trucks):
            sequence = truck.delivery_sequence
            
            for j in range(len(sequence) - 1):
                from_node = int(sequence[j])
                to_node = int(sequence[j + 1])
                
                energy = env.transport_graph.get_path_energy(from_node, to_node)
                
                all_hop_distances.append(energy)
                all_hop_energies.append(energy)
                
                # Check for issues
                if energy > battery_capacity:
                    # Check if reachable via any charger
                    reachable = False
                    for charger in env.charging_nodes:
                        e1 = env.transport_graph.get_path_energy(from_node, int(charger))
                        e2 = env.transport_graph.get_path_energy(int(charger), to_node)
                        if e1 <= battery_capacity and e2 <= battery_capacity:
                            reachable = True
                            break
                    
                    if not reachable:
                        issues.append({
                            'sample': i,
                            'truck': truck_idx,
                            'leg': f"{from_node} -> {to_node}",
                            'energy': energy,
                            'issue': 'Infeasible even with charging'
                        })
        
        env.close()
    
    # Report results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}\n")
    
    all_hop_distances = np.array(all_hop_distances)
    all_hop_energies = np.array(all_hop_energies)
    
    print(f"Total routes checked: {num_samples * num_trucks}")
    print(f"Total hops analyzed: {len(all_hop_distances)}\n")
    
    print(f"Hop Energy Distances:")
    print(f"  Mean: {np.mean(all_hop_distances):.1f} kWh")
    print(f"  Min: {np.min(all_hop_distances):.1f} kWh")
    print(f"  Max: {np.max(all_hop_distances):.1f} kWh\n")
    
    print(f"Hop Energy:")
    print(f"  Mean: {np.mean(all_hop_energies):.1f} kWh")
    print(f"  Max: {np.max(all_hop_energies):.1f} kWh")
    print(f"  Battery Capacity: {battery_capacity:.1f} kWh")
    print(f"  Exceeds capacity: {np.sum(all_hop_energies > battery_capacity)} hops\n")
    
    if issues:
        print(f"⚠️  ISSUES FOUND: {len(issues)}")
        print(f"\nInfeasible legs (cannot complete even with 1 charging stop):")
        for issue in issues[:10]:  # Show first 10
            print(f"  Sample {issue['sample']}, Truck {issue['truck']}: "
                  f"{issue['leg']} ({issue['energy']:.1f} kWh)")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
    else:
        print(f"✓ All routes appear feasible!")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick route generation check')
    parser.add_argument('--config', type=str, default='EVRoutingEnv/config_files/config.yaml')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-trucks', type=int, default=5)
    parser.add_argument('--num-stops', type=int, default=5)
    
    args = parser.parse_args()
    
    quick_check(args.config, args.num_samples, args.num_trucks, args.num_stops)

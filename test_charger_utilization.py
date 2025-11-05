"""
Test script to demonstrate charging station utilization with waiting times.
"""
from simple_truck_env import EventDrivenTruckEnv
import numpy as np

# Create environment with more trucks and longer routes
env = EventDrivenTruckEnv(
    num_trucks=10,
    num_stops=5,
    max_time=72.0,
    verbose=True
)

print("="*80)
print("CHARGING STATION UTILIZATION TEST")
print("="*80)
print(f"Trucks: {env.num_trucks}")
print(f"Charging Stations: {len(env.charging_nodes)}")
print(f"Max Time: {env.max_time}h")
print()

# Reset environment
obs, info = env.reset(seed=123)

decisions = 0
max_decisions = 100

# Strategy: charge aggressively to create utilization
while decisions < max_decisions:
    decisions += 1
    
    if env.active_truck_id is None:
        break
    
    truck = env.trucks[env.active_truck_id]
    battery_pct = truck.get_battery_percentage()
    at_charger = truck.current_node in env.charging_nodes
    
    # Aggressive charging strategy
    if battery_pct < 70.0 and not at_charger:
        # Go to nearest charger
        action = 0
    elif battery_pct < 90.0 and at_charger:
        # Charge for 2 hours
        action = env.num_navigation_actions + 1
    else:
        # Go to next delivery
        action = env.num_charging_nodes
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

# Print final statistics
print("\n" + "="*80)
print("FINAL CHARGING STATION STATISTICS")
print("="*80)

util = info['charger_utilization']
overall = util['overall']
level2 = util['level2']
dcfast = util['dcfast']

print(f"\n📊 Overall Statistics:")
print(f"  Simulation Time:        {env.global_clock:.2f} hours")
print(f"  Average Utilization:    {overall['avg_utilization']*100:.1f}%")
print(f"  Total Charge Sessions:  {overall['total_sessions']}")
print(f"  Total Charge Time:      {overall['total_charge_time']:.2f} hours")

print(f"\n🔌 Level 2 Chargers ({level2['num_chargers']} stations):")
print(f"  Average Utilization:    {level2['avg_utilization']*100:.1f}%")
print(f"  Total Sessions:         {level2['total_sessions']}")
print(f"  Total Charge Time:      {level2['total_charge_time']:.2f} hours")

print(f"\n⚡ DC Fast Chargers ({dcfast['num_chargers']} stations):")
print(f"  Average Utilization:    {dcfast['avg_utilization']*100:.1f}%")
print(f"  Total Sessions:         {dcfast['total_sessions']}")
print(f"  Total Charge Time:      {dcfast['total_charge_time']:.2f} hours")

# Show top 10 most utilized chargers
all_chargers = sorted(util['all_chargers'], key=lambda x: x['utilization_rate'], reverse=True)
if all_chargers:
    print(f"\n🏆 Top 10 Most Utilized Chargers:")
    print(f"{'Rank':<6} {'Node':<12} {'Type':<10} {'Capacity':<10} {'Utilization':<15} {'Sessions':<10} {'Charge Time':<12}")
    print("-"*80)
    for i, charger in enumerate(all_chargers[:10], 1):
        util_pct = charger['utilization_rate'] * 100
        util_bar = '█' * int(util_pct / 5) + '░' * (20 - int(util_pct / 5))
        print(f"{i:<6} {charger['node']:<12} {charger['type']:<10} {charger['capacity']:<10} " +
              f"[{util_bar}] {util_pct:5.1f}% {charger['sessions']:<10} {charger['charge_time']:.1f}h")

# Show truck statistics
print(f"\n\n🚛 Truck Statistics:")
print(f"{'Truck':<8} {'Status':<12} {'Deliveries':<12} {'Battery':<10} {'Charges':<10} {'Total Time':<12}")
print("-"*80)
for truck in env.trucks:
    status = "COMPLETE" if truck.is_complete else ("FAILED" if truck.failed else "ACTIVE")
    deliveries = f"{len(truck.delivery_sequence) - 1 - len(truck.get_remaining_deliveries())}/{len(truck.delivery_sequence) - 1}"
    battery = f"{truck.get_battery_percentage():.1f}%"
    charges = truck.num_charging_sessions
    total_time = f"{truck.total_time_elapsed:.1f}h"
    
    print(f"{truck.truck_id:<8} {status:<12} {deliveries:<12} {battery:<10} {charges:<10} {total_time:<12}")

print("\n" + "="*80)

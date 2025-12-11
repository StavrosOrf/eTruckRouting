"""
Test strategic charging durations computation.
"""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_single_charger import GNNStateSpaceSingleCharger
from EVRoutingEnv.utils.utils import load_config

# Load config and create environment
config = load_config("EVRoutingEnv/config_files/config.yaml")
config["environment"]["num_trucks"] = 1
config["environment"]["num_stops"] = 5

env = EventDrivenTruckEnv(config=config, verbose=False)
env.reset(seed=42)

gnn = GNNStateSpaceSingleCharger(
    num_trucks=len(env.trucks),
    num_stops=env.num_stops,
    max_time=env.max_time,
    num_charging_nodes=env.num_charging_nodes,
    verbose=True,
)
env.gnn_state_space = gnn

obs, info = env.reset(seed=42)
print(f'\n{"="*60}')
print(f'Initial state')
print(f'{"="*60}')

# Take steps until we're at a strategic charger
max_steps = 20
for step in range(max_steps):
    print(f'\n{"="*60}')
    print(f'STEP {step+1}')
    print(f'{"="*60}')
    
    # Get GNN state
    data = gnn.get_state_GNN(env)
    
    truck = env.trucks[env.active_truck_id]
    print(f'Truck location: {truck.current_node}')
    print(f'At charger: {truck.current_node in env.charging_nodes}')
    print(f'Must leave: {truck.must_leave_charger}')
    print(f'Battery: {truck.current_battery:.1f} / {truck.battery_capacity:.1f} kWh')
    print(f'Remaining deliveries: {truck.get_remaining_deliveries()}')
    
    # Check action types
    feasible_actions = [i for i, mask in enumerate(data.feasible_action_mask) if mask]
    charging_actions = []
    routing_actions = []
    
    for action_idx in feasible_actions:
        node_id, is_charging = data.action_to_node_map[action_idx]
        if is_charging:
            charging_actions.append((action_idx, data.action_charge_durations[action_idx].item()))
        else:
            routing_actions.append((action_idx, node_id))
    
    print(f'\nFeasible actions: {len(feasible_actions)}')
    print(f'  Charging actions: {len(charging_actions)}')
    if charging_actions:
        durations = [d for _, d in charging_actions]
        print(f'    Durations: {sorted(set(durations))}')
    print(f'  Routing actions: {len(routing_actions)}')
    if routing_actions:
        nodes = [n for _, n in routing_actions]
        chargers = [n for n in nodes if n in env.charging_nodes]
        deliveries = [n for n in nodes if n not in env.charging_nodes]
        print(f'    To chargers: {chargers}')
        print(f'    To deliveries: {deliveries}')
    
    if not feasible_actions:
        print('\nNo feasible actions!')
        break
    
    # Take a random action
    action = np.random.choice(feasible_actions)
    node_id, is_charging = data.action_to_node_map[action]
    
    if is_charging:
        duration = data.action_charge_durations[action].item()
        print(f'\nTaking CHARGING action: {duration} hours')
    else:
        print(f'\nTaking ROUTING action to node {node_id}')
    
    _, reward, done, truncated, info = env.step(action)
    print(f'Reward: {reward:.2f}')
    
    if done or truncated:
        print(f'\nEpisode ended: done={done}, truncated={truncated}')
        break

print(f'\n{"="*60}')
print(f'Test completed after {step+1} steps')
print(f'{"="*60}')

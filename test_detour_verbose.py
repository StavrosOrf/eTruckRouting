#!/usr/bin/env python3
"""Test detour-based GNN with verbose output."""
import torch
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_detour import GNNStateSpaceDetourBased
from EVRoutingEnv.utils.utils import load_config

# Load config
config = load_config('config.yaml')
env = EventDrivenTruckEnv(config, num_trucks=1, num_stops=5, verbose=False)
gnn = GNNStateSpaceDetourBased(
    num_trucks=1,
    num_stops=5,
    max_time=env.max_time,
    num_charging_nodes=len(env.charging_nodes),
    device='cpu',
    verbose=True  # Enable verbose
)

# Reset
env.reset(seed=42)
print(f'Initial state: truck at {env.trucks[0].current_node}, battery {env.trucks[0].current_battery:.1f} kWh')
print(f'Next delivery: {env.trucks[0].get_next_delivery_target()}')

# Take first action (route to charger 14)
action = 2  # Based on previous run
obs, reward, done, trunc, info = env.step(action)
print(f'\nAfter action: truck at {env.trucks[0].current_node}, battery {env.trucks[0].current_battery:.1f} kWh')
print(f'Must leave: {env.trucks[0].must_leave_charger}')
print(f'Next delivery: {env.trucks[0].get_next_delivery_target()}')

# Now get state
print("\n" + "="*60)
print("Getting GNN state...")
print("="*60)
state = gnn.get_state_GNN(env)
print(f'\nFeasible actions: {state.feasible_action_mask.sum().item()}/{len(state.feasible_action_mask)}')
print(f'Action to node map length: {len(state.action_to_node_map)}')

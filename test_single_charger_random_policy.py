#!/usr/bin/env python3
"""Test single-charger GNN with random feasible actions."""
import numpy as np
from EVRoutingEnv.models.environment.event_driven_env import EventDrivenTruckEnv
from EVRoutingEnv.state.gnn_state_space_single_charger import GNNStateSpaceSingleCharger
from EVRoutingEnv.utils.utils import load_config

def run_episode(env, gnn,seed=None, max_steps=1000):
    """Run one episode with random feasible actions."""
    env.reset(seed=seed)
    total_reward = 0
    steps = 0
    infeasible_route = False
    
    while steps < max_steps:
        # Check if episode is done
        if all(t.is_complete or t.failed for t in env.trucks):
            break
        
        # Check if there's an active truck (ready for decision)
        if env.active_truck_id is None:
            # No active truck - this is a bug
            raise RuntimeError("No active truck but episode not terminated")
        
        # Double-check truck is actually ready (not in transient state)
        active_truck_state = env.truck_states.get(env.active_truck_id, "unknown")
        if active_truck_state != "ready":
            # Truck is in transient state - this is a bug
            truck = env.trucks[env.active_truck_id]
            raise RuntimeError(
                f"Active truck {env.active_truck_id} in state '{active_truck_state}' but marked as active. "
                f"Location: {truck.current_node}, Battery: {truck.current_battery:.1f} kWh, "
                f"Route dest: {truck.route_destination}, Clock: {env.global_clock:.2f}"
            )
            
        # Get GNN state - may raise ValueError if route is infeasible
        try:
            data = gnn.get_state_GNN(env)
        except ValueError as e:
            # Route is fundamentally infeasible (network topology + delivery sequence)
            if "stranded at charger" in str(e) or "infeasible" in str(e).lower():
                infeasible_route = True
                break
            else:
                raise
        
        feasible_actions = [i for i, mask in enumerate(data.feasible_action_mask) if mask]
        
        if not feasible_actions:
            # No feasible actions - route is infeasible (network topology + delivery sequence)
            # This can happen with random policy making suboptimal choices
            truck = env.trucks[env.active_truck_id]
            if steps < 3:
                # If this happens very early, it's likely a real bug
                print(f"\n{'='*60}")
                print(f"CRITICAL ERROR: No feasible actions at step {steps}")
                print(f"{'='*60}")
                print(f"Truck ID: {env.active_truck_id}")
                print(f"Location: {truck.current_node}")
                print(f"Battery: {truck.current_battery:.2f} kWh")
                print(f"Next delivery: {truck.get_next_delivery_target()}")
                print(f"Remaining: {len(truck.get_remaining_deliveries())} deliveries")
                print(f"At charger: {truck.current_node in env.charging_nodes}")
                print(f"Must leave: {truck.must_leave_charger}")
                print(f"{'='*60}\n")
                raise RuntimeError("No feasible actions available - likely a bug!")
            else:
                # Random policy led to infeasible state - mark as failed
                infeasible_route = True
                break
            
        action = np.random.choice(feasible_actions)
        _, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        
        if done or truncated:
            break
    
    success = all(t.is_complete for t in env.trucks)
    failed = any(t.failed for t in env.trucks) or infeasible_route
    timeout = steps >= max_steps
    no_actions = False  # We handle no actions by marking truck as failed
    return total_reward, steps, success, failed, timeout, no_actions

def main():
    config = load_config("EVRoutingEnv/config_files/config.yaml")
    config["environment"]["num_trucks"] = 1
    config["environment"]["num_stops"] = 5
    
    seed = 42
    
    env = EventDrivenTruckEnv(config=config,
                              verbose=False)
    
    # Reset once to initialize trucks
    env.reset(seed=seed)
    
    print(f"Environment initialized with {len(env.trucks)} trucks, {env.num_stops} stops")
    
    gnn = GNNStateSpaceSingleCharger(
        num_trucks=len(env.trucks),
        num_stops=env.num_stops,
        max_time=env.max_time,
        num_charging_nodes=env.num_charging_nodes,
        verbose=True,
    )
    
    n_episodes = 20
    print(f"Running {n_episodes} episodes...", flush=True)
    results = []
    for i in range(n_episodes):
        results.append(run_episode(env, gnn, seed=seed+i))
        if (i+1) % 5 == 0:
            print(f"  Completed {i+1}/{n_episodes}", flush=True)
    
    rewards = [r[0] for r in results]
    steps = [r[1] for r in results]
    successes = [r[2] for r in results]
    failures = [r[3] for r in results]
    timeouts = [r[4] for r in results]
    no_actions = [r[5] for r in results]
    
    n_success = sum(successes)
    n_failed = sum(failures)
    n_timeout = sum(timeouts)
    n_no_actions = sum(no_actions)
    
    print(f"\n{'='*60}")
    print(f"Single-Charger GNN Random Policy ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Success:          {n_success}/{n_episodes} ({n_success/n_episodes*100:.1f}%)")
    print(f"Failed (truck):   {n_failed}/{n_episodes} ({n_failed/n_episodes*100:.1f}%)")
    print(f"Timeout (steps):  {n_timeout}/{n_episodes} ({n_timeout/n_episodes*100:.1f}%)")
    print(f"No actions:       {n_no_actions}/{n_episodes} ({n_no_actions/n_episodes*100:.1f}%)")
    print(f"Other:            {n_episodes-n_success-n_failed-n_timeout-n_no_actions}/{n_episodes}")
    print(f"")
    print(f"Avg reward:       {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg steps:        {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    print(f"Best reward:      {np.max(rewards):.2f}")
    print(f"Worst reward:     {np.min(rewards):.2f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

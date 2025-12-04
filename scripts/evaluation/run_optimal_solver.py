"""
Run Gurobi optimal solver and compare with baseline policies.

This script:
1. Generates a scenario by resetting the environment with a specific seed
2. Solves the optimal routing and charging problem using Gurobi
3. Validates the solution
4. Compares with heuristic and PPO policies
5. Generates visualization plots
"""

import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.state.gnn_state_space import GNNStateSpace
from truck_env.utils.utils import load_config
from truck_env.utils.plotter import EnvironmentPlotter
from truck_env.optimization.gurobi_solver import GurobiOptimalPlanner
from scripts.training.train import compute_action_mask
from algo.policy_utils import load_policy

# ============ CONFIGURATION ============
POLICY_PATH = "saved_models/NewFeasibleSpace_FixedGraph_ppo-variable_steps=512_epochs=5_ent=0.1_seed=0_gnnhd=32_mlphd=256/"
CONFIG_FILE = "truck_env/config_files/config.yaml"
NUM_TRUCKS = 10
NUM_STOPS = 12
SEED = 1000
OUTPUT_DIR = "results/optimal_comparison"
GUROBI_TIME_LIMIT = 600  # 10 minutes
# =======================================

class InstrumentedEnv(EventDrivenTruckEnv):
    """Subclass that records state history for visualization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def _advance_to_next_decision(self):
        """Override to capture time intervals."""
        t_start = self.global_clock
        
        truck_details = {}
        for truck in self.trucks:
            state = self.truck_states.get(truck.truck_id, "unknown")
            details = {
                "state": state,
                "current_node": int(truck.current_node),
                "destination": int(truck.route_destination) if truck.route_destination is not None else None,
            }
            truck_details[truck.truck_id] = details
            
        super()._advance_to_next_decision()
        
        t_end = self.global_clock
        
        if t_end > t_start:
            for truck in self.trucks:
                if truck.truck_id in truck_details:
                    truck_details[truck.truck_id]["end_soc"] = truck.get_battery_percentage()

            self.history.append({
                "start": t_start,
                "end": t_end,
                "trucks": truck_details
            })

def run_optimal_solver(config):
    """Run Gurobi optimal solver."""
    print("\n" + "="*100)
    print(" "*35 + "GUROBI OPTIMAL SOLVER")
    print("="*100)
    
    # Create environment
    env = EventDrivenTruckEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    obs, info = env.reset(seed=SEED)
    
    # Create solver
    solver = GurobiOptimalPlanner(env, config, time_limit=GUROBI_TIME_LIMIT, verbose=True)
    
    # Build and solve model
    solver.build_model()
    success = solver.solve()
    
    if not success:
        print("\n✗ Failed to find optimal solution")
        return None, None, None
    
    # Print solution
    solver.print_solution_summary()
    
    # Validate solution
    is_valid, message = solver.validate_solution(env)
    print(f"\nValidation: {message}")
    
    if not is_valid:
        return None, None, None
    
    # Extract metrics
    solution = solver.solution
    metrics = {
        'objective': solution['objective'],
        'makespan': solution['makespan'],
        'routes': solution['routes'],
        'charging': solution['charging'],
        'timing': solution['timing'],
    }
    
    return metrics, env, solver

def run_policy_scenario(policy_type, policy_path, config, gnn_state_space):
    """Run a policy (heuristic or PPO) on the scenario."""
    print(f"\n{'='*100}")
    print(f" "*35 + f"{policy_type.upper()} POLICY")
    print("="*100)
    
    # Load policy
    policy, active_policy_type = load_policy(policy_path, policy_type, gnn_state_space, config, device="cpu")
    
    # Run instrumented environment
    env = InstrumentedEnv(config=copy.deepcopy(config), verbose=False, enable_plotting=False)
    obs, info = env.reset(seed=SEED)
    
    done = truncated = False
    episode_steps = 0
    
    while not (done or truncated) and episode_steps < 200:
        gnn_state = gnn_state_space.get_state_GNN(env)

        if active_policy_type == "heuristic":
            action = policy.get_action(env)
        elif active_policy_type == "ppo-variable":
            raw_action = policy.select_action(gnn_state, deterministic=True)
            action = policy.to_env_action(gnn_state, int(raw_action))
        else:
            mask = torch.tensor(compute_action_mask(env), dtype=torch.bool)
            raw_action = policy.select_action(gnn_state, deterministic=True, action_mask=mask)
            if isinstance(raw_action, tuple):
                action = raw_action
            else:
                action = int(raw_action) % env.action_space.n

        obs, reward, done, truncated, info = env.step(action)
        episode_steps += 1
    
    metrics = {
        'reward': env.episode_reward,
        'steps': episode_steps,
        'history': env.history,
        'completion_time': env.global_clock,
    }
    
    print(f"\nCompleted in {env.global_clock:.2f} hours")
    print(f"Reward: {env.episode_reward:.2f}")
    print(f"Steps: {episode_steps}")
    
    return metrics, env

def compare_solutions(optimal_metrics, heuristic_metrics, ppo_metrics):
    """Print comparison of solutions."""
    print("\n" + "="*100)
    print(" "*35 + "SOLUTION COMPARISON")
    print("="*100)
    
    print(f"\n{'Method':<20} {'Completion Time':>20} {'Reward':>15} {'Steps':>10}")
    print("-"*70)
    
    opt_time = optimal_metrics['makespan']
    print(f"{'Optimal (Gurobi)':<20} {opt_time:>18.2f}h {'-':>15} {'-':>10}")
    
    heur_time = heuristic_metrics['completion_time']
    heur_reward = heuristic_metrics['reward']
    heur_steps = heuristic_metrics['steps']
    gap_heur = ((heur_time - opt_time) / opt_time) * 100 if opt_time > 0 else 0
    print(f"{'Heuristic':<20} {heur_time:>18.2f}h {heur_reward:>15.2f} {heur_steps:>10}")
    print(f"{'  (vs optimal)':<20} {f'+{gap_heur:.1f}%':>20}")
    
    ppo_time = ppo_metrics['completion_time']
    ppo_reward = ppo_metrics['reward']
    ppo_steps = ppo_metrics['steps']
    gap_ppo = ((ppo_time - opt_time) / opt_time) * 100 if opt_time > 0 else 0
    print(f"{'PPO':<20} {ppo_time:>18.2f}h {ppo_reward:>15.2f} {ppo_steps:>10}")
    print(f"{'  (vs optimal)':<20} {f'+{gap_ppo:.1f}%':>20}")
    
    print("\n" + "="*100)

def visualize_optimal_route(optimal_metrics, env, max_time, output_dir):
    """Visualize the optimal route as a Gantt chart."""
    colors = {
        "routing": "#3498db",
        "charging": "#2ecc71",
        "complete": "#f1c40f",
    }
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    truck_ids = sorted(optimal_metrics['routes'].keys())
    
    for k in truck_ids:
        route = optimal_metrics['routes'][k]
        charging = optimal_metrics['charging'][k]
        timing = optimal_metrics['timing'][k]
        
        # Background track
        ax.hlines(y=k, xmin=0, xmax=max_time, colors='gray', linestyles=':', alpha=0.1, linewidth=20)
        
        # Plot route segments
        for idx in range(len(route) - 1):
            current = route[idx]
            next_node = route[idx + 1]
            
            t_start = timing[current]['arrival_time']
            
            # Add charging time if at charger
            if current in charging:
                charge_duration = charging[current]['duration']
                # Charging segment
                ax.hlines(y=k, xmin=t_start, xmax=t_start + charge_duration, 
                         colors=colors['charging'], linewidth=3)
                ax.plot(t_start + charge_duration, k, marker='o', markersize=6,
                       markerfacecolor='white', markeredgecolor=colors['charging'], 
                       markeredgewidth=1.5, zorder=10)
                ax.text(t_start + charge_duration, k + 0.1, str(current), 
                       ha='center', va='bottom', fontsize=6, fontweight='bold', 
                       color=colors['charging'])
                t_start = t_start + charge_duration
            
            # Routing segment
            if next_node in timing:
                t_end = timing[next_node]['arrival_time']
                ax.hlines(y=k, xmin=t_start, xmax=t_end, 
                         colors=colors['routing'], linewidth=3)
                
                # Mark destination
                is_charger = next_node in env.charging_nodes
                marker = 'o' if is_charger else 's'
                color = colors['charging'] if is_charger else colors['routing']
                ax.plot(t_end, k, marker=marker, markersize=6,
                       markerfacecolor='white', markeredgecolor=color,
                       markeredgewidth=1.5, zorder=10)
                ax.text(t_end, k + 0.1, str(next_node),
                       ha='center', va='bottom', fontsize=6, fontweight='bold',
                       color=color)
                
                # Battery level
                battery = timing[next_node]['battery']
                battery_pct = (battery / 400.0) * 100
                ax.text(t_end, k - 0.08, f"{battery_pct:.0f}%",
                       ha='center', va='top', fontsize=5, color='black', alpha=0.7)
        
        # Completion marker
        last_node = route[-1]
        t_complete = timing[last_node]['arrival_time']
        ax.plot(t_complete, k, marker='*', markersize=10,
               markerfacecolor='gold', markeredgecolor='black',
               markeredgewidth=0.5, zorder=15)
    
    ax.set_xlabel("Simulation Time (hours)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Truck ID", fontsize=12, fontweight='bold')
    ax.set_title(f"Optimal Schedule (Gurobi) - Seed {SEED}", fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_time)
    ax.set_yticks(truck_ids)
    ax.set_yticklabels([f"Truck {k}" for k in truck_ids])
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=colors["routing"], label='Routing'),
        mpatches.Patch(color=colors["charging"], label='Charging'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                  markeredgecolor=colors["charging"], markersize=8, 
                  markeredgewidth=2, label='Charger Node'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                  markeredgecolor=colors["routing"], markersize=8,
                  markeredgewidth=2, label='Delivery Node'),
        plt.Line2D([0], [0], marker='*', color='gold', markerfacecolor='gold',
                  markeredgecolor='black', markersize=12, label='Completed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.12, 1))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "optimal_schedule.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Optimal schedule plot saved to {save_path}")
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    config["environment"]["num_trucks"] = NUM_TRUCKS
    config["environment"]["num_stops"] = NUM_STOPS
    
    # Initialize GNN state space (for policies)
    env_init = EventDrivenTruckEnv(config=config, verbose=False, enable_plotting=False)
    gnn_state_space = GNNStateSpace(
        num_trucks=NUM_TRUCKS,
        num_stops=NUM_STOPS,
        max_time=config["environment"]["max_time"],
        num_charging_nodes=env_init.num_charging_nodes,
    )
    env_init.close()
    
    # Run optimal solver
    optimal_metrics, optimal_env, solver = run_optimal_solver(config)
    
    if optimal_metrics is None:
        print("\n✗ Optimal solver failed. Exiting.")
        return
    
    # Run heuristic policy
    heuristic_metrics, heuristic_env = run_policy_scenario(
        "heuristic", POLICY_PATH, config, gnn_state_space
    )
    
    # Run PPO policy
    ppo_metrics, ppo_env = run_policy_scenario(
        "ppo-variable", POLICY_PATH, config, gnn_state_space
    )
    
    # Compare solutions
    compare_solutions(optimal_metrics, heuristic_metrics, ppo_metrics)
    
    # Visualize optimal route
    visualize_optimal_route(
        optimal_metrics, optimal_env, 
        config["environment"]["max_time"], OUTPUT_DIR
    )
    
    print(f"\n{'='*100}")
    print(f" All results saved to: {OUTPUT_DIR}")
    print(f"{'='*100}\n")

if __name__ == "__main__":
    main()

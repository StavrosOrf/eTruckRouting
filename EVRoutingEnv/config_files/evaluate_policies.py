# #!/usr/bin/env python3
# """
# Evaluate a trained PPO policy against heuristic and random baselines.

# The script runs multiple deterministic scenarios, reports aggregated metrics,
# and creates summary visualizations so policy quality differences are obvious.
# """

# import argparse
# import copy
# import os
# from dataclasses import dataclass
# from typing import Callable, Dict, List

# import numpy as np
# import torch
# from tqdm import tqdm
# import pandas as pd

# import matplotlib

# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns

# from algo.PPO_actionGNN import PPOActionGNN
# from EVRoutingEnv.baselines.heuristic_policy import HeuristicPolicy
# from EVRoutingEnv.models.event_driven_env import EventDrivenTruckEnv
# from EVRoutingEnv.state.gnn_state_space import GNNStateSpace
# from EVRoutingEnv.utils.utils import load_config


# NODE_FEATURE_DIMS = {"truck": 13, "delivery": 3, "charger": 4}


# @dataclass
# class EpisodeResult:
#     policy: str
#     seed: int
#     reward: float
#     success: float
#     episode_length: int
#     charging_time: float
#     charging_sessions: float
#     waiting_time: float
#     total_distance: float
#     episode_time: float
#     num_trucks: int = None
#     num_stops: int = None


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Compare trained PPO policy with heuristic and random baselines."
#     )
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="EVRoutingEnv/config_files/config.yaml",
#         help="Environment config file.",
#     )
#     parser.add_argument(
#         "--model-dir",
#         type=str,
#         default="saved_models/PPO_steps=64_epochs=10_ent=0.01_seed=0",
#         help="Directory that contains the trained PPO weights.",
#     )
#     parser.add_argument(
#         "--episodes", type=int, default=10, help="Number of evaluation scenarios per configuration."
#     )
#     parser.add_argument(
#         "--generalization", action="store_true", help="Run generalization analysis across different truck/delivery counts."
#     )
#     parser.add_argument(
#         "--seed", type=int, default=0, help="Base seed for deterministic scenarios."
#     )
#     parser.add_argument(
#         "--max-steps",
#         type=int,
#         default=400,
#         help="Safety cap on steps per episode to avoid endless rollouts.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="results/policy_eval",
#         help="Directory for metrics and figures.",
#     )
#     parser.add_argument(
#         "--gnn-hidden-dim",
#         type=int,
#         default=64,
#         help="Hidden dimension used by the PPO GNN encoder.",
#     )
#     parser.add_argument(
#         "--gnn-layers",
#         type=int,
#         default=3,
#         help="Number of GNN layers used during training.",
#     )
#     parser.add_argument(
#         "--mlp-dim",
#         type=int,
#         default=256,
#         help="Hidden dimension of PPO policy/value heads.",
#     )
#     return parser.parse_args()


# def compute_action_mask(env: EventDrivenTruckEnv) -> np.ndarray:
#     """Return boolean feasibility mask for each discrete action."""
#     num_actions = env.action_space.n
#     mask = np.zeros(num_actions, dtype=bool)

#     if env.active_truck_id is None:
#         return mask

#     truck = env.trucks[env.active_truck_id]
#     if truck.failed or truck.is_complete:
#         return mask

#     current_node = int(truck.current_node)
#     battery = float(truck.current_battery)

#     # Charger routing actions
#     for idx, charger_node in enumerate(env.charging_nodes[: env.num_charging_nodes]):
#         energy = env.transport_graph.get_path_energy(current_node, int(charger_node))
#         mask[idx] = (not np.isinf(energy)) and (energy <= battery)

#     # Delivery routing action
#     delivery_idx = env.num_charging_nodes
#     next_delivery = truck.get_next_delivery_target()
#     if next_delivery is not None:
#         energy = env.transport_graph.get_path_energy(current_node, int(next_delivery))
#         mask[delivery_idx] = (not np.isinf(energy)) and (energy <= battery)

#     # Charging actions
#     can_charge_here = (current_node in env.charging_nodes) and (
#         truck.get_battery_percentage() < 95.0
#     )
#     for i in range(env.num_charge_actions):
#         mask[env.num_navigation_actions + i] = can_charge_here

#     # Always allow delivery action if nothing else looks feasible
#     if not mask.any():
#         mask[delivery_idx] = True
#     return mask


# def load_trained_policy(
#     env: EventDrivenTruckEnv,
#     args: argparse.Namespace,
# ) -> PPOActionGNN:
#     """Instantiate PPOActionGNN and load weights from disk."""
#     policy = PPOActionGNN(
#         action_dim=env.action_space.n,
#         node_feature_dims=NODE_FEATURE_DIMS,
#         hidden_dim=args.gnn_hidden_dim,
#         num_layers=args.gnn_layers,
#         mlp_dim=args.mlp_dim,
#     )

#     model_prefix_options = [
#         os.path.join(args.model_dir, "ppo_model_best"),
#         os.path.join(args.model_dir, "ppo_model_final"),
#         os.path.join(args.model_dir, "ppo_model"),
#     ]
#     prefix = next(
#         (p for p in model_prefix_options if os.path.exists(f"{p}_actor.pt")), None
#     )
#     if prefix is None:
#         raise FileNotFoundError(
#             f"Could not find a saved PPO actor in {args.model_dir}. "
#             "Expected files like 'ppo_model_best_actor.pt'."
#         )
#     policy.load(prefix)
#     return policy


# def collect_episode_metrics(info: Dict) -> Dict[str, float]:
#     """Extract aggregated per-episode metrics from env info."""
#     trucks = info.get("trucks", [])
#     total_charging = sum(t.get("total_charging_time", 0.0) for t in trucks)
#     total_sessions = sum(t.get("num_charging_sessions", 0) for t in trucks)
#     total_waiting = sum(t.get("waiting_time", 0.0) for t in trucks)
#     total_distance = sum(t.get("total_distance", 0.0) for t in trucks)
#     return {
#         "charging_time": total_charging,
#         "charging_sessions": total_sessions,
#         "waiting_time": total_waiting,
#         "total_distance": total_distance,
#         "episode_time": info.get("global_clock", 0.0),
#         "success": 1.0 if info.get("all_complete", False) else 0.0,
#     }


# def run_policy(
#     policy_name: str,
#     env: EventDrivenTruckEnv,
#     seeds: List[int],
#     max_steps: int,
#     select_action_fn: Callable[[EventDrivenTruckEnv, np.ndarray], int],
#     show_progress: bool = True,
#     num_trucks: int = None,
#     num_stops: int = None,
# ) -> List[EpisodeResult]:
#     """Roll out one policy across all scenario seeds."""
#     results: List[EpisodeResult] = []

#     iterator = tqdm(seeds, desc=f"{policy_name}", leave=False) if show_progress else seeds
#     for seed in iterator:
#         env.reset(seed=seed)
#         done = False
#         truncated = False
#         episode_reward = 0.0
#         steps = 0
#         info: Dict = {}

#         while not (done or truncated) and steps < max_steps:
#             mask = compute_action_mask(env)
#             action = select_action_fn(env, mask)
#             _, reward, done, truncated, info = env.step(action)
#             episode_reward += reward
#             steps += 1

#         metrics = collect_episode_metrics(info)
#         results.append(
#             EpisodeResult(
#                 policy=policy_name,
#                 seed=seed,
#                 reward=episode_reward,
#                 success=metrics["success"],
#                 episode_length=steps,
#                 charging_time=metrics["charging_time"],
#                 charging_sessions=metrics["charging_sessions"],
#                 waiting_time=metrics["waiting_time"],
#                 total_distance=metrics["total_distance"],
#                 episode_time=metrics["episode_time"],
#                 num_trucks=num_trucks,
#                 num_stops=num_stops,
#             )
#         )

#     return results


# def aggregate_results(results: List[EpisodeResult]) -> Dict[str, Dict[str, float]]:
#     """Compute summary statistics for each policy."""
#     summary: Dict[str, Dict[str, float]] = {}
#     policies = sorted({r.policy for r in results})

#     for policy in policies:
#         subset = [r for r in results if r.policy == policy]
#         if not subset:
#             continue

#         def mean_std(values: List[float]) -> (float, float):
#             arr = np.array(values, dtype=np.float32)
#             return float(arr.mean()), float(arr.std(ddof=0))

#         reward_mean, reward_std = mean_std([r.reward for r in subset])
#         success_rate = float(np.mean([r.success for r in subset]))
#         steps_mean = float(np.mean([r.episode_length for r in subset]))
#         distance_mean = float(np.mean([r.total_distance for r in subset]))
#         charging_mean = float(np.mean([r.charging_time for r in subset]))
#         waiting_mean = float(np.mean([r.waiting_time for r in subset]))
#         sessions_mean = float(np.mean([r.charging_sessions for r in subset]))
#         episode_time_mean = float(np.mean([r.episode_time for r in subset]))

#         summary[policy] = {
#             "mean_reward": reward_mean,
#             "std_reward": reward_std,
#             "success_rate": success_rate,
#             "mean_episode_length": steps_mean,
#             "mean_total_distance": distance_mean,
#             "mean_charging_time": charging_mean,
#             "mean_waiting_time": waiting_mean,
#             "mean_charging_sessions": sessions_mean,
#             "mean_episode_time": episode_time_mean,
#         }
#     return summary


# def print_metrics(summary: Dict[str, Dict[str, float]]):
#     """Pretty-print evaluation summary."""
#     print("\nPolicy Evaluation Summary")
#     print("-" * 80)
#     header = (
#         f"{'Policy':<12} {'Reward (mean±std)':<22} {'Success %':<12} "
#         f"{'Steps':<10} {'Distance':<12} {'Charge h':<10} {'Wait h':<10}"
#     )
#     print(header)
#     print("-" * 80)
#     for policy, metrics in summary.items():
#         reward = f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
#         success = f"{metrics['success_rate'] * 100:5.1f}"
#         steps = f"{metrics['mean_episode_length']:.1f}"
#         distance = f"{metrics['mean_total_distance']:.1f}"
#         charge = f"{metrics['mean_charging_time']:.2f}"
#         wait = f"{metrics['mean_waiting_time']:.2f}"
#         print(
#             f"{policy:<12} {reward:<22} {success:<12} {steps:<10} "
#             f"{distance:<12} {charge:<10} {wait:<10}"
#         )
#     print("-" * 80)


# def plot_summary(
#     summary: Dict[str, Dict[str, float]],
#     output_path: str,
# ):
#     """Save comparison figure with key metrics."""
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     policies = list(summary.keys())
#     if not policies:
#         print("No policies to visualize.")
#         return

#     metrics_to_plot = [
#         ("mean_reward", "Mean Reward", "Reward", "std_reward"),
#         ("success_rate", "Success Rate", "Rate (%)", None),
#         ("mean_episode_length", "Episode Length", "Steps", None),
#         ("mean_total_distance", "Distance Traveled", "km", None),
#         ("mean_charging_time", "Charging Time", "Hours", None),
#         ("mean_waiting_time", "Waiting Time", "Hours", None),
#     ]

#     fig, axes = plt.subplots(2, 3, figsize=(14, 8))
#     axes = axes.flatten()

#     from matplotlib import cm

#     colors = cm.Set2(np.linspace(0, 1, len(policies)))

#     for ax, (metric_key, title, ylabel, err_key) in zip(axes, metrics_to_plot):
#         values = np.array([summary[p][metric_key] for p in policies], dtype=float)
#         if metric_key == "success_rate":
#             values = values * 100.0
#         errors = (
#             np.array([summary[p][err_key] for p in policies], dtype=float)
#             if err_key
#             else None
#         )
#         ax.bar(policies, values, yerr=errors, capsize=6, color=colors)
#         ax.set_title(title)
#         ax.set_ylabel(ylabel)
#         ax.set_xticklabels(policies, rotation=20)
#         ax.grid(axis="y", linestyle="--", alpha=0.5)

#     plt.tight_layout()
#     plt.savefig(output_path, dpi=200)
#     plt.close(fig)
#     print(f"Saved comparison figure to {output_path}")


# def plot_generalization_heatmaps(
#     results: List[EpisodeResult],
#     output_dir: str,
# ):
#     """Create heatmaps showing generalization across truck/delivery counts."""
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Convert to DataFrame
#     df = pd.DataFrame([{
#         'policy': r.policy,
#         'num_trucks': r.num_trucks,
#         'num_stops': r.num_stops,
#         'success': r.success,
#         'reward': r.reward,
#     } for r in results])
    
#     # Focus on PPO and Heuristic for comparison (exclude Random)
#     policies = ['PPO', 'Heuristic']
#     metrics = [
#         ('success', 'Success Rate', '.0%', 'RdYlGn', 0, 1),
#         ('reward', 'Mean Reward', '.1f', 'viridis', None, None)
#     ]
    
#     for metric_key, metric_name, fmt, cmap, vmin_val, vmax_val in metrics:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
#         for ax, policy in zip(axes, policies):
#             policy_df = df[df['policy'] == policy]
            
#             # Compute mean and std
#             pivot_mean = policy_df.pivot_table(
#                 values=metric_key,
#                 index='num_trucks',
#                 columns='num_stops',
#                 aggfunc='mean'
#             )
#             pivot_std = policy_df.pivot_table(
#                 values=metric_key,
#                 index='num_trucks',
#                 columns='num_stops',
#                 aggfunc='std'
#             )
            
#             # Create annotations with mean ± std
#             annotations = []
#             for i in range(len(pivot_mean.index)):
#                 row = []
#                 for j in range(len(pivot_mean.columns)):
#                     mean_val = pivot_mean.iloc[i, j]
#                     std_val = pivot_std.iloc[i, j]
#                     if pd.isna(mean_val):
#                         row.append('')
#                     elif metric_key == 'success':
#                         row.append(f'{mean_val:.0%}\n±{std_val:.0%}')
#                     else:
#                         row.append(f'{mean_val:.1f}\n±{std_val:.1f}')
#                 annotations.append(row)
            
#             sns.heatmap(
#                 pivot_mean,
#                 annot=annotations,
#                 fmt='',
#                 cmap=cmap,
#                 ax=ax,
#                 cbar_kws={'label': metric_name},
#                 vmin=vmin_val,
#                 vmax=vmax_val,
#             )
#             ax.set_title(f"{policy} - {metric_name}", fontsize=12, fontweight='bold')
#             ax.set_xlabel('Number of Deliveries per Truck', fontsize=10)
#             ax.set_ylabel('Number of Trucks', fontsize=10)
        
#         plt.suptitle(f'{metric_name} Comparison: PPO vs Heuristic', fontsize=14, fontweight='bold', y=1.02)
#         plt.tight_layout()
#         output_path = os.path.join(output_dir, f"generalization_{metric_key}_comparison.png")
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close(fig)
#         print(f"Saved generalization comparison to {output_path}")


# def run_generalization_analysis(
#     args: argparse.Namespace,
#     config: Dict,
#     policy,
#     heuristic_policy,
#     gnn_state_space: GNNStateSpace,
# ) -> List[EpisodeResult]:
#     """Run generalization analysis across different configurations."""
#     truck_counts = [1, 5]
#     delivery_counts = [3, 5]
#     seeds_per_config = args.episodes
    
#     all_results = []
#     total_configs = len(truck_counts) * len(delivery_counts) * 3  # 3 policies
    
#     with tqdm(total=total_configs, desc="Generalization Analysis") as pbar:
#         for num_trucks in truck_counts:
#             for num_stops in delivery_counts:
#                 # Update config
#                 test_config = copy.deepcopy(config)
#                 test_config["environment"]["num_trucks"] = num_trucks
#                 test_config["environment"]["num_stops"] = num_stops
                
#                 # Update GNN state space
#                 gnn_state_space.num_trucks = num_trucks
#                 gnn_state_space.num_stops = num_stops
                
#                 # Create environments for this configuration
#                 env_trained = EventDrivenTruckEnv(
#                     config=copy.deepcopy(test_config),
#                     verbose=False,
#                     enable_plotting=False,
#                     run_id=f"eval_trained_{num_trucks}t_{num_stops}s",
#                 )
#                 env_heuristic = EventDrivenTruckEnv(
#                     config=copy.deepcopy(test_config),
#                     verbose=False,
#                     enable_plotting=False,
#                     run_id=f"eval_heuristic_{num_trucks}t_{num_stops}s",
#                 )
#                 env_random = EventDrivenTruckEnv(
#                     config=copy.deepcopy(test_config),
#                     verbose=False,
#                     enable_plotting=False,
#                     run_id=f"eval_random_{num_trucks}t_{num_stops}s",
#                 )
                
#                 seeds = [args.seed + i for i in range(seeds_per_config)]
#                 rng = np.random.default_rng(args.seed)
#                 action_dim = env_trained.action_space.n
                
#                 # Define action selectors
#                 def rl_selector(env: EventDrivenTruckEnv, mask: np.ndarray) -> int:
#                     gnn_state = gnn_state_space.get_state_GNN(env)
#                     action = policy.select_action(
#                         gnn_state,
#                         deterministic=True,
#                         action_mask=mask,
#                     )
#                     return int(action)
                
#                 def heuristic_selector(env: EventDrivenTruckEnv, _: np.ndarray) -> int:
#                     return heuristic_policy.get_action(env)
                
#                 def random_selector(_: EventDrivenTruckEnv, mask: np.ndarray) -> int:
#                     feasible = np.flatnonzero(mask)
#                     if feasible.size:
#                         return int(rng.choice(feasible))
#                     return int(rng.integers(low=0, high=action_dim))
                
#                 # Run policies
#                 pbar.set_description(f"T={num_trucks}, D={num_stops} - PPO")
#                 all_results.extend(
#                     run_policy("PPO", env_trained, seeds, args.max_steps, rl_selector, 
#                               show_progress=False, num_trucks=num_trucks, num_stops=num_stops)
#                 )
#                 pbar.update(1)
                
#                 pbar.set_description(f"T={num_trucks}, D={num_stops} - Heuristic")
#                 all_results.extend(
#                     run_policy("Heuristic", env_heuristic, seeds, args.max_steps, heuristic_selector,
#                               show_progress=False, num_trucks=num_trucks, num_stops=num_stops)
#                 )
#                 pbar.update(1)
                
#                 pbar.set_description(f"T={num_trucks}, D={num_stops} - Random")
#                 all_results.extend(
#                     run_policy("Random", env_random, seeds, args.max_steps, random_selector,
#                               show_progress=False, num_trucks=num_trucks, num_stops=num_stops)
#                 )
#                 pbar.update(1)
                
#                 # Clean up
#                 env_trained.close()
#                 env_heuristic.close()
#                 env_random.close()
    
#     return all_results


# def main():
#     args = parse_args()
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)

#     config = load_config(args.config)
#     base_env = EventDrivenTruckEnv(
#         config=copy.deepcopy(config),
#         verbose=False,
#         enable_plotting=False,
#         run_id="eval_base",
#     )
#     action_dim = base_env.action_space.n
#     num_charging_nodes = base_env.num_charging_nodes
#     base_env.close()

#     gnn_state_space = GNNStateSpace(
#         num_trucks=config["environment"]["num_trucks"],
#         num_stops=config["environment"]["num_stops"],
#         max_time=config["environment"]["max_time"],
#         num_charging_nodes=num_charging_nodes,
#     )

#     env_trained = EventDrivenTruckEnv(
#         config=copy.deepcopy(config),
#         verbose=False,
#         enable_plotting=False,
#         run_id="eval_trained",
#     )
#     env_heuristic = EventDrivenTruckEnv(
#         config=copy.deepcopy(config),
#         verbose=False,
#         enable_plotting=False,
#         run_id="eval_heuristic",
#     )
#     env_random = EventDrivenTruckEnv(
#         config=copy.deepcopy(config),
#         verbose=False,
#         enable_plotting=False,
#         run_id="eval_random",
#     )

#     policy = load_trained_policy(env_trained, args)
#     heuristic_policy = HeuristicPolicy(verbose=False)
#     rng = np.random.default_rng(args.seed)

#     def rl_selector(env: EventDrivenTruckEnv, mask: np.ndarray) -> int:
#         gnn_state = gnn_state_space.get_state_GNN(env)
#         action = policy.select_action(
#             gnn_state,
#             deterministic=True,
#             action_mask=mask,
#         )
#         return int(action)

#     def heuristic_selector(env: EventDrivenTruckEnv, _: np.ndarray) -> int:
#         return heuristic_policy.get_action(env)

#     def random_selector(_: EventDrivenTruckEnv, mask: np.ndarray) -> int:
#         feasible = np.flatnonzero(mask)
#         if feasible.size:
#             return int(rng.choice(feasible))
#         return int(rng.integers(low=0, high=action_dim))

#     if args.generalization:
#         # Run generalization analysis
#         print("\n" + "="*80)
#         print("RUNNING GENERALIZATION ANALYSIS")
#         print(f"Trucks: [1, 5], Deliveries: [3, 5], Episodes per config: {args.episodes}")
#         print(f"Total evaluations: {2*2*args.episodes*3} = {2*2*args.episodes*3} episodes")
#         print("="*80 + "\n")
        
#         all_results = run_generalization_analysis(
#             args, config, policy, heuristic_policy, gnn_state_space
#         )
        
#         # Close base environments
#         for env in (env_trained, env_heuristic, env_random):
#             env.close()
        
#         # Save detailed results
#         results_df = pd.DataFrame([{
#             'policy': r.policy,
#             'num_trucks': r.num_trucks,
#             'num_stops': r.num_stops,
#             'seed': r.seed,
#             'reward': r.reward,
#             'success': r.success,
#             'episode_length': r.episode_length,
#             'charging_time': r.charging_time,
#             'waiting_time': r.waiting_time,
#         } for r in all_results])
        
#         csv_path = os.path.join(args.output_dir, "generalization_results.csv")
#         results_df.to_csv(csv_path, index=False)
#         print(f"\nSaved detailed results to {csv_path}")
        
#         # Generate heatmaps
#         plot_generalization_heatmaps(all_results, args.output_dir)
        
#         # Print summary by configuration
#         print("\n" + "="*80)
#         print("GENERALIZATION SUMMARY (Mean ± Std)")
#         print("="*80)
#         for policy_name in ['PPO', 'Heuristic', 'Random']:
#             print(f"\n{policy_name}:")
#             print("-" * 80)
#             policy_df = results_df[results_df['policy'] == policy_name]
            
#             # Compute mean and std for each configuration
#             summary = policy_df.groupby(['num_trucks', 'num_stops']).agg({
#                 'success': ['mean', 'std'],
#                 'reward': ['mean', 'std'],
#             })
            
#             # Format the output
#             print(f"{'Trucks':<8} {'Stops':<8} {'Success Rate':<20} {'Reward':<20}")
#             print("-" * 80)
#             for (trucks, stops), row in summary.iterrows():
#                 success_mean = row[('success', 'mean')]
#                 success_std = row[('success', 'std')]
#                 reward_mean = row[('reward', 'mean')]
#                 reward_std = row[('reward', 'std')]
                
#                 success_str = f"{success_mean:.1%} ± {success_std:.1%}"
#                 reward_str = f"{reward_mean:.1f} ± {reward_std:.1f}"
                
#                 print(f"{trucks:<8} {stops:<8} {success_str:<20} {reward_str:<20}")
        
#     else:
#         # Standard evaluation
#         seeds = [args.seed + i for i in range(args.episodes)]
        
#         print(f"\nEvaluating policies over {args.episodes} episodes...")
#         print(f"Max steps per episode: {args.max_steps}")
#         print("=" * 80)

#         all_results: List[EpisodeResult] = []
#         with tqdm(total=3, desc="Evaluating policies") as pbar:
#             pbar.set_description("Evaluating PPO")
#             all_results.extend(
#                 run_policy("PPO", env_trained, seeds, args.max_steps, rl_selector)
#             )
#             pbar.update(1)
            
#             pbar.set_description("Evaluating Heuristic")
#             all_results.extend(
#                 run_policy("Heuristic", env_heuristic, seeds, args.max_steps, heuristic_selector)
#             )
#             pbar.update(1)
            
#             pbar.set_description("Evaluating Random")
#             all_results.extend(
#                 run_policy("Random", env_random, seeds, args.max_steps, random_selector)
#             )
#             pbar.update(1)

#         for env in (env_trained, env_heuristic, env_random):
#             env.close()

#         summary = aggregate_results(all_results)
#         print_metrics(summary)
#         figure_path = os.path.join(args.output_dir, "policy_comparison.png")
#         plot_summary(summary, figure_path)


# if __name__ == "__main__":
#     main()

import random
from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.baselines.heuristic_policy import HeuristicPolicy
from truck_env.utils.utils import load_config

# improt gnn state space
from truck_env.state.gnn_state_space import GNNStateSpace


def evaluate_env(seed: int = 42):
    config_file = "truck_env/config_files/config.yaml"
    # overweiten config for testing
    config_overwrite = {
        "num_trucks": 100,
        "num_stops": 3,
        "max_time": 200.0,
    }

    config = load_config(config_file)
    config['environment'].update(config_overwrite)
    
    # create an instance of the environment
    env = EventDrivenTruckEnv(
        config=config,
        run_id="test_run",
        verbose=True,
        enable_plotting=True,
    )

    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    print("Initial Observation:", obs)
    # print("Initial Info:", info)

    # get action space
    action_space = env.action_space
    print("Action Space:", action_space)

    total_reward = 0.0
    total_steps = 0

    policy = HeuristicPolicy(verbose=False)
    gnn_state_space = GNNStateSpace(
        num_trucks=config['environment']["num_trucks"],
        num_stops=config['environment']["num_stops"],
        max_time=config['environment']["max_time"],
        num_charging_nodes=len(env.charging_nodes),
        device="cpu",
    )

    while True:
        gnn_graph = gnn_state_space.get_state_GNN(env)
        
        # Get feasible actions from the GNN graph
        feasible_mask = gnn_graph.feasible_action_mask.cpu().numpy()
        action_to_node_map = gnn_graph.action_to_node_map
        
        # Find indices of feasible actions
        feasible_indices = [i for i, is_feasible in enumerate(feasible_mask) if is_feasible]
        
        if not feasible_indices:
            print("ERROR: No feasible actions available!")
            break
        
        # Select a random feasible action
        selected_idx = random.choice(feasible_indices)
        node_id, is_charging = action_to_node_map[selected_idx]
        charge_duration = gnn_graph.action_charge_durations[selected_idx].item()
        
        # Create action tuple in GNN format: (node_id, charging_duration, is_charging)
        action = (node_id, charge_duration, is_charging)
        
        print(f"\nSelected action {selected_idx}: node={node_id}, charging={is_charging}, duration={charge_duration:.1f}h")
        print(f"Feasible actions: {len(feasible_indices)}/{len(feasible_mask)}")
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

        print("\n--- Step Result ---")
        print(f"action taken: {action}")
        # print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done, " | Truncated:", truncated)

        if done or truncated:
            break

    print("\n=== Episode Summary ===")
    print(f"Total Steps: {total_steps}")
    print(f"Total Reward: {total_reward:.2f}")
    # Close the environment to generate final plots
    env.close()


if __name__ == "__main__":

    for i in range(100,101):
        print(f"\n\n===== EVALUATION RUN {i+1} =====")
        evaluate_env(seed=i)

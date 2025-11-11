from truck_env.models.event_driven_env import EventDrivenTruckEnv
from truck_env.baselines.heuristic_policy import HeuristicPolicy

def evaluate_env(seed: int = 42):
    config_file = "truck_env/config_files/config.yaml"

    # create an instance of the environment
    env = EventDrivenTruckEnv(
        config=config_file, run_id="test_run",
        verbose=True,
        enable_plotting=True
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

    while True:
        action = action_space.sample()  # Sample a random action
        # action = 20  
        # Use heuristic policy to get action
        
        # action = policy.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1

        print("\n--- Step Result ---")
        print(f"action taken: {action}")
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done, " | Truncated:", truncated)
        # input("Press Enter to continue...")

        if done or truncated:
            break

    print("\n=== Episode Summary ===")
    print(f"Total Steps: {total_steps}")
    print(f"Total Reward: {total_reward:.2f}")
    # Close the environment to generate final plots
    env.close()


if __name__ == "__main__":
    
    for i in range(1):  
        evaluate_env(seed=i)
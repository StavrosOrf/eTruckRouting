from truck_env.models.event_driven_env import EventDrivenTruckEnv

config_file = "truck_env/config_files/config.yaml"

#create an instance of the environment
env = EventDrivenTruckEnv(config=config_file,
                          verbose=True,)

obs, info = env.reset(seed=42)

print("Initial Observation:", obs)
# print("Initial Info:", info)

# get action space
action_space = env.action_space
print("Action Space:", action_space)

while True:
    action = action_space.sample()  # Sample a random action
    obs, reward, done, truncated, info = env.step(action)
    
    print("Observation:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Truncated:", truncated)
    # print("Info:", info)
    
    if done or truncated:
        break

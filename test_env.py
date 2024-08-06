import gym
from claim_processor_env import ClaimProcessorEnv

# Create environment
env = ClaimProcessorEnv()

# Test the environment
state = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action for testing
    state, reward, done, _ = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")


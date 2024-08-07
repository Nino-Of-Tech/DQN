import gym
from claim_processor_env import ClaimProcessorEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# The steps in the health insurance claims process
steps = [
    "Patient Verification",
    "Consultation",
    "Test Results",
    "Medicine Dispensation",
    "Claim Pre-Authorization",
    "Claims Submission",
    "Review",
    "Approval"
]

# Create environment
env = ClaimProcessorEnv()
nb_actions = env.action_space.n

# Simple model with the same architecture used during training
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configuring the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

# Load the trained policy weights
dqn.compile(optimizer=Adam(learning_rate=1e-3), metrics=['mae'])
dqn.load_weights('dqn_claim_processor_weights.h5f')

# Run multiple episodes
num_episodes = 5
for episode in range(num_episodes):
    state = env.reset()
    done = False
    print(f"\nEpisode {episode + 1}")
    while not done:
        action = dqn.forward(state)
        state, reward, done, _ = env.step(action)
        state_name = steps[state] if state < len(steps) else "Unknown"
        print(f"State: {state_name}, Reward: {reward}, Done: {done}")
        env.render()

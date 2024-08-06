import gym
from claim_processor_env import ClaimProcessorEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Create environment
env = ClaimProcessorEnv()
nb_actions = env.action_space.n

# Build a simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(optimizer='adam', metrics=['mae'])

# Load the trained policy
dqn.load_weights('dqn_claim_processor_weights.h5f')

# Test the trained policy
state = env.reset()
done = False
while not done:
    action = dqn.forward(state)
    state, reward, done, _ = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    env.render()

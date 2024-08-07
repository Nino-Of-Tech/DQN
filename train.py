import gym
from claim_processor_env import ClaimProcessorEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from stable_baselines3 import DQN
from keras.optimizers import Adam

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

# Create an optimizer instance and pass it to compile
optimizer = Adam(learning_rate=1e-3)
dqn.compile(optimizer=optimizer, metrics=['mae'])

# Train the agent
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

# Save the trained policy
dqn.save_weights('dqn_claim_processor_weights.h5f', overwrite=True)

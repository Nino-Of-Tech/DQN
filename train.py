import gym
from claim_processor_env import ClaimProcessorEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Custom Epsilon Decay Callback
from rl.callbacks import Callback

class CustomEpsilonDecayCallback(Callback):
    def __init__(self, initial_eps=1.0, final_eps=0.1, decay_factor=0.995):
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.decay_factor = decay_factor
        self.epsilon = initial_eps

    def on_step_end(self, episode_step, logs={}):
        self.epsilon = max(self.final_eps, self.epsilon * self.decay_factor)
        self.model.policy.eps = self.epsilon
        return True

# Create environment
env = ClaimProcessorEnv()
nb_actions = env.action_space.n

# Build a simple model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(24))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configure and compile the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy(eps=1.0)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy)

# Create an optimizer instance and pass it to compile
optimizer = Adam(learning_rate=1e-3)
dqn.compile(optimizer=optimizer, metrics=['mae'])

# Train the agent with custom epsilon decay callback
epsilon_decay_callback = CustomEpsilonDecayCallback()
dqn.fit(env, nb_steps=100000, visualize=False, verbose=2, callbacks=[epsilon_decay_callback])

# Save the trained policy
dqn.save_weights('dqn_claim_processor_weights.h5f', overwrite=True)

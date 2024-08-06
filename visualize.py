import gym
from claim_processor_env import ClaimProcessorEnv
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Initialize Pygame
pygame.init()

# Create environment
env = ClaimProcessorEnv()
nb_actions = env.action_space.n

# Build a simple model (This is required to define the architecture so the weights can be loaded)
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Configure the agent
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(optimizer='adam', metrics=['mae'])

# Load the trained policy weights
dqn.load_weights('dqn_claim_processor_weights.h5f')

# Pygame settings
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Health Insurance Claims Processor")

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Define grid parameters
grid_size = 8
cell_size = screen_width // grid_size

# Define steps names
steps = ["Patient Verification", "Consultation", "Test Results", "Medicine Dispensation", 
         "Claim Pre-Authorization", "Claims Submission", "Review", "Approval"]

# Main visualization loop
running = True
state = env.reset()
done = False
clock = pygame.time.Clock()

while running and not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    action = dqn.forward(state)
    state, reward, done, _ = env.step(action)

    screen.fill(WHITE)
    
    # Draw grid
    for x in range(0, screen_width, cell_size):
        for y in range(0, screen_height, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(screen, BLACK, rect, 1)

    # Highlight current state
    x = (state % grid_size) * cell_size
    y = (state // grid_size) * cell_size
    pygame.draw.rect(screen, BLUE, (x, y, cell_size, cell_size))

    # Display step names
    for i, step_name in enumerate(steps):
        font = pygame.font.Font(None, 36)
        text = font.render(step_name, True, BLACK)
        text_rect = text.get_rect(center=((i % grid_size) * cell_size + cell_size // 2,
                                           (i // grid_size) * cell_size + cell_size // 2))
        screen.blit(text, text_rect)

    # Display reward
    reward_text = font.render(f"Reward: {reward}", True, RED if reward < 0 else GREEN)
    screen.blit(reward_text, (10, screen_height - 40))

    pygame.display.flip()
    clock.tick(5)  # Limit to 5 frames per second

pygame.quit()

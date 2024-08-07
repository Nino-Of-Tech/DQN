# Health Insurance Claims Processor Using Deep Q-Learning :man_scientist:

## Project Description :syringe:

This project is directly in line with my mission and venture, <a href="www.sekofia.com" target="_blank">_Sekofia_</a>, as it implements a Deep Q-Learning agent to simulate a health insurance claims processor. The environment is designed as a sequence of steps that a health insurance claim must go through, from patient verification to final approval. The goal of the agent is to navigate through these steps correctly to maximize rewards.

## Project Structure :pill:

The project consists of the following files:
- `claim_processor_env.py`: Custom Gym environment for the health insurance claims process.
- `train.py`: Script to train the Deep Q-Learning agent.
- `play.py`: Script to test the trained policy and observe the agent's performance.
- `visualize.py`: Script to visualize the agent's actions in the environment using Pygame.
- `test_env.py`: Script to test the custom environment.
- `requirements.txt`: File listing the required libraries and dependencies.

## Custom Environment :clamp:

The custom environment simulates the steps involved in processing a health insurance claim. It is defined as follows:

### Actions

The actions represent the steps in the claims process performed by a patient (the agent):
1. Patient Verification
2. Consultation
3. Test Results
4. Medicine Dispensation
5. Claim Pre-Authorization
6. Claims Submission
7. Review
8. Approval

### Rewards

The agent receives a positive reward _(+1)_ for taking the correct action in the correct sequence and a negative reward _(-1)_ for incorrect actions or skipping steps.

## Training the Agent

The agent/patient is trained using the Deep Q-Learning algorithm. The training process involves:
- Building a neural network model to approximate the Q-value function.
- Using an epsilon-greedy policy to balance exploration and exploitation.
- Decaying the epsilon value over time to reduce exploration as the agent learns.

## Running the Project

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Nino-Of-Tech/DQN.git
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv myenv
    # On Windows
    myenv\Scripts\activate
    # On macOS/Linux
    source myenv/bin/activate
    ```

3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
    
#### Training the Agent

Run the `train.py` script to train the agent:
```bash
python train.py
```

#### Testing the Trained Policy

Run the `play.py` script to test the trained policy:
```bash
python play.py
```


#### Visualizing the Agent's Performance

Run the `visualize.py` script to see the agent's actions and rewards in a graphical interface:
```bash
python visualize.py
```


## Conclusion

My Claim Processor project demonstrates the use of Deep Q-Learning to train an agent to process health insurance claims correctly. By training the agent in a custom environment and using an epsilon-greedy policy for exploration, the agent learns to navigate the steps of the claims process to maximize rewards.


## Video Demonstration

[Link to Video Demonstration](https://www.loom.com/share/900fe37e516a452fb9689673a014eecf?sid=3a8f7fcd-4574-4986-b0a2-f50df311a627)

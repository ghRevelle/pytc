import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
import random
import torch.optim as optim
from commands import *

# Use this to check if GPU is available
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    raise RuntimeWarning("It is strongly advised not to train on CPU.")
print(f"Using device: {device}")

class AirTrafficControlDQN(nn.Module):
    def __init__(self, input_dim=120, n_commands=7, n_planes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Output heads
        self.command_head = nn.Linear(128, n_commands)   # logits for command
        self.plane_head = nn.Linear(128, n_planes)       # logits for target plane
        self.argument_head = nn.Linear(128, 1)           # regression for heading / runway ID

    def forward(self, x):
        x = self.fc(x)
        command_logits = self.command_head(x)            # (batch, n_commands)
        plane_logits = self.plane_head(x)                # (batch, n_planes)
        argument = torch.tanh(self.argument_head(x))     # (batch, 1)
        argument = argument * 360                        # map to [–360, +360] or clamp for specific uses
        return command_logits, plane_logits, argument
 
def compute_dqn_loss(policy_net, target_net, batch, gamma=0.99):
    losses = []

    for state, action, reward, next_state, done in batch:
        command_logits, plane_logits, arg_pred = policy_net(state)
        next_command_logits, next_plane_logits, next_arg_pred = target_net(next_state)


        # Get Q(s,a)
        q_pred = extract_q_value(command_logits, plane_logits, arg_pred, action)

        # Get Q_target = r + γ * max_a' Q(s',a')
        with torch.no_grad():
            next_q_pred = compute_max_q_value(next_command_logits, next_plane_logits)
            q_target = reward if done else reward + gamma * next_q_pred

        loss = F.mse_loss(q_pred, q_target)
        losses.append(loss)

    return torch.stack(losses).mean()

def extract_q_value(command_logits, plane_logits, arg_pred, actions):
    # actions: list/array of tuples (cmd_idx, plane_idx, arg_val)
    q_vals = []
    for i, (cmd_idx, plane_idx, arg_val) in enumerate(actions):
        cmd_logit = command_logits[i, cmd_idx]
        plane_logit = plane_logits[i, plane_idx]
        arg_q = -((arg_pred[i] - arg_val) ** 2)
        q_val = cmd_logit + plane_logit + arg_q
        q_vals.append(q_val)
    return torch.stack(q_vals)

def compute_max_q_value(command_logits, plane_logits):
    # Max over possible command/plane pairs
    all_qs = command_logits.unsqueeze(2) + plane_logits.unsqueeze(1)  # (batch, n_commands, n_planes)
    max_q = torch.max(all_qs.view(all_qs.size(0), -1), dim=1).values
    return max_q

class AirTrafficControlEnv(gym.Env):
    def __init__(self, flight_simulator):
        super().__init__()
        self.sim = flight_simulator
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(120,), dtype=np.float32)
        
        self.action_space = spaces.Dict({
            "command": spaces.Discrete(7),      # NONE to TURN
            "plane_id": spaces.Discrete(10),    # plane slots
            "argument": spaces.Box(low=0, high=360, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed=None, options=None):
        observation = self.sim.reset()  # Should return flat 120-D array
        return observation, {}

    def step(self, action):
        reward, done = self.sim.step(action)  # Action must be mapped to internal Command class
        observation = self.sim.get_state_vector()
        info = {}
        return observation, reward, done, False, info

def train_dqn(env, policy_net, target_net, episodes=1000, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, target_update=10):
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action
            if random.random() < epsilon:
                action = {
                    'command': env.action_space['command'].sample(),
                    'plane_id': env.action_space['plane_id'].sample(),
                    'argument': env.action_space['argument'].sample()[0]
                }
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                command_logits, plane_logits, argument = policy_net(state_tensor)
                command = torch.argmax(command_logits).item()
                plane_id = torch.argmax(plane_logits).item()
                arg = argument.squeeze().item()
                action = {'command': command, 'plane_id': plane_id, 'argument': arg}

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            memory.append((state, (action['command'], action['plane_id'], action['argument']),
                          reward, next_state, done))
            state = next_state

            # Train
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                batch = [(torch.tensor(s, dtype=torch.float32, device=device),
                          a, torch.tensor(r, dtype=torch.float32, device=device),
                          torch.tensor(ns, dtype=torch.float32, device=device),
                          d) for s, a, r, ns, d in batch]

                loss = compute_dqn_loss(policy_net, target_net, batch, gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Epsilon decay
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Sync target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")


print("passed initialization test")
"""
outputs = model(state)

command_dist = D.Categorical(logits=outputs["command_logits"])
plane_dist = D.Categorical(logits=outputs["plane_logits"])

command_index = command_dist.sample()
plane_index = plane_dist.sample()

argument = outputs["argument"]  # optional: only used if command is CLEARED_FOR_TAKEOFF
"""


# to put in flightsim.py
"""
def compute_reward(env_state, command_executed, sim_tick):
    reward = 0.0

    # Reward for successful landings
    for plane in env_state.planes:
        if plane.landed_this_tick:
            reward += 1.0

    # Reward for successful takeoff
        if plane.took_off_this_tick:
            reward += 1.0

    # Penalty for invalid or illegal commands
    if command_executed.is_invalid:
        reward -= 10.0

    if command_executed.caused_conflict:
        reward -= 50.0

    # Penalty for a crash
    for plane in env_state.planes:
        if plane.crashed:
            reward -= 100.0

    # Small time pressure penalty per plane still in air
    reward -= 0.01 * env_state.num_planes_in_air

    return reward
"""


"""
import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Define the Q-Network (DQN)
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output Q-values for each action

# Initialize environment
env = gym.make("CartPole-v1")

# Set hyperparameters
num_episodes = 1000  # Number of episodes to train the agent
max_timesteps = 200  # Max timesteps per episode
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate (epsilon-greedy)
epsilon_min = 0.01  # Minimum epsilon
epsilon_decay = 0.995  # Epsilon decay rate
batch_size = 32  # Minibatch size for training
learning_rate = 0.001  # Learning rate for optimizer

# Define Q-Network, target network, and optimizer
model = DQNNetwork(input_dim=4, output_dim=2).to(device)  # CartPole state space: 4, action space: 2 (left, right)
target_model = DQNNetwork(input_dim=4, output_dim=2).to(device)  # Target model for DQN (copy of Q-network)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize the experience replay buffer
replay_buffer = deque(maxlen=10000)

# Copy weights from model to target model
def update_target():
    target_model.load_state_dict(model.state_dict())

# Define a function for epsilon-greedy action selection
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Random action (exploration)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            q_values = model(state_tensor)
            return torch.argmax(q_values).item()  # Action with highest Q-value (exploitation)

# Define a function to train the model
def train_dqn():
    global epsilon

    for episode in range(num_episodes):
        state, _ = env.reset()  # Reset the environment at the start of the episode
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Move state to GPU
        done = False
        total_reward = 0
        timestep = 0

        while not done and timestep < max_timesteps:
            action = select_action(state, epsilon)  # Choose an action
            next_state, reward, done, _, _ = env.step(action)  # Take action and observe the next state

            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)  # Move next state to GPU
            reward = torch.tensor(reward, dtype=torch.float32).to(device)  # Move reward to GPU

            # Store the experience in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state  # Transition to the next state
            total_reward += reward.item()
            timestep += 1

            # Train the model from experience replay (sample random batch)
            if len(replay_buffer) >= batch_size:
                # Sample a minibatch from the replay buffer
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*minibatch)

                # Convert to tensors and move to device (GPU)
                states = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in states])
                next_states = torch.stack([torch.tensor(ns, dtype=torch.float32).to(device) for ns in next_states])
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)

                # Compute Q-values for current states (from Q-network)
                q_values = model(states)
                current_q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values using the target network
                next_q_values = target_model(next_states)
                next_q_value = torch.max(next_q_values, dim=1)[0]
                target_q_value = rewards + (gamma * next_q_value * (1 - dones))

                # Compute loss (Mean Squared Error loss)
                loss = nn.MSELoss()(current_q_value, target_q_value)

                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update the target network every few episodes
        if episode % 10 == 0:
            update_target()

        # Decay epsilon (exploration rate) over time
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward} - Epsilon: {epsilon:.4f}")

    # Save the model after training
    save_model(model)

# Save the model after training
def save_model(model, filename="dqn_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

# Load the model from a saved file
def load_model(model, filename="dqn_model.pth"):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode (disable dropout, etc.)
    print(f"Model loaded from {filename}")

# Start the training
train_dqn()
"""
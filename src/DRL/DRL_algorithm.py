import torch
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Use this to check if GPU is available
"""
print(torch.cuda.is_available())
print(torch.cuda.device_count())  # Shows how many CUDA devices (GPUs) are available
"""

# Set up the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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


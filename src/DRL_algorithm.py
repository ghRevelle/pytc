import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym  # Changed from 'gym' to 'gymnasium'
from collections import deque
import random

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())  # Shows how many CUDA devices (GPUs) are available

"""
# 1. Define the neural network for Q-value approximation
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# 3. DQN Agent Class
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-3, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)  # Explore: Random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()  # Exploit: Action with highest Q-value

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch of experiences
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get Q-values from the current and next states
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Backpropagate and update the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# 4. Main training loop
def train_dqn():
    env = gym.make('CartPole-v1', render_mode='rgb_array')  # Specify the render_mode for Gymnasium
    agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    episodes = 1000
    max_timesteps = 200

    for episode in range(episodes):
        state, _ = env.reset()  # Gymnasium: reset() returns a tuple (observation, info)
        episode_reward = 0

        for t in range(max_timesteps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Gymnasium returns both `terminated` and `truncated`, so we combine them
            done = terminated or truncated

            # Store the transition in the replay buffer
            agent.replay_buffer.push((state, action, reward, next_state, float(done)))
            
            # Update the agent
            agent.update()

            # Update the target network every 10 episodes
            if episode % 10 == 0:
                agent.update_target_network()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.update_epsilon()

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{episodes}, Reward: {episode_reward}")

    # Save the model
    torch.save(agent.q_network.state_dict(), "cartpole_dqn.pth")

    env.close()

# Start training
train_dqn()
"""
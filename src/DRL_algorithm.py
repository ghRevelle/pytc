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

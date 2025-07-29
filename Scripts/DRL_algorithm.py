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
from DRL_env import AirTrafficControlEnv

# Use this to check if GPU is available
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeWarning("It is strongly advised not to train on CPU.")

print(f"Using device: {device}")

class AirTrafficControlDQN(nn.Module):
    def __init__(self, input_dim=70, n_commands=4, n_planes=10):
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

    def forward(self, x):
        x = self.fc(x)
        command_logits = self.command_head(x)            # (batch, n_commands)
        plane_logits = self.plane_head(x)                # (batch, n_planes)
        return command_logits, plane_logits


def compute_dqn_loss(policy_net, target_net, batch, gamma=0.99):
    losses = []

    for state, action, reward, next_state, done in batch:
        command_logits, plane_logits = policy_net(state)
        next_command_logits, next_plane_logits = target_net(next_state)

        # Get Q(s,a)
        q_pred = extract_q_value(command_logits, plane_logits, [action])

        # Get Q_target = r + γ * max_a' Q(s',a')
        with torch.no_grad():
            next_q_pred = compute_max_q_value(next_command_logits, next_plane_logits)
            q_target = reward + gamma * next_q_pred * (~done)

        loss = F.mse_loss(q_pred, q_target)
        losses.append(loss)

    return torch.stack(losses).mean()


def extract_q_value(command_logits, plane_logits, actions):
    # actions: list/array of tuples (cmd_idx, plane_idx, arg_val)
    cmd_idx, plane_idx, _ = actions[0]
    # If logits are 1D (single sample), add batch dimension
    if command_logits.dim() == 1:
        command_logits = command_logits.unsqueeze(0)
    if plane_logits.dim() == 1:
        plane_logits = plane_logits.unsqueeze(0)
    log_cmd_probs = F.log_softmax(command_logits, dim=1)
    log_plane_probs = F.log_softmax(plane_logits, dim=1)
    cmd_logit = log_cmd_probs[0, cmd_idx]
    plane_logit = log_plane_probs[0, plane_idx]
    q_val = cmd_logit + plane_logit
    return q_val.unsqueeze(0)


def compute_max_q_value(command_logits, plane_logits):
    # Max over possible command/plane pairs
    # Ensure batch dimension exists
    if command_logits.dim() == 1:
        command_logits = command_logits.unsqueeze(0)
    if plane_logits.dim() == 1:
        plane_logits = plane_logits.unsqueeze(0)
    all_qs = command_logits.unsqueeze(2) + plane_logits.unsqueeze(1)  # (batch, n_commands, n_planes)
    max_q = torch.max(all_qs.view(all_qs.size(0), -1), dim=1).values
    return max_q

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
                }
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    command_logits, plane_logits = policy_net(state_tensor)
                    command = torch.argmax(command_logits, dim=1).item()
                    plane_id = torch.argmax(plane_logits, dim=1).item()
                    action = {'command': command, 'plane_id': plane_id}

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            memory.append((state, (action['command'], action['plane_id'], 0),
                          reward, next_state, torch.tensor(done)))
            state = next_state

            # Train
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                batch = [(torch.tensor(s, dtype=torch.float32, device=device),
                          a, torch.tensor(r, dtype=torch.float32, device=device),
                          torch.tensor(ns, dtype=torch.float32, device=device),
                          torch.tensor(d, dtype=torch.bool, device=device)) for s, a, r, ns, d in batch]

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


def run_episode(env, policy_net, eval=False):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if eval:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                command_logits, plane_logits = policy_net(state_tensor)
                command = torch.argmax(command_logits, dim=1).item()
                plane_id = torch.argmax(plane_logits, dim=1).item()
        else:
            command = env.action_space['command'].sample()
            plane_id = env.action_space['plane_id'].sample()

        action = {'command': command, 'plane_id': plane_id}
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

print("passed initialization test")

if __name__ == "__main__":
    env = AirTrafficControlEnv()
    input_dim = env._state_dim()  # This will be 70 if max_planes=10 and 7 features per plane
    n_commands = env.action_space['command'].n
    n_planes = env.action_space['plane_id'].n

    policy_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    train_dqn(env, policy_net, target_net, episodes=100, batch_size=32)
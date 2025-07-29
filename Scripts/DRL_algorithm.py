from tabnanny import check
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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_experiences(env_seed, policy_net_state, epsilon, num_episodes=1):
    """Collect experiences from complete episodes."""
    env = AirTrafficControlEnv()
    experiences = []
    
    # Load the policy network state
    policy_net = AirTrafficControlDQN(
        input_dim=env._state_dim(),
        n_commands=env.action_space['command'].n,
        n_planes=env.action_space['plane_id'].n
    ).to(device)  # ADD THIS!
    
    policy_net.load_state_dict(policy_net_state)
    policy_net.eval()
    
    total_reward = 0
    
    #print(f"Policy net device: {next(policy_net.parameters()).device}")  # Should show 'cuda:0'
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=env_seed + episode)
        done = False
        episode_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = {
                    'command': env.action_space['command'].sample(),
                    'plane_id': env.action_space['plane_id'].sample(),
                }
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # ADD .to(device)!
                    #print(f"Input tensor device: {state_tensor.device}")  # Should show 'cuda:0'
                    command_logits, plane_logits = policy_net(state_tensor)
                    #print(f"Output tensor device: {command_logits.device}")  # Should show 'cuda:0'
                    command = torch.argmax(command_logits, dim=1).item()
                    plane_id = torch.argmax(plane_logits, dim=1).item()
                    action = {'command': command, 'plane_id': plane_id}

            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            experiences.append({
                'state': state,
                'action': (action['command'], action['plane_id'], 0),
                'reward': reward,
                'next_state': next_state,
                'done': done
            })
            
            state = next_state
        
        total_reward += episode_reward
    
    return experiences, total_reward

def save_model(policy_net, target_net, optimizer, episode, epsilon, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved at episode {episode} to {filepath}")

def load_model(policy_net, target_net, optimizer, filepath):
    """Load model checkpoint."""
    try:
        checkpoint = torch.load(filepath, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}, resuming from episode {start_episode}")
        return start_episode, epsilon
    except FileNotFoundError:
        print(f"No checkpoint found at {filepath}, starting from scratch")
        return 0, None

def train_dqn_parallel(env, policy_net, target_net, episodes=1000, batch_size=128, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
              num_workers=1, episodes_per_worker=1, checkpoint_dir="checkpoints"):  # Changed parameter name

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = deque(maxlen=20000)  # Reduced from 100,000 to 20,000
    
    # Try to load from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    start_episode, loaded_epsilon = load_model(policy_net, target_net, optimizer, checkpoint_path)
    epsilon = loaded_epsilon if loaded_epsilon is not None else epsilon_start

    for episode in range(start_episode, episodes):
        start_time = time.time()
        
        # Collect complete episodes in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(
                    collect_experiences,
                    env_seed=episode * num_workers + worker_id,
                    policy_net_state=policy_net.state_dict(),
                    epsilon=epsilon,
                    num_episodes=episodes_per_worker  # Complete episodes, not steps
                )
                futures.append(future)
            
            # Collect results
            total_episode_reward = 0
            experiences = []  # Collect experiences here for training
            for future in as_completed(futures):
                worker_experiences, worker_reward = future.result()
                total_episode_reward += worker_reward
                experiences.extend(worker_experiences)  # Add worker experiences to the main list
                
                # Add experiences to memory
                for exp in worker_experiences:
                    memory.append((
                        exp['state'],
                        exp['action'],
                        exp['reward'],
                        exp['next_state'],
                        torch.tensor(exp['done'])
                    ))
        
        # Train with reduced frequency for speed
        if len(memory) >= batch_size:
            num_new_experiences = len(experiences)
            training_steps = max(5, num_new_experiences // 15)  # Reduced training frequency
            
            for _ in range(training_steps):
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

        # Save checkpoint every 50 episodes
        if episode % 50 == 0 and episode > 0:
            save_model(policy_net, target_net, optimizer, episode, epsilon, 
                      os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth"))
            # Also save as latest checkpoint
            save_model(policy_net, target_net, optimizer, episode, epsilon, checkpoint_path)

        # Print progress
        if episode % 10 == 0 or episode < 10:
            avg_reward = total_episode_reward / num_workers
            episode_time = time.time() - start_time
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Time: {episode_time:.2f}s, Epsilon: {epsilon:.3f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_model(policy_net, target_net, optimizer, episodes-1, epsilon, final_path)


def train_dqn_with_checkpoints(env, policy_net, target_net, episodes=1000, batch_size=64, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, target_update=10,
                              checkpoint_dir="checkpoints"):

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = deque(maxlen=100000)
    
    # Try to load from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    start_episode, loaded_epsilon = load_model(policy_net, target_net, optimizer, checkpoint_path)
    epsilon = loaded_epsilon if loaded_epsilon is not None else epsilon_start

    for episode in range(start_episode, episodes):
        start_time = time.time()
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

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
            step_count += 1

            memory.append((state, (action['command'], action['plane_id'], 0),
                          reward, next_state, torch.tensor(done)))
            state = next_state

            # Train every 4 steps for efficiency
            if len(memory) >= batch_size and step_count % 4 == 0:
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

        # Save checkpoint every 50 episodes
        if episode % 50 == 0 and episode > 0:
            save_model(policy_net, target_net, optimizer, episode, epsilon, 
                      os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth"))
            save_model(policy_net, target_net, optimizer, episode, epsilon, checkpoint_path)

        # Print progress
        if episode % 10 == 0 or episode < 10:
            episode_time = time.time() - start_time
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, "
                  f"Steps: {step_count}, Time: {episode_time:.2f}s, Epsilon: {epsilon:.3f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_model(policy_net, target_net, optimizer, episodes-1, epsilon, final_path)


def train_dqn(env, policy_net, target_net, episodes=1000, batch_size=64, gamma=0.99,
              epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, target_update=10):

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = deque(maxlen=10000)
    epsilon = epsilon_start

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_count = 0

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
            step_count += 1

            memory.append((state, (action['command'], action['plane_id'], 0),
                          reward, next_state, torch.tensor(done)))
            state = next_state

            # Train less frequently to reduce overhead
            if len(memory) >= batch_size and step_count % 4 == 0:  # Train every 4 steps
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

        # Print progress less frequently
        if episode % 10 == 0 or episode < 10:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Steps: {step_count}, Epsilon: {epsilon:.3f}")


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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = AirTrafficControlEnv()
    input_dim = env._state_dim()
    n_commands = env.action_space['command'].n
    n_planes = env.action_space['plane_id'].n

    policy_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    train_dqn_parallel(env, policy_net, target_net, episodes=1000, 
                      batch_size=64,
                      num_workers=1,
                      episodes_per_worker=1,
                      checkpoint_dir="checkpoints")
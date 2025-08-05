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
from AnalysisScripts.simple_csv_writer import write_episode_to_csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os

test = True
test = True

class AirTrafficControlDQN(nn.Module):
    def __init__(self, input_dim=61, n_commands=4, n_planes=10):  # Updated to 61: 60 plane features + 1 tick
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),
        )

        # Output heads
        self.command_head = nn.Linear(512, n_commands)   # logits for command
        self.plane_head = nn.Linear(512, n_planes)       # logits for target plane

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
    
    # For multi-discrete actions, treat the combination as a single Q-value
    # Use the raw logits, not log probabilities
    cmd_q = command_logits[0, cmd_idx]
    plane_q = plane_logits[0, plane_idx]
    q_val = cmd_q + plane_q  # Simple additive combination
    return q_val.unsqueeze(0)


def compute_max_q_value(command_logits, plane_logits):
    # Max over possible command/plane pairs
    # Ensure batch dimension exists
    if command_logits.dim() == 1:
        command_logits = command_logits.unsqueeze(0)
    if plane_logits.dim() == 1:
        plane_logits = plane_logits.unsqueeze(0)
    
    # Compute all possible Q-values for command-plane combinations
    all_qs = command_logits.unsqueeze(2) + plane_logits.unsqueeze(1)  # (batch, n_commands, n_planes)
    max_q = torch.max(all_qs.view(all_qs.size(0), -1), dim=1).values
    return max_q

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collect_experiences(env_seed, policy_net_state, epsilon, num_episodes=1):
    """Collect experiences from complete episodes."""
    env = AirTrafficControlEnv(test=test)
    experiences = []
    
    # Load the policy network state
    policy_net = AirTrafficControlDQN(
        input_dim=env._state_dim(),
        n_commands=4,  # Fixed: Use direct value instead of action_space access
        n_planes=10   # Fixed: Use direct value instead of action_space access
    ).to(device)  # ADD THIS!
    
    policy_net.load_state_dict(policy_net_state)
    policy_net.eval()
    
    total_reward = 0
    
    #print(f"Policy net device: {next(policy_net.parameters()).device}")  # Should show 'cuda:0'
    
    for episode in range(num_episodes):
        state, info = env.reset(seed=env_seed + episode)
        done = False
        episode_reward = 0
        
        while not done:
            # Get action mask
            action_mask = info.get('action_mask', {'command': np.ones(4, dtype=bool), 
                                                 'plane_id': np.ones(10, dtype=bool)})
            
            if random.random() < epsilon:
                # Masked random action - use command-plane combination masking for better exploration
                combo_mask = action_mask.get('command_plane_combinations', np.ones((4, 10), dtype=bool))
                
                # Find all valid command-plane combinations
                valid_combinations = []
                for cmd in range(4):
                    if action_mask['command'][cmd]:
                        valid_planes_for_cmd = np.where(combo_mask[cmd, :])[0]
                        for plane in valid_planes_for_cmd:
                            if action_mask['plane_id'][plane]:
                                valid_combinations.append((cmd, plane))
                
                # If no valid combinations, fall back to basic masking
                if len(valid_combinations) == 0:
                    valid_commands = np.where(action_mask['command'])[0]
                    valid_planes = np.where(action_mask['plane_id'])[0]
                    
                    command = np.random.choice(valid_commands) if len(valid_commands) > 0 else 0
                    plane_id = np.random.choice(valid_planes) if len(valid_planes) > 0 else 0
                else:
                    # Sample from valid combinations
                    command, plane_id = random.choice(valid_combinations)
                    
                action = {'command': command, 'plane_id': plane_id}
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # ADD .to(device)!
                    #print(f"Input tensor device: {state_tensor.device}")  # Should show 'cuda:0'
                    command_logits, plane_logits = policy_net(state_tensor)
                    #print(f"Output tensor device: {command_logits.device}")  # Should show 'cuda:0'
                    
                    # Apply command-plane combination masking
                    command_logits_masked = command_logits.clone()
                    plane_logits_masked = plane_logits.clone()
                    
                    # Get the 2D mask for valid command-plane combinations
                    combo_mask = action_mask.get('command_plane_combinations', np.ones((4, 10), dtype=bool))
                    
                    # Set invalid actions to very negative values
                    command_logits_masked[0, ~action_mask['command']] = -float('inf')
                    plane_logits_masked[0, ~action_mask['plane_id']] = -float('inf')
                    
                    # Select command first
                    command = torch.argmax(command_logits_masked, dim=1).item()
                    
                    # Then select plane based on what's valid for that specific command
                    valid_planes_for_command = combo_mask[command, :]
                    if np.any(valid_planes_for_command):
                        # Mask plane logits to only valid planes for this command
                        plane_logits_for_command = plane_logits_masked.clone()
                        plane_logits_for_command[0, ~valid_planes_for_command] = -float('inf')
                        plane_id = torch.argmax(plane_logits_for_command, dim=1).item()
                    else:
                        # Fallback to any valid plane if no command-specific plane is available
                        plane_id = torch.argmax(plane_logits_masked, dim=1).item()
                    
                    action = {'command': command, 'plane_id': plane_id}

            next_state, reward, done, _, info = env.step(action)
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
              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995, target_update=10,
              num_workers=1, episodes_per_worker=1, checkpoint_dir="checkpoints", checkpoint_file="latest_checkpoint.pth"):  # Changed parameter name

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)  # Reduced learning rate
    memory = deque(maxlen=100000)  # Reduced from 100,000 to 20,000
    
    # Try to load from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
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
        if test:
            env.fs.pg_display.stop_display()
        # Save checkpoint every 50 episodes
        if episode % 50 == 0 and episode > 0:
            save_model(policy_net, target_net, optimizer, episode, epsilon, 
                      os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}.pth"))
            # Also save as latest checkpoint
            save_model(policy_net, target_net, optimizer, episode, epsilon, checkpoint_path)

        # Print progress
        if episode % 10 == 0 or episode < 10 or test:
            avg_reward = total_episode_reward / num_workers
            episode_time = time.time() - start_time
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                  f"Time: {episode_time:.2f}s, Epsilon: {epsilon:.3f}")
            
        # write_episode_to_csv(episode, env, total_episode_reward, 0, filename='testing_data.csv')

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    save_model(policy_net, target_net, optimizer, episodes-1, epsilon, final_path)


def train_dqn_with_checkpoints(env, policy_net, target_net, episodes=1000, batch_size=64, gamma=0.99,
                              epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995, target_update=10,
                              checkpoint_dir="checkpoints"):

    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)  # Reduced learning rate
    memory = deque(maxlen=100000)
    
    # Try to load from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    start_episode, loaded_epsilon = load_model(policy_net, target_net, optimizer, checkpoint_path)
    epsilon = loaded_epsilon if loaded_epsilon is not None else epsilon_start

    for episode in range(start_episode, episodes):
        start_time = time.time()
        state, info = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # Get action mask
            action_mask = info.get('action_mask', {'command': np.ones(4, dtype=bool), 
                                                 'plane_id': np.ones(10, dtype=bool)})
            
            # Epsilon-greedy action with masking
            if random.random() < epsilon:
                # Masked random action - only sample from valid actions
                valid_commands = np.where(action_mask['command'])[0]
                valid_planes = np.where(action_mask['plane_id'])[0]
                
                if len(valid_commands) == 0:
                    command = 0  # Default to NONE command
                else:
                    command = np.random.choice(valid_commands)
                    
                if len(valid_planes) == 0:
                    plane_id = 0  # Default to plane 0 (though this shouldn't happen)
                else:
                    plane_id = np.random.choice(valid_planes)
                    
                action = {'command': command, 'plane_id': plane_id}
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    command_logits, plane_logits = policy_net(state_tensor)
                    
                    # Apply action masking
                    command_logits_masked = command_logits.clone()
                    plane_logits_masked = plane_logits.clone()
                    
                    # Set invalid actions to very negative values
                    command_logits_masked[0, ~action_mask['command']] = -float('inf')
                    plane_logits_masked[0, ~action_mask['plane_id']] = -float('inf')
                    
                    command = torch.argmax(command_logits_masked, dim=1).item()
                    plane_id = torch.argmax(plane_logits_masked, dim=1).item()
                    action = {'command': command, 'plane_id': plane_id}

            next_state, reward, done, _, info = env.step(action)
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
        if episode % 10 == 0 or episode < 10 or test:
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
        if episode % 10 == 0 or episode < 10 or test:
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


def test_dqn(model_filepath, episodes=5, display=True, recordData=False, filename='m11xl_data.csv'):
    """
    Test a trained DQN model by running episodes with optional display.
    
    Args:
        model_filepath (str): Path to the saved model checkpoint
        episodes (int): Number of episodes to run for testing
        display (bool): Whether to show the pygame display during testing
    
    Returns:
        list: List of rewards for each episode
    """
    import pygame

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    record_data = recordData
    
    # Initialize environment - same pattern as training functions
    env = AirTrafficControlEnv(test=test, record_data=record_data, no_display=not display)  # Pass display control to environment

    # Set display mode - FlightSimulator handles all display initialization
    env.fs.no_display = not display
    
    # Create the policy network - same as training functions
    
    policy_net = AirTrafficControlDQN(
        input_dim=env._state_dim(), 
        n_commands=4,  # Fixed: Use direct value instead of action_space access
        n_planes=10    # Fixed: Use direct value instead of action_space access
    ).to(device)
    
    # Load the trained model
    try:
        checkpoint = torch.load(model_filepath, map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        print(f"Model loaded successfully from {model_filepath}")
        print(f"Model was trained for {checkpoint['episode']} episodes")
        print(f"Final epsilon value: {checkpoint['epsilon']:.4f}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_filepath}")
        return []
    except KeyError as e:
        print(f"Error: Invalid checkpoint format, missing key: {e}")
        return []
    
    policy_net.eval()  # Set to evaluation mode
    
    episode_rewards = []
    
    print(f"\nRunning {episodes} test episodes...")
    
    for episode in range(episodes):
        state, info = env.reset(seed=checkpoint['episode'] + episode)  # Use episode number for reproducibility
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        
        while not done:
            # Use the trained policy (no epsilon-greedy exploration)
            # Same pattern as training functions

            action_mask = info.get('action_mask', {'command': np.ones(4, dtype=bool), 
                                                'plane_id': np.ones(10, dtype=bool)})
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # ADD .to(device)!
                #print(f"Input tensor device: {state_tensor.device}")  # Should show 'cuda:0'
                command_logits, plane_logits = policy_net(state_tensor)
                #print(f"Output tensor device: {command_logits.device}")  # Should show 'cuda:0'
                
                # Apply action masking
                command_logits_masked = command_logits.clone()
                plane_logits_masked = plane_logits.clone()
                
                # Get the 2D mask for valid command-plane combinations
                combo_mask = action_mask.get('command_plane_combinations', np.ones((4, 10), dtype=bool))
                
                # Apply masking: for each command, only allow valid plane targets
                # Set invalid actions to very negative values
                command_logits_masked[0, ~action_mask['command']] = -float('inf')
                plane_logits_masked[0, ~action_mask['plane_id']] = -float('inf')
                
                # Select command first
                command = torch.argmax(command_logits_masked, dim=1).item()
                command = int(command)
                
                # Then select plane based on what's valid for that specific command
                valid_planes_for_command = combo_mask[command, :]
                if np.any(valid_planes_for_command):
                    # Mask plane logits to only valid planes for this command
                    plane_logits_for_command = plane_logits_masked.clone()
                    plane_logits_for_command[0, ~valid_planes_for_command] = -float('inf')
                    plane_id = torch.argmax(plane_logits_for_command, dim=1).item()
                else:
                    # Fallback to any valid plane if no command-specific plane is available
                    plane_id = torch.argmax(plane_logits_masked, dim=1).item()
                
                plane_id = int(plane_id)
                action = {'command': command, 'plane_id': plane_id}
            
            # Take the action - same as training functions
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Handle display events if display is enabled
            # FlightSimulator handles all display updates automatically during env.step()
            if display:
                try:
                    # Handle pygame events to prevent window from becoming unresponsive
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("Display window closed, ending test")
                            return episode_rewards
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                print("ESC pressed, ending test")
                                return episode_rewards
                except Exception as e:
                    print(f"Display error: {e}")
                    # Don't disable display, let FlightSimulator handle it
            
            state = next_state
            
            # Debug: Track command selection
            if step_count % 50 == 0:
                command_names = {0: "NONE", 1: "LINE_UP_AND_WAIT", 2: "CLEARED_TO_LAND", 3: "GO_AROUND"}
                valid_planes = np.where(action_mask['plane_id'])[0]
                print(f"  Step {step_count}: {command_names[command]} → Plane {plane_id}")
                print(f"    Valid planes: {valid_planes}")
                print(f"    Reward: {reward:.2f}")
                print(f"    Planes in sim: {len(env.fs.plane_manager.planes)}")
                
                # Quick state check
                plane_states = {}
                for plane in env.fs.plane_manager.planes:
                    state_name = plane.state.name
                    plane_states[state_name] = plane_states.get(state_name, 0) + 1
                print(f"    Plane states: {plane_states}")
        
        if recordData:
            write_episode_to_csv(episode, env, episode_reward, step_count, filename=filename)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1} completed: {step_count} steps, Total Reward: {episode_reward:.2f}")
    
    # Print summary statistics
    print(f"\n=== Test Results ===")
    print(f"Episodes: {episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Deviation: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {min(episode_rewards):.2f}")
    print(f"Max Reward: {max(episode_rewards):.2f}")
    
    # FlightSimulator handles display cleanup automatically
    if display:
        print("\nDisplay managed by FlightSimulator. Close window manually or press ESC to exit.")
    
    return episode_rewards

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = AirTrafficControlEnv(test=test)  # Set test=True for evaluation mode
    input_dim = env._state_dim()
    n_commands = env.action_space['command'].n
    n_planes = env.action_space['plane_id'].n

    policy_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net = AirTrafficControlDQN(input_dim=input_dim, n_commands=n_commands, n_planes=n_planes).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Uncomment the line below to train the model
    # train_dqn_parallel(env, policy_net, target_net, episodes=1000, 
    #                   batch_size=256,
    #                   num_workers=1,
    #                   episodes_per_worker=1,
    #                   checkpoint_dir="checkpoints",
    #                   checkpoint_file="latest_checkpoint.pth",
    #                   epsilon_start=1.0,
    #                   epsilon_end=0.01,
    #                     epsilon_decay=0.995
    #                   )
    
    # Example: Test a trained model
    # Uncomment the lines below to test a trained model with display
    model_path = "checkpoints/M11XL.pth"  # Use absolute path
    rewards = test_dqn(model_path, episodes=100, display=True, recordData=True)
    print(f"Test completed. Rewards: {rewards}")

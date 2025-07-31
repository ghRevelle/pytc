#!/usr/bin/env python3

import torch
import numpy as np
from DRL_env import AirTrafficControlEnv
from DRL_algorithm import AirTrafficControlDQN
import random

def debug_single_episode():
    """Run a single episode with detailed logging to see what's happening."""
    
    # Create environment
    env = AirTrafficControlEnv(max_planes=10, max_ticks=200, test=False)  # Shorter episode for debugging
    
    # Create a simple random policy for testing
    obs, info = env.reset()
    
    print("=== DEBUGGING SINGLE EPISODE ===")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0
    step_count = 0
    
    while True:
        # Take random action for debugging
        action = {
            'command': random.randint(0, 3),  # Random command
            'plane_id': random.randint(0, 9)  # Random plane
        }
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        if reward != 0:
            print(f"Step {step_count}: Action={action}, Reward={reward:.3f}")
        
        if done:
            print("\n=== EPISODE COMPLETE ===")
            print(f"Total steps: {step_count}")
            print(f"Total reward: {total_reward:.3f}")
            
            if 'episode_stats' in info:
                stats = info['episode_stats']
                print("\nEpisode Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                
                # Calculate major penalty sources
                crash_penalty = stats['crashes'] * -25.0
                invalid_penalty = stats['invalid_commands'] * -0.1
                time_penalty = stats['time_pressure_penalties'] * -0.01
                
                print(f"\nPenalty Breakdown:")
                print(f"  Crash penalties: {crash_penalty:.3f} ({stats['crashes']} crashes)")
                print(f"  Invalid command penalties: {invalid_penalty:.3f} ({stats['invalid_commands']} invalid)")
                print(f"  Time pressure penalties: {time_penalty:.3f} ({stats['time_pressure_penalties']} penalty ticks)")
                
                positive_rewards = (stats['landings'] * 10.0 + 
                                  stats['takeoffs'] * 8.0 + 
                                  stats['go_arounds'] * 1.0 + 
                                  stats['no_commands'] * 0.05)
                print(f"  Total positive rewards: {positive_rewards:.3f}")
                
            break
    
    return total_reward, info.get('episode_stats', {})

def debug_action_validation():
    """Test if action validation is working properly."""
    
    print("\n=== DEBUGGING ACTION VALIDATION ===")
    
    env = AirTrafficControlEnv(max_planes=10, max_ticks=50, test=False)
    obs, info = env.reset()
    
    # Test various actions to see which ones are invalid
    test_actions = [
        {'command': 0, 'plane_id': 0},  # NONE command
        {'command': 1, 'plane_id': 0},  # LINE_UP_AND_WAIT
        {'command': 2, 'plane_id': 0},  # CLEARED_TO_LAND
        {'command': 3, 'plane_id': 0},  # GO_AROUND
        {'command': 1, 'plane_id': 99}, # Invalid plane ID
    ]
    
    for i, action in enumerate(test_actions):
        obs, reward, done, truncated, info = env.step(action)
        print(f"Action {i}: {action} -> Reward: {reward:.3f}")
        
        if done:
            break

if __name__ == "__main__":
    # Run debugging
    total_reward, stats = debug_single_episode()
    debug_action_validation()
    
    print(f"\n=== SUMMARY ===")
    print(f"Random policy achieved: {total_reward:.3f} total reward")
    
    if stats:
        # Identify the biggest problem
        if stats['invalid_commands'] > stats['landings'] + stats['takeoffs']:
            print("❌ MAIN ISSUE: Too many invalid commands!")
            print("   Solution: Improve action validation or mask invalid actions")
        
        if stats['time_pressure_penalties'] > 1000:
            print("❌ MAIN ISSUE: Excessive time pressure penalties!")
            print("   Solution: Reduce time pressure penalty or improve efficiency")
        
        if stats['crashes'] > 0:
            print("❌ MAIN ISSUE: Crashes occurring!")
            print("   Solution: Improve crash detection and avoidance")
        
        if stats['landings'] + stats['takeoffs'] == 0:
            print("❌ MAIN ISSUE: No successful actions!")
            print("   Solution: Check if commands are being processed correctly")

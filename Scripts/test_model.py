#!/usr/bin/env python3
"""
Example script to test a trained DQN model.

This script demonstrates how to use the test_dqn function to evaluate
a trained air traffic control DQN model with visual display.

Usage:
    python test_model.py [model_path] [--episodes N] [--no-display]

Examples:
    python test_model.py checkpoints/final_model.pth
    python test_model.py checkpoints/checkpoint_episode_500.pth --episodes 5
    python test_model.py checkpoints/latest_checkpoint.pth --no-display
"""

import sys
import argparse
import os
from DRL_algorithm import test_dqn

def main():
    parser = argparse.ArgumentParser(description='Test a trained DQN model for air traffic control')
    parser.add_argument('model_path', 
                       help='Path to the trained model checkpoint file')
    parser.add_argument('--episodes', '-e', type=int, default=3,
                       help='Number of test episodes to run (default: 3)')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without pygame display (faster)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found!")
        print("\nAvailable checkpoints:")
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.pth'):
                    print(f"  {os.path.join(checkpoint_dir, file)}")
        else:
            print(f"  Checkpoint directory '{checkpoint_dir}' not found")
        return 1
    
    print(f"Testing model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Display: {'Disabled' if args.no_display else 'Enabled'}")
    print("-" * 50)
    
    try:
        # Run the test
        rewards = test_dqn(
            model_filepath=args.model_path,
            episodes=args.episodes,
            display=not args.no_display
        )
        
        if rewards:
            print(f"\nTest completed successfully!")
            print(f"Individual episode rewards: {[f'{r:.2f}' for r in rewards]}")
            return 0
        else:
            print(f"\nTest failed or returned no results.")
            return 1
            
    except KeyboardInterrupt:
        print(f"\nTest interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
Simple CSV writer that adds just one function call to your training loop.
No external classes, no complex setup; just write data directly to CSV.
"""

import pandas as pd
import os
from datetime import datetime

def write_episode_to_csv(episode_num, env, total_reward, ending_time, filename="../Data/testing_data.csv"):
    """
    Write episode statistics directly to CSV file.
    
    Args:
        episode_num: Episode number
        env: Your AirTrafficControlEnv instance
        total_reward: Total reward from the episode
        ending_time: Number of ticks the episode lasted
        filename: CSV filename to write to
    """
    
    # If filename is just a name, put it in the Data/GeneratedData directory
    if not os.path.dirname(filename):
        # Get the script directory and go up to project root, then to Data/GeneratedData
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_dir = os.path.join(project_root, "Data", "GeneratedData")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        filename = os.path.join(data_dir, filename)
    
    # Extract stats from environment
    stats = env.episode_stats if hasattr(env, 'episode_stats') else {}
    
    # Collect the data you requested
    data = {
        'episode': episode_num,
        'total_reward': stats['total_reward'],
        'max_reward': stats['max_reward'],  # Your target
        'ending_time': ending_time,
        'planes_taken_off': stats['planes_taken_off'],
        'planes_landed': stats['planes_landed'],
        'planes_encountered': stats['planes_encountered'],
        'go_arounds': stats['go_arounds'],
        'crashes': stats['crashes'],
        'processed_planes': stats['processed_planes'],
        'reward_efficiency': stats['reward_efficiency']
    }
    df = pd.DataFrame(data)
    
    # Check if file exists to decide if we need to write headers
    file_exists = os.path.exists(filename)
    
    # Write to CSV
    df.to_csv(filename, mode='a', header=not file_exists, index=False)

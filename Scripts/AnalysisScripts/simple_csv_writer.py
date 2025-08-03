"""
Simple CSV writer that adds just one function call to your training loop.
No external classes, no complex setup; just write data directly to CSV.
"""

import os
import pandas as pd

def write_episode_to_csv(episode_num, env, total_reward, ending_time, filename="m9-3350_data.csv"):
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
    
    # Create data dictionary
    data = {
        'episode': episode_num,
        'total_reward': stats.get('total_reward', 0),
        'max_reward': stats.get('max_reward', 0),
        'end_time': ending_time,
        'planes_taken_off': stats.get('planes_taken_off', 0),
        'planes_landed': stats.get('planes_landed', 0),
        'planes_encountered': stats.get('planes_encountered', 0),
        'go_arounds': stats.get('go_arounds', 0),
        'crashes': stats.get('crashes', 0),
        'processed_planes': stats.get('processed_planes', 0),
        'reward_efficiency': stats.get('reward_efficiency', 0)
    }
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Check if file exists
    file_exists = os.path.exists(filename)
    
    # Write to CSV
    if file_exists:
        # Append without header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # Write with header
        df.to_csv(filename, mode='w', header=True, index=False)

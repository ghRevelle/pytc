"""
Simple CSV writer that adds just one function call to your training loop.
No external classes, no complex setup; just write data directly to CSV.
"""

import csv
import os

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
    
    # Define headers
    headers = ['episode', 'total_reward', 'max_reward', 'end_time', 'planes_taken_off', 
               'planes_landed', 'planes_encountered', 'go_arounds', 'crashes', 
               'processed_planes', 'reward_efficiency']
    
    # Check if file exists to decide if we need to write headers
    file_exists = os.path.exists(filename)
    
    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers if file is new
        if not file_exists:
            # Add comment header
            writer.writerow(['# episode, total_reward, max_reward, end_time, planes_taken_off, planes_landed, planes_encountered, go_arounds, crashes, processed_planes, reward_efficiency'])
            writer.writerow(headers)
        
        # Write the data row directly from stats
        row = [
            episode_num,
            stats['total_reward'],
            stats['max_reward'],
            ending_time,
            stats['planes_taken_off'],
            stats['planes_landed'],
            stats['planes_encountered'],
            stats['go_arounds'],
            stats['crashes'],
            stats['processed_planes'],
            stats['reward_efficiency']
        ]
        writer.writerow(row)

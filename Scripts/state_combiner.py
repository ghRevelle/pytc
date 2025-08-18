import re

def extract_states_from_file(filename):
    """Extract all rolling_initial_state variables from a file."""
    states = {}
    
    with open(filename, 'r') as file:
        content = file.read()
    
    # Find all rolling_initial_state_XX = [...] patterns
    pattern = r'rolling_initial_state_(\d+)\s*=\s*(\[.*?\])'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for state_num, state_data in matches:
        states[int(state_num)] = state_data
    
    return states

def combine_rolling_states(file1, file2, output_file):
    """Combine rolling states from two files and renumber sequentially."""
    
    # Extract states from both files
    states1 = extract_states_from_file(file1)
    states2 = extract_states_from_file(file2)

    # Sort by original state number to maintain order
    sorted_states1 = [state for _, state in sorted(states1.items())]
    sorted_states2 = [state for _, state in sorted(states2.items())]

    # Concatenate the two lists
    all_states = sorted_states1 + sorted_states2

    # Generate the combined file
    with open(output_file, 'w') as f:
        f.write("# Combined rolling initial states\n\n")
        for new_index, state_data in enumerate(all_states):
            f.write(f"rolling_initial_state_{new_index:04d} = {state_data}\n")

    print(f"Combined {len(all_states)} states into {output_file}")
    print(f"States renumbered from 0000 to {len(all_states)-1:04d}")

def combine_multiple_files(file_list, output_file):
    """Combine rolling states from multiple files."""
    
    all_states = {}
    
    # Extract states from all files
    for filename in file_list:
        states = extract_states_from_file(filename)
        all_states.update(states)
        print(f"Extracted {len(states)} states from {filename}")
    
    # Sort by original state number
    sorted_states = sorted(all_states.items())
    
    # Generate the combined file
    with open(output_file, 'w') as f:
        f.write("# Combined rolling initial states from multiple files\n")
        f.write(f"# Source files: {', '.join(file_list)}\n\n")
        
        for new_index, (original_index, state_data) in enumerate(sorted_states):
            f.write(f"rolling_initial_state_{new_index:02d} = {state_data}\n")
    
    print(f"\nCombined {len(sorted_states)} total states into {output_file}")
    print(f"States renumbered from 00 to {len(sorted_states)-1:02d}")

# Example usage:
if __name__ == "__main__":
    # For your specific files:
    file1 = "rolling_initial_state_20250301.py"
    file2 = "rolling_initial_state_20250501.py"
    output = "combined_rolling_states.py"
    
    # Method 1: Combine two files
    combine_rolling_states(file1, file2, output)
    
    # Method 2: Combine multiple files (if you have more)
    # file_list = [
    #     "rolling_initial_state_20250301.py",
    #     "rolling_initial_state_20250501.py"
    # ]
    # combine_multiple_files(file_list, "combined_rolling_states.py")
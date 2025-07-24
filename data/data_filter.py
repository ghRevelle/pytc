# Filter operations data by airport code

import pandas as pd

df = pd.read_csv('data/operations.csv') # Load the CSV file; not included in this repository due to size

# Filter by specific airport
df_filtered = df[df['airport'] == 'KSAN']
# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('data/filtered_data/filtered_operations_20250301.csv', index=False)
# Filter operations data by airport code

import pandas as pd

df = pd.read_csv('operations.csv') # Load the CSV file; not included in this repository due to size

# Filter by specific airport
df_filtered = df[df['airport'] == 'KRAP']
# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv('filtered_operations.csv', index=False)
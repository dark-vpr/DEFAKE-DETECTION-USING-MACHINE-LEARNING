import pandas as pd

# Read the original CSV file into a Pandas DataFrame
df = pd.read_csv('metadata.csv')

# Extract the "name" and "label" columns
new_df = df[['URI', 'label']]

# Write the new DataFrame to a new CSV file
new_df.to_csv('new_file.csv', index=False)

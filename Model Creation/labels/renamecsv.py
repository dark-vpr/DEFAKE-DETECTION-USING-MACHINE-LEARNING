import os
import pandas as pd

# Set the directory where the files are located
directory = 'D:\SHEBIN\PROJECT\DEEPFAKE\DATASET\deepfake-detection\deep-fake-dataset\Celeb-Youtube-fake'

# Get a list of all the files in the directory
files = os.listdir(directory)

# Sort the files in alphabetical order
files.sort()

# Create a DataFrame to store the file names and labels
df = pd.DataFrame(columns=['name', 'label'])

# Rename the files and add the file names and labels to the DataFrame
for i, file in enumerate(files):
    new_filename = f'celytbf{i+1}.mp4'
    old_path = os.path.join(directory, file)
    new_path = os.path.join(directory, new_filename)
    os.rename(old_path, new_path)
    df = df.append({'name': new_filename, 'label': 'REAL'}, ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv('celytbf.csv', index=False)

print('File renaming and CSV creation complete!')
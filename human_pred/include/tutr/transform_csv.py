import pandas as pd
from sklearn.model_selection import train_test_split
import os

position_columns = ['Time', 'X', 'Y', 'Z']
# orientation_columns = ['Time', 'Roll', 'Pitch', 'Yaw']
base_directory = './dataset/Offical_data_sharing'
output_directory = './data/sfu/train/'
agent_id_counter = 0  # Initialize agent ID counter

def process_files(position_file_path, agent_id):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the CSV files
    position_df = pd.read_csv(position_file_path, names=position_columns)
    # orientation_df = pd.read_csv(orientation_file_path, names=orientation_columns)

    # Convert 'Time' column to datetime
    position_df['Time'] = pd.to_datetime(position_df['Time'], unit='s')
    # orientation_df['Time'] = pd.to_datetime(orientation_df['Time'], unit='s')

    # Interpolate the data
    position_df.set_index('Time', inplace=True)
    # orientation_df.set_index('Time', inplace=True)

    # Resample the data to a consistent interval of 0.4 seconds and fill missing values
    position_df = position_df.resample('400ms').ffill().bfill()
    # orientation_df = orientation_df.resample('400ms').ffill().bfill()

    # Reset index to get Time column back
    position_df.reset_index(inplace=True)
    # orientation_df.reset_index(inplace=True)

    # Convert 'Time' column to milliseconds starting from 0
    initial_time = position_df['Time'].iloc[0]
    position_df['frame_id'] = (position_df['Time'] - initial_time).dt.total_seconds() * 1000

    # Round frame_id to the nearest 400 ms to ensure consistency
    position_df['frame_id'] = (position_df['frame_id'] // 400) * 400

    # Sort the DataFrame by 'frame_id'
    position_df.sort_values(by='frame_id', inplace=True)

    # Initialize an empty list to store formatted data
    formatted_data = []

    # Iterate over the rows of the position DataFrame
    for row in position_df.itertuples():
        frame_id = int(row.frame_id)  # Use the computed frame_id in milliseconds
        pos_x = row.X
        pos_y = row.Y
        group = 'human'  # Assuming a single group

        if pd.isna(pos_x) or pd.isna(pos_y):
            continue
        
        formatted_data.append(f"{frame_id} {agent_id} {pos_x} {pos_y} {group}")

    # Ensure frame_id is in ascending order
    formatted_data.sort(key=lambda x: int(x.split()[0]))

    # Use the parent folder's name for the output filename
    folder_name = os.path.basename(os.path.dirname(position_file_path)).replace('-', '_')
    unique_file_name = f"{folder_name}_translation.txt"
    output_file_path = os.path.join(output_directory, unique_file_name)

    # Save data to a file
    with open(output_file_path, 'w') as file:
        for line in formatted_data:
            file.write(line + '\n')
    
    print(f"Data processed and saved to: {output_file_path}")

# Walk through the directory structure
if not os.path.exists(base_directory):
    print(f"Directory not found: {base_directory}")
else:
    print(f"Directory found: {base_directory}")

for subdir, dirs, files in os.walk(base_directory):
    print(f"Processing files in: {subdir}")
    for file in files:
        print(f"Processing file: {file}")
        if file.endswith("_translation.csv"):
            agent_id_counter += 1  # Increment agent ID for each file
            print(f"Processing translation file: {file}")
            position_file_path = os.path.join(subdir, file)
            process_files(position_file_path, agent_id_counter)



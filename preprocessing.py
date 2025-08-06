import os
import re
import pandas as pd

def extract_config_segments(user_folder: str, output_folder: str, time_buffer: float = 0.5):
    """
    Extracts config segments from each transform log file based on event log timelines.

    Args:
        user_folder (str): Path to the folder containing a user's event log and transform logs.
        output_folder (str): Path where extracted config segments should be saved.
        time_buffer (float): Extra time (in seconds) to add at the end of each segment.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Locate the event log file
    event_file = next((f for f in os.listdir(user_folder) if "EventLogFile" in f and f.endswith(".csv")), None)
    if not event_file:
        raise FileNotFoundError("Event log file not found in user folder.")

    event_path = os.path.join(user_folder, event_file)
    event_df = pd.read_csv(event_path)

    # Filter rows corresponding to config segments (Name like 'config0', 'config1', ...)
    config_rows = event_df[event_df['Name'].str.match(r'config\d+')]

    # Process each config segment
    for _, row in config_rows.iterrows():
        config_name = row['Name']  # e.g., 'config19'
        start_time = row['StartTime']
        end_time = row['EndTime'] + time_buffer  # Add safety margin

        # Find the corresponding transform log file
        transform_file_pattern = re.compile(rf".*_{config_name}\.csv")
        transform_file = next((f for f in os.listdir(user_folder) if transform_file_pattern.match(f)), None)

        if not transform_file:
            print(f"Warning: Transform log for {config_name} not found.")
            continue

        transform_path = os.path.join(user_folder, transform_file)
        transform_df = pd.read_csv(transform_path)

        # Filter transform log based on timestamps
        filtered_df = transform_df[
            (transform_df['Timestamp'] >= start_time) &
            (transform_df['Timestamp'] <= end_time)
        ]

        # Save cropped data
        output_path = os.path.join(output_folder, transform_file)
        filtered_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")


def process_all_users(base_input_folder: str, base_output_folder: str):
    """
    Processes all user folders from user0 to user12 (excluding user1).

    Args:
        base_input_folder (str): Base path where all user folders are located.
        base_output_folder (str): Base path to save all processed outputs.
    """
    for i in range(13):
        if i == 1:
            continue  # Skip user1 as it's invalid

        user_folder = os.path.join(base_input_folder, f"user{i}")
        output_folder = os.path.join(base_output_folder, f"user{i}")

        if not os.path.exists(user_folder):
            print(f"User folder not found: {user_folder}")
            continue

        print(f"Processing {user_folder}...")
        extract_config_segments(user_folder, output_folder)

process_all_users(base_input_folder="RawData", base_output_folder="processed")

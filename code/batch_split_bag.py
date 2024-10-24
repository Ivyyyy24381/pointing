import pandas as pd
import argparse
import os
import subprocess
def generate_and_run_commands(df):
    for index, row in df.iterrows():
        command = (
            f"python ~/Documents/GitHub/pointing/code/bag_to_video.py "
            f"--bag_filepath {row['Filename']} "
            f"--date {row['Date']} "
            f"--trial {row['trial#']} "
            f"--start_sec {int(row['start seconds'])} "
            f"--end_sec {int(row['end seconds'])}"
        )
        # print(command)
        print(f"Running command: {command}")
        # Execute the command using subprocess
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        # Check if the command executed successfully
        if process.returncode == 0:
            print(f"Command executed successfully: {command}")
        else:
            print(f"Error occurred while executing command: {command}")
            print(process.stderr)  # Print any error messages

    return command

def main():
    # Argument parser to take CSV file as input
    parser = argparse.ArgumentParser(description='Generate commands from CSV')
    parser.add_argument('--csv_filepath', type=str, help='Path to the CSV file')
    args = parser.parse_args()

    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(args.csv_filepath, header = 2)
    
    
    # Generate the commands based on the CSV content
    command = generate_and_run_commands(df)

if __name__ == "__main__":
    main()

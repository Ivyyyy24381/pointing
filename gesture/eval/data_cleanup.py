import pandas as pd
import argparse

# Function to clean up the data
def clean_data(input_file, output_file, short_gesture_threshold=0.4):
    # Load the data from the CSV file
    df = pd.read_csv(input_file)

    # Step 1: Group by pointing_count to process each unique gesture
    gesture_groups = df.groupby('pointing_count')

    # Step 2: Process each gesture group
    for pointing_count, group in gesture_groups:
        
        # Get the last row of the gesture that has a non-empty pointing_arm to calculate the total duration
        non_empty_rows = group[group['pointing_arm'].notna()]
        
        if len(non_empty_rows) > 0:
            total_gesture_duration = non_empty_rows.iloc[-1]['gesture_duration']  # Last non-empty row's gesture_duration

            # Step 3: If the gesture duration is less than the threshold, make it empty
            if total_gesture_duration < short_gesture_threshold:
                # Clear all columns except 'frame' and 'wrist_location' for this gesture
                columns_to_clear = ['gesture_duration', 'pointing_count', 'pointing_arm', 'eye_to_wrist', 
                                    'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']
                df.loc[group.index, columns_to_clear] = ''  # Clear the specified columns

    # Step 4: Renumber the pointing_count sequentially, but only for non-empty cells
    non_empty_mask = df['pointing_arm'].notna() & df['pointing_arm'].ne('')

    # Get unique gestures for non-empty rows
    unique_gestures = df.loc[non_empty_mask, 'pointing_count'].unique()

    # Create a mapping from old pointing_count to new sequential numbers
    new_gesture_map = {old_count: new_count + 1 for new_count, old_count in enumerate(unique_gestures)}

    # Apply the renumbering to the pointing_count
    df.loc[non_empty_mask, 'pointing_count'] = df.loc[non_empty_mask, 'pointing_count'].map(new_gesture_map)

    # Step 5: Remove all rows where 'pointing_arm' is empty
    df = df[df['pointing_arm'].notna() & df['pointing_arm'].ne('')]

    # Save the cleaned data
    df.to_csv(output_file, index=False)
    print(f"Data cleaned, renumbered, empty rows removed, and saved as '{output_file}'")

# Main function for argument parsing
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean up gesture data based on pointing duration.")
    parser.add_argument('--input', type=str,  help="Path to the input CSV file.")
    parser.add_argument('--output', type=str,  help="Path to save the cleaned output CSV file.")
    parser.add_argument('--threshold', type=float, default=0.4, help="Threshold for short gestures in seconds (default: 0.4).")

    # Parse the arguments
    args = parser.parse_args()
    if args.input:
        # Call the clean_data function with parsed arguments
        clean_data(args.input, args.output, args.threshold)
    else:
        input ='/Users/ivy/Desktop/spot_gesture_eval/3m_video_updated.csv'
        output = '/Users/ivy/Desktop/spot_gesture_eval/3m_video_updated_cleanup.csv'
        clean_data(input, output)
# Entry point for the script
if __name__ == "__main__":
    main()
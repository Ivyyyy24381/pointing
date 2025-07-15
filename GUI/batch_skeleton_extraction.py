# get root folder
# loop over all dog folders in the folder
# in each folder, loop over all trials
# in each trial, look for processed_gesture_Data.csv
# if it exists, load it and append to df
# in df, add a column for dog_id, dog_name, and trial_number
# save the df to a csv file
import yaml
import os
import pandas as pd
import sys          
sys.path.append('./')
sys.path.append('visualize')  # Adjust this path based on your project structure

# save the processed gesture data to a csv file

root_path = input("Enter the root path to the dog folders: ")
if not os.path.exists(root_path):
    print(f"Root path {root_path} does not exist.")
    sys.exit(1) 
root_data_path = os.path.join(root_path, 'overeall_processed_gesture.csv')
    os.remove(root_data_path)
dog_data = []
for dog_folder in os.listdir(root_path):
    print(f'dog_folder:{dog_folder}')
    dog_folder_path = os.path.join(root_path, dog_folder)
    if os.path.isdir(dog_folder_path):
        for trial_folder in os.listdir(dog_folder_path):
            trial_folder_path = os.path.join(dog_folder_path, trial_folder)
            if os.path.isdir(trial_folder_path):
                processed_csv_path = os.path.join(trial_folder_path, "processed_gesture_data.csv")
                if os.path.exists(processed_csv_path):
                    df = pd.read_csv(processed_csv_path)
                    dog_id = dog_folder.split('_')[0]
                    dog_name = dog_folder.split('_')[1]
                    trial_number = trial_folder.split('_')[0]
                    # add dog_id, dog_name, and trial_number to the df
                    df['dog_id'] = dog_id
                    df['dog_name'] = dog_name
                    df['trial_number'] = trial_number
                    # add these parameter to the left of df
                    df = df[['dog_id', 'dog_name', 'trial_number'] + [col for col in df.columns if col not in ['dog_id', 'dog_name', 'trial_number']]]
                    dog_data.append(df)
# concatenate all the dataframes in dog_data
if dog_data:
    overall_df = pd.concat(dog_data, ignore_index=True)
    overall_df.to_csv(root_data_path, index=False)
    print(f"Processed gesture data saved to {root_data_path}")
else:
    print("No processed gesture data found in the specified folders.")  

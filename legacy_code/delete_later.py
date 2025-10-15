import os
import shutil

def organize_bag_files(folder_path):
    # List all .bag files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".bag"):
            file_path = os.path.join(folder_path, file_name)

            # Check if 'Cam2' is in the filename
            if "Cam2" in file_name:
                folder_suffix = "side"
            else:
                folder_suffix = "front"

            # Remove the .bag extension and _Cam2 if present
            base_name = file_name.replace("_Cam2", "").replace(".bag", "")
            folder_name = f"{base_name}_{folder_suffix}"

            # Create the full path for the new folder
            folder_full_path = os.path.join(folder_path, folder_name)
            os.makedirs(folder_full_path, exist_ok=True)

            # Move the .bag file to the new folder
            shutil.move(file_path, os.path.join(folder_full_path, file_name))
            print(f"Moved {file_name} to {folder_name}/")

# Example usage
organize_bag_files("/media/xhe71/TOSHIBA EXT/")
import os
import sys
sys.path.append('visualize')
from gesture_detection import PointingGestureDetector  # Update this import path based on your file structure

def run_gesture_detection(base_path, subject_folder=None, trial_id=None):
    """
    Run gesture detection on a dataset.

    Args:
        base_path (str): Base directory path containing subject folders.
        subject_folder (str, optional): Specific subject folder to process.
        trial_id (str, optional): Specific trial folder name to process.
    """
    subjects = [subject_folder] if subject_folder else os.listdir(base_path)

    for subj in subjects:
        subj_path = os.path.join(base_path, subj)
        if not os.path.isdir(subj_path):
            continue

        trials = [trial_id] if trial_id else os.listdir(subj_path)
        for trial in trials:
            trial_path = os.path.join(subj_path, trial)
            if not os.path.isdir(trial_path):
                continue

            color_video_path = os.path.join(trial_path, 'Color.mp4')
            if not os.path.exists(color_video_path):
                print(f"Color video not found for trial: {trial} in subject: {subj}")
                continue

            print(f"Processing {color_video_path}")
            gesture_processor = PointingGestureDetector()
            gesture_processor.run_video(color_video_path)

            output_csv_path = os.path.join(trial_path, 'gesture_data.csv')
            gesture_processor.data.to_csv(output_csv_path, index=False)
            print(f"Saved gesture data to {output_csv_path}")

if __name__ == "__main__":
    base_path = input("Enter base path to dataset: ").strip()
    subject_folder = input("Enter subject folder (press Enter to process all subjects): ").strip() or None
    trial_id = input("Enter trial ID (press Enter to process all trials): ").strip() or None

    run_gesture_detection(base_path, subject_folder=subject_folder, trial_id=trial_id)
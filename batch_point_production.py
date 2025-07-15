import os
import sys
sys.path.append('visualize')
from gesture_detection import PointingGestureDetector  # Update this import path based on your file structure
import os
import sys
import cv2
from natsort import natsorted


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
                color_image_folder = os.path.join(trial_path, 'Color')
                if os.path.exists(color_image_folder):
                    print(f"'Color.mp4' not found, compiling from images in: {color_image_folder}")
                    image_files = [f for f in os.listdir(color_image_folder) if f.endswith(('.png', '.jpg'))]
                    if not image_files:
                        print(f"No image files found in {color_image_folder}, skipping.")
                        continue
                    image_files = natsorted(image_files)
                    first_frame = cv2.imread(os.path.join(color_image_folder, image_files[0]))
                    height, width, _ = first_frame.shape
                    out = cv2.VideoWriter(color_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
                    for img_name in image_files:
                        img_path = os.path.join(color_image_folder, img_name)
                        frame = cv2.imread(img_path)
                        if frame is not None:
                            out.write(frame)
                    out.release()
                    print(f"Compiled video saved to: {color_video_path}")
                else:
                    print(f"Neither 'Color.mp4' nor 'Color/' folder found in {trial_path}, skipping.")
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
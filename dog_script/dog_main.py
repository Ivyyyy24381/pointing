from detect_dog_skeleton import detect_dog, run_mediapipe_json
from dog_pose_visualize import pose_visualize
import os
import argparse
import matplotlib
from segmentation import SAM2VideoSegmenter
import shutil
import os

def segment_subject(folder_path):
    segmenter = SAM2VideoSegmenter(folder_path)
    segmenter.load_video_frames()
    segmenter.interactive_segmentation()
    segmenter.propagate_segmentation()
    segmenter.visualize_results()
    # Clean up temp JPEG folder if it was created
    if hasattr(segmenter, "tmp_jpeg_dir") and segmenter.tmp_jpeg_dir is not None:
        try:
            shutil.rmtree(segmenter.tmp_jpeg_dir)
            shutil.rmtree(os.path.join(folder_path, 'segmented_path'))
        except Exception as e:
            print(f"Warning: failed to remove temp JPEG dir {segmenter.tmp_jpeg_dir}: {e}")
    print("Segmented_Subject")


def process_dog(folder_path, side_view=False):
    for folder_name in sorted(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_full_path) or not folder_name.isdigit():
            continue
        video_path = os.path.join(folder_full_path, "Color")

        segment_subject(video_path)
        segmented_video_path = os.path.join(folder_full_path, 'masked_video.mp4')

        trial_name = folder_full_path.split('/')[-1]
        subject_name = folder_full_path.split('/')[-2]
        if 'CCD' in subject_name:
            dog = False
            print("tracking baby...")
            # use mediapipe to extract the baby skeleton
            run_mediapipe_json(segmented_video_path)
        else:
            detect_dog(segmented_video_path)
            print("tracking dog...")
            dog = True

        json_files = [f for f in os.listdir(folder_full_path) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON found in {folder_full_path}")
            continue

        json_path = os.path.join(folder_full_path, json_files[0])
        pose_visualize(json_path, side_view=side_view, dog = dog)


# ---- New function to concatenate processed results ----
import pandas as pd

def concatenate_processed_results(root_folder):
    all_data = []
    for trial_folder in sorted(os.listdir(root_folder)):
        trial_path = os.path.join(root_folder, trial_folder)
        if not os.path.isdir(trial_path) or not trial_folder.isdigit():
            continue
        csv_path = os.path.join(trial_path, 'processed_subject_result_table.csv')
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                df.insert(0, 'trial', trial_folder)
                dog_id = os.path.basename(os.path.normpath(root_folder))
                df.insert(0, 'dog', dog_id)
                all_data.append(df)
            except Exception as e:
                print(f"⚠️ Failed to read {csv_path}: {e}")
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        if 'frame_index' in combined_df.columns:
            combined_df = combined_df.sort_values(by='frame_index')
        output_csv = os.path.join(root_folder, f"{dog_id}_combined_result.csv")
        combined_df.to_csv(output_csv, index=False)
        print(f"✅ Saved combined results to {output_csv}")
    else:
        print("⚠️ No processed_subject_result_table.csv files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default='/home/xhe71/Desktop/dog_data/baby/CCD0384_PVPT_004E_side/', help="Path to the root dog video dataset directory")
    parser.add_argument("--side_view", action='store_true', help="Flag to indicate if this is the side view")
    args = parser.parse_args()
    # process_dog(os.path.expanduser(args.root_path), side_view=args.side_view)
    process_dog(os.path.expanduser(args.root_path), side_view=True)
    concatenate_processed_results(os.path.expanduser(args.root_path))
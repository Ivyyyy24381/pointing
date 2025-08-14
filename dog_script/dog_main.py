from detect_dog_skeleton import detect_dog, run_mediapipe_json
from dog_pose_visualize import pose_visualize
import os
import argparse
import matplotlib
from segmentation import SAM2VideoSegmenter
import shutil
import os
import torch
import gc
import cv2
import numpy as np
from pathlib import Path

# Standard camera intrinsics dimensions
STANDARD_WIDTH = 640
STANDARD_HEIGHT = 480

def standardize_images(color_dir, depth_dir):
    """
    Standardize all color and depth images to match camera intrinsics (640x480).
    Uses proper interpolation methods to preserve image quality and depth accuracy.
    """
    print(f"📐 Standardizing images to {STANDARD_WIDTH}x{STANDARD_HEIGHT} to match camera intrinsics...")
    
    current_w, current_h = None, None
    
    # Process color images
    color_files = [f for f in os.listdir(color_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if color_files:
        # Check current dimensions
        first_img_path = os.path.join(color_dir, color_files[0])
        first_img = cv2.imread(first_img_path)
        current_h, current_w = first_img.shape[:2]
        print(f"🔍 Current color dimensions: {current_w}x{current_h}")
        
        if current_w != STANDARD_WIDTH or current_h != STANDARD_HEIGHT:
            print(f"🔄 Resizing {len(color_files)} color images...")
            for filename in color_files:
                img_path = os.path.join(color_dir, filename)
                img = cv2.imread(img_path)
                
                # Use INTER_LANCZOS4 for high-quality color image interpolation
                resized_img = cv2.resize(img, (STANDARD_WIDTH, STANDARD_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(img_path, resized_img)
            print("✅ Color images standardized with high-quality interpolation")
        else:
            print("✅ Color images already at standard dimensions")
    
    # Process depth images if they exist
    if os.path.exists(depth_dir):
        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.raw')]
        if depth_files:
            print(f"🔄 Standardizing {len(depth_files)} depth images...")
            
            # Try to infer depth dimensions from first file
            first_depth = depth_files[0]
            depth_path = os.path.join(depth_dir, first_depth)
            file_size = os.path.getsize(depth_path)
            
            # Try common depth formats
            possible_dimensions = [
                (640, 480),   # Common RealSense resolution
                (848, 480),   # RealSense D435 depth resolution
                (640, 480),   # VGA resolution
                (424, 240),   # Kinect v2 depth resolution
            ]
            
            depth_w, depth_h = None, None
            for w, h in possible_dimensions:
                if file_size == w * h * 2:  # uint16 format
                    depth_w, depth_h = w, h
                    break
                    
            # Fallback to color dimensions if available
            if depth_w is None and current_w is not None:
                if file_size == current_w * current_h * 2:
                    depth_w, depth_h = current_w, current_h
                    
            if depth_w is None:
                print(f"⚠️ Cannot determine depth dimensions (file size: {file_size} bytes)")
                return
                
            print(f"🔍 Current depth dimensions: {depth_w}x{depth_h}")
            
            if depth_w != STANDARD_WIDTH or depth_h != STANDARD_HEIGHT:
                for filename in depth_files:
                    depth_path = os.path.join(depth_dir, filename)
                    
                    # Read raw depth data (uint16)
                    with open(depth_path, 'rb') as f:
                        depth_data = np.frombuffer(f.read(), dtype=np.uint16).reshape((depth_h, depth_w))
                    
                    # Use INTER_NEAREST for depth data to preserve depth values exactly
                    # This prevents interpolation artifacts in depth measurements
                    resized_depth = cv2.resize(depth_data, (STANDARD_WIDTH, STANDARD_HEIGHT), 
                                               interpolation=cv2.INTER_NEAREST)
                    
                    # Write back as raw
                    with open(depth_path, 'wb') as f:
                        f.write(resized_depth.astype(np.uint16).tobytes())
                        
                print("✅ Depth images standardized with nearest-neighbor interpolation")
            else:
                print("✅ Depth images already at standard dimensions")

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Print memory info
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🟢 GPU memory cleared: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print("🟢 GPU memory cleared (CPU mode)")


def segment_subject(folder_path):
    segmenter = SAM2VideoSegmenter(folder_path)
    # Reset model memory for new trial
    segmenter.reset_model_memory()
    segmenter.load_video_frames()
    segmenter.interactive_segmentation()
    segmenter.propagate_segmentation()
    segmenter.visualize_results()
    
    # Clear GPU memory before continuing
    segmenter.clear_gpu_memory()
    
    # Clean up temp JPEG folder if it was created
    if hasattr(segmenter, "tmp_jpeg_dir") and segmenter.tmp_jpeg_dir is not None:
        try:
            shutil.rmtree(segmenter.tmp_jpeg_dir)
            shutil.rmtree(os.path.join(folder_path, 'segmented_path'))
        except Exception as e:
            print(f"Warning: failed to remove temp JPEG dir {segmenter.tmp_jpeg_dir}: {e}")
    
    # Delete the segmenter to free memory
    del segmenter
    
    print("Segmented_Subject")


def process_dog(folder_path, side_view=False):
    for folder_name in sorted(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_full_path) or not folder_name.isdigit():
            continue
        video_path = os.path.join(folder_full_path, "Color")
        depth_path = os.path.join(folder_full_path, "Depth")
        
        # Standardize images to match camera intrinsics BEFORE segmentation
        standardize_images(video_path, depth_path)

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
        clear_gpu()
        
        # Find the skeleton JSON file
        json_files = [f for f in os.listdir(folder_full_path) if f.endswith('.json')]
        skeleton_files = [f for f in json_files if 'skeleton' in f]
        
        if skeleton_files:
            json_path = os.path.join(folder_full_path, skeleton_files[0])
        elif json_files:
            json_path = os.path.join(folder_full_path, json_files[0])
        else:
            print(f"No JSON found in {folder_full_path}")
            continue

        # Label targets for distance calculations (if not already done)
        target_file = os.path.join(folder_full_path, "target_coordinates.json")
        if not os.path.exists(target_file):
            print(f"🎯 No target coordinates found. Running target labeling for {folder_full_path}")
            from label_targets import label_single_trial
            label_single_trial(folder_full_path, force_relabel=False)
        else:
            print(f"✅ Target coordinates already exist: {target_file}")

        print(f"Using JSON file: {os.path.basename(json_path)}")
        pose_visualize(json_path, side_view=side_view, dog = dog)
        
        # Clear GPU memory after pose processing
        clear_gpu()


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
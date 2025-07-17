import sys
sys.path.append('./')
from GUI.bag_slice_gui import run_rs_convert, run_ffmpeg_convert
import os
import argparse
import sys
from pathlib import Path
import cv2
import os
import csv
import time
from natsort import natsorted
import pandas as pd
import shutil
def play_and_mark_frames(image_folder, output_csv="marked_frames.csv", frame_rate=100):
    """
    Plays a folder of images as a video, allows user to mark start ('s') and end ('e') frames,
    and saves them to a CSV.
    """
    # Get all image files and sort them naturally
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = natsorted(image_files)

    if not image_files:
        print("No images found in folder.")
        return

    print("Instructions:")
    print("- Press 's' to mark START of a segment")
    print("- Press 'e' to mark END of the segment")
    print("- Press 'q' to finish and save CSV")

    frame_index = 0
    marked_segments = []
    start_frame = None
    delay = int(1000 / frame_rate)  # milliseconds per frame

    while frame_index < len(image_files):
        image_path = os.path.join(image_folder, image_files[frame_index])
        frame = cv2.imread(image_path)
        # downside the image by 1/2
        if frame is not None:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        if frame is None:
            print(f"Failed to load {image_path}")
            frame_index += 1
            continue

        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame {frame_index}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Mark Frames", display_frame)
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('s'):
            start_frame = frame_index
            print(f"Start marked at frame {start_frame}")
        elif key == ord('e') and start_frame is not None:
            end_frame = frame_index
            print(f"End marked at frame {end_frame}")
            marked_segments.append((start_frame, end_frame))
            start_frame = None  # Reset for next segment
        elif key == ord('q'):
            break

        frame_index += 1

    cv2.destroyAllWindows()

    # Save to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start_seconds', 'end_seconds'])
        for segment in marked_segments:
            writer.writerow(segment)

    print(f"Saved {len(marked_segments)} segments to {output_csv}")

def copy_frame_range(color_dir, depth_dir, depth_color_dir, out_dir, start_frame, end_frame):
        os.makedirs(os.path.join(out_dir, "Color"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Depth"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "Depth_Color"), exist_ok=True)
        # os.makedirs(os.path.join(out_dir, "Raw"), exist_ok=True)

        start_frame = int(start_frame)
        end_frame = int(end_frame)

        print(f"Copying frames from {start_frame} to {end_frame - 1} into {out_dir}")

        for idx in range(start_frame, end_frame):
            color_src = os.path.join(color_dir, f"Color_{idx:06d}.png")
            depth_src = os.path.join(depth_dir, f"Depth_{idx:06d}.png")
            depth_color_src = os.path.join(depth_color_dir, f"Depth_Color_{idx:06d}.png")

            # RAW files (adjust extensions/names as needed)
            color_raw_src = os.path.join(color_dir, f"Color_{idx:06d}.raw")
            depth_raw_src = os.path.join(depth_dir, f"Depth_{idx:06d}.raw")
            depth_color_raw_src = os.path.join(depth_color_dir, f"Depth_Color_{idx:06d}.raw")

            # Destinations
            color_dst = os.path.join(out_dir, "Color", f"Color_{idx:06d}.png")
            depth_dst = os.path.join(out_dir, "Depth", f"Depth_{idx:06d}.png")
            depth_color_dst = os.path.join(out_dir, "Depth_Color", f"_Depth_Color_{idx:06d}.png")

            color_raw_dst = os.path.join(out_dir, "Color",  f"Color_{idx:06d}.raw")
            depth_raw_dst = os.path.join(out_dir, "Depth",  f"Depth_{idx:06d}.raw")
            depth_color_raw_dst = os.path.join(out_dir, "Depth_Color",f"Depth_Color_{idx:06d}.raw")

            # Copy images
            if os.path.exists(color_src):
                shutil.copy2(color_src, color_dst)
            else:
                print(f"Warning: Missing color frame {color_src}")
            if os.path.exists(depth_src):
                shutil.copy2(depth_src, depth_dst)
            else:
                print(f"Warning: Missing depth frame {depth_src}")
            if os.path.exists(depth_color_src):
                shutil.copy2(depth_color_src, depth_color_dst)
            else:
                print(f"Warning: Missing depth_color frame {depth_color_src}")

            # Copy raw files
            if os.path.exists(color_raw_src):
                shutil.copy2(color_raw_src, color_raw_dst)
            if os.path.exists(depth_raw_src):
                shutil.copy2(depth_raw_src, depth_raw_dst)
            if os.path.exists(depth_color_raw_src):
                shutil.copy2(depth_color_raw_src, depth_color_raw_dst)


# ask user to input directory
bag_folders = input("Enter the directory containing the rosbag folder files: ")
if not bag_folders:
    print("No directory provided. using default directory.")
    bag_folders = "/home/xhe71/Desktop/dog_data/baby"

# inside every subfolder of the provided directory, go in and search for rosbag
path_lists = os.listdir(bag_folders)

for path in sorted(path_lists):
    full_path = os.path.join(bag_folders, path)
    if os.path.isdir(full_path):
        rosbag_files = [f for f in os.listdir(full_path) if f.endswith('.bag')]
        if rosbag_files:
            rosbag_path = os.path.join(full_path, rosbag_files[0])
            color_dir = os.path.join(full_path, 'Color')
            depth_dir = os.path.join(full_path, 'Depth')
            depth_color_dir = os.path.join(full_path, 'Depth_Color')
            start_sec = 0
            end_sec = 1000000
            
            print(f"Processing rosbag file: {rosbag_path}")
            if not os.path.exists(os.path.join(full_path, 'Color')):
            
                recored_fps = run_rs_convert(rosbag_path, full_path,start_sec, end_sec)
                run_ffmpeg_convert(full_path, framerate = recored_fps)
            if not os.path.exists(os.path.join(full_path,'auto_split.csv')) or not os.path.exists(os.path.join(full_path, '5')):
                if not os.path.exists(os.path.join(full_path, 'auto_split.csv')):
                    print("No auto_split.csv found, marking frames...")
                    input("Press Enter to start marking frames...")
                    play_and_mark_frames(os.path.join(full_path, 'Color'), 
                    output_csv=os.path.join(full_path, 'auto_split.csv'))
            
                df = pd.read_csv(os.path.join(full_path, 'auto_split.csv'))
                # Step 2: Loop through CSV, slice frames
                for i, row in df.iterrows():
                    start_frame = int(row['start_seconds'])  # assuming seconds == frames
                    end_frame = int(row['end_seconds'])
                    index = i + 1
                    out_path = os.path.join(full_path, str(index))
                    os.makedirs(out_path, exist_ok=True)

                    copy_frame_range(color_dir, depth_dir, depth_color_dir, out_path, start_frame, end_frame)

                    run_ffmpeg_convert(out_path, framerate=6)
                    color_video_path = os.path.join(out_path, "Color.mp4")
                    depth_video_path = os.path.join(out_path, "Depth.mp4")
                
            
            print(f"Processed rosbag file: {rosbag_path}")
            
        else:
            print(f"No rosbag files found in {full_path}.")
    else:
        print(f"{full_path} is not a directory.")


print("finish extracting bag")

for filename in os.listdir(full_path):
    if filename.endswith(".bag"):
        file_path = os.path.join(full_path, filename)
        os.remove(file_path)
        print(f"Deleted: {file_path}")
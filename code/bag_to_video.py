import os
import argparse
from moviepy.editor import VideoFileClip, clips_array

def run_rs_convert(bag_filepath, output_prefix, start_sec, end_sec):
    color_dir = os.path.join(output_prefix, "Color/")
    depth_dir = os.path.join(output_prefix, "Depth/")
    
    # Ensure the output directories exist
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    command = f"rs-convert -i {bag_filepath} -p {color_dir} -r {depth_dir} -s {start_sec} -e {end_sec}"
    print(f"Running command: {command}")
    os.system(command)

def run_ffmpeg_convert(output_prefix, framerate=6):
    color_dir = os.path.join(output_prefix, "Color/")
    depth_dir = os.path.join(output_prefix, "Depth/")
    
    color_video_command = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{color_dir}_Color_*.png' -c:v libx264 -r 30 -pix_fmt yuv420p {output_prefix}_Color.mp4"
    depth_video_command = f"ffmpeg -framerate {framerate} -pattern_type glob -i '{color_dir}_Depth_*.png' -c:v libx264 -r 30 -pix_fmt yuv420p {output_prefix}_Depth.mp4"

    print(f"Running command: {color_video_command}")
    os.system(color_video_command)

    print(f"Running command: {depth_video_command}")
    os.system(depth_video_command)

def concat_videos(color_video_path, depth_video_path, output_path):
    video_top = VideoFileClip(color_video_path)
    video_bottom = VideoFileClip(depth_video_path)

    width = min(video_top.w, video_bottom.w)
    video_top = video_top.resize(width=width)
    video_bottom = video_bottom.resize(width=width)

    final_video = clips_array([[video_top], [video_bottom]])
    final_video.write_videofile(output_path)

def main():
    # Step 1: Parse the arguments
    parser = argparse.ArgumentParser(description='Process rosbag files and create videos.')
    parser.add_argument('--bag_filepath', type=str, required=True, help='Path to the rosbag file')
    parser.add_argument('--date', type=str, required=True, help='Date of the recording')
    parser.add_argument('--trial', type=str, required=True, help='Trial number')
    parser.add_argument('--start_sec', type=int, required=True, help='Start second for the video')
    parser.add_argument('--end_sec', type=int, required=True, help='End second for the video')
    args = parser.parse_args()

    # Step 1.5: Define the root directory and file paths
    root_directory = os.path.dirname(os.path.abspath(args.bag_filepath))
    output_prefix = os.path.join(root_directory, f"{args.date}_{args.trial}/")

    # Step 2: Run rs-convert to extract images from the rosbag
    run_rs_convert(args.bag_filepath, output_prefix, args.start_sec, args.end_sec)

    # Step 3: Convert the extracted images to color and depth videos using ffmpeg
    run_ffmpeg_convert(output_prefix)

    # Step 4: Concatenate the videos vertically (color on top, depth on bottom)
    color_video_path = f"{output_prefix}_Color.mp4"
    depth_video_path = f"{output_prefix}_Depth.mp4"
    output_video_path = os.path.join(root_directory, f"{args.date}_t{args.trial}_output.mp4")
    concat_videos(color_video_path, depth_video_path, output_video_path)

if __name__ == "__main__":
    main()

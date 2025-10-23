import os
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from natsort import natsorted
import copy
# from moviepy.editor import VideoFileClip, clips_array
import yaml
from tqdm import tqdm
import subprocess
import glob

def run_rs_convert_cli(bag_filepath, output_dir, start_sec=0, end_sec=10000):
    command = [
        "rs-convert",
        "-i", bag_filepath,
        "-p", os.path.join(output_dir, "Color/"),
        "-r", os.path.join(output_dir, "Depth/"),
        "-s", str(start_sec),
        "-e", str(end_sec),
    ]
    print(f"⚙️ Running rs-convert: {' '.join(command)}")
    subprocess.run(command, check=True)


def extract_rs_bag_metadata_to_yaml(bag_filepath, output_yaml_path):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_filepath, repeat_playback=False)
    profile = pipeline.start(config)

    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    duration = playback.get_duration()

    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()

    depth_fps = depth_profile.fps()
    color_fps = color_profile.fps()
    recorded_fps = min(depth_fps, color_fps)

    intr = depth_profile.get_intrinsics()

    data = {
        "bag_filepath": bag_filepath,
        "duration_seconds": round(duration.total_seconds(), 2),
        "depth_fps": depth_fps,
        "color_fps": color_fps,
        "recorded_fps": recorded_fps,
        "intrinsics": {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs)
        }
    }

    with open(output_yaml_path, 'w') as f:
        yaml.dump(data, f)

    pipeline.stop()
    print(f"✅ Metadata saved to: {output_yaml_path}")
    return data



def postprocess_rs_convert_output(output_dir, color_fps, depth_fps):
    color_dir = os.path.join(output_dir, "Color")
    depth_dir = os.path.join(output_dir, "Depth")

    color_files = sorted(glob.glob(os.path.join(color_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    print(f"🧹 Found {len(color_files)} color frames, {len(depth_files)} depth frames")

    if not color_files or not depth_files:
        print("❌ Missing one of the streams in rs-convert output")
        return

    # Determine which stream is faster
    if depth_fps >= color_fps:
        fast_stream = "depth"
        ratio = int(round(depth_fps / color_fps))
        fast_files = depth_files
        slow_files = color_files
    else:
        fast_stream = "color"
        ratio = int(round(color_fps / depth_fps))
        fast_files = color_files
        slow_files = depth_files

    print(f"🎯 Aligning frames: skipping 1 out of every {ratio} frames from {fast_stream}")

    frame_idx = 0
    for i in range(min(len(slow_files), len(fast_files) // ratio)):
        slow_path = slow_files[i]
        fast_path = fast_files[i * ratio]

        new_name = f"_{frame_idx:06}.png"

        if fast_stream == "depth":
            os.rename(slow_path, os.path.join(color_dir, f"Color_{new_name}"))
            os.rename(fast_path, os.path.join(depth_dir, f"Depth_{new_name}"))
        else:
            os.rename(fast_path, os.path.join(color_dir, f"Color_{new_name}"))
            os.rename(slow_path, os.path.join(depth_dir, f"Depth_{new_name}"))

        frame_idx += 1

    # Remove all unmatched files (leftovers)
    for f in glob.glob(os.path.join(color_dir, "*.png")):
        if "Color_" not in f:
            os.remove(f)
    for f in glob.glob(os.path.join(depth_dir, "*.png")):
        if "Depth_" not in f:
            os.remove(f)

    # Remove leftover metadata
    for meta_file in glob.glob(os.path.join(output_dir, "**", "*.txt"), recursive=True):
        os.remove(meta_file)

    print(f"✅ Aligned and renamed {frame_idx} frames")

def run_rs_convert(bag_filepath, output_prefix, start_sec=0, end_sec=10000):
    color_dir = os.path.join(output_prefix, "Color/")
    depth_dir = os.path.join(output_prefix, "Depth/")
    depth_color_dir = os.path.join(output_prefix, "Depth_Color/")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_color_dir, exist_ok=True)

    # Step 1: Extract and load metadata
    metadata_path = os.path.join(output_prefix, "rosbag_metadata.yaml")
    extract_rs_bag_metadata_to_yaml(bag_filepath, metadata_path)

    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)
        color_fps = metadata.get("color_fps", 30)
        depth_fps = metadata.get("depth_fps", 30)
        duration = metadata.get("duration_seconds", 10000)
    else:
        print("⚠️ No metadata file found. Defaulting to 30 FPS and 10000s.")
        color_fps = depth_fps = 30
        duration = 10000

    # if color_fps != depth_fps:
    #     run_rs_convert_cli(bag_filepath, output_prefix,  start_sec=0, end_sec=10000)
    #     postprocess_rs_convert_output(output_prefix, color_fps, depth_fps)
    #     return min(color_fps, depth_fps)
    
    # Estimate total number of saved frames
    fps = color_fps
    duration_range = min(duration, end_sec) - start_sec
    est_total_frames = int(duration_range * fps)
    pbar = tqdm(total=est_total_frames, desc="Extracting aligned frames")

    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_filepath, repeat_playback=False)
    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    colorizer = rs.colorizer()

    color_count = 0
    depth_count = 0
    frame_idx = 0
    skip_counter = 0

    try:
        frames = pipeline.wait_for_frames()
        start_time = frames.get_timestamp() / 1000.0

        while playback.current_status() != rs.playback_status.stopped:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                print("🎬 End of rosbag or timeout.")
                break

            timestamp_sec = frames.get_timestamp() / 1000.0 - start_time
            if timestamp_sec < start_sec:
                continue
            if timestamp_sec >= end_sec:
                print("✅ Done extracting all frames.")
                break

            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            color_valid = color_frame and color_frame.is_video_frame()
            depth_valid = depth_frame and depth_frame.is_video_frame()

            if not (color_valid or depth_valid):
                continue  # nothing to save
                
            if color_valid:
                color_count += 1
            if depth_valid:
                depth_count += 1

            # Only save if both are present
            if not (color_valid and depth_valid):
                print(f"⚠️ Skipped: missing {'color' if not color_valid else 'depth'} frame")
                continue

            # Save aligned frame pair
            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_colored = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            fname = f"{frame_idx:06}.png"
            cv2.imwrite(os.path.join(color_dir, f"Color_{fname}"), cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(depth_color_dir, f"Depth_Color_{fname}"), cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(depth_dir, f"Depth_{fname}"), cv2.cvtColor(depth_raw, cv2.COLOR_BGR2RGB))
            depth_raw.tofile(os.path.join(depth_dir, f"Depth_{frame_idx:06}.raw"))

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        pipeline.stop()

    return fps

def run_ffmpeg_convert(output_prefix, framerate=6, concat =True):
    import subprocess
    from natsort import natsorted

    frame_duration = 1 / framerate / 2  # e.g., 1/6 = 0.1667s

    color_dir = os.path.join(output_prefix, "Color/")
    depth_dir = os.path.join(output_prefix, "Depth/")
    depth_color_dir = os.path.join(output_prefix, "Depth_Color/")
    depth_color_output = os.path.join(output_prefix, "Depth_Color.mp4")
    color_output = os.path.join(output_prefix, "Color.mp4")
    depth_output = os.path.join(output_prefix, "Depth.mp4")

    def write_concat_list(image_list, list_path):
        with open(list_path, "w") as f:
            for filename in image_list[:-1]:  # All but last need duration
                f.write(f"file '{filename}'\n")
                f.write(f"duration {frame_duration:.4f}\n")
            # Write the last image without duration (FFmpeg requires this)
            if image_list:
                f.write(f"file '{image_list[-1]}'\n")

    # === COLOR VIDEO ===
    color_images = natsorted([os.path.join(color_dir, f) for f in os.listdir(color_dir) if f.endswith(".png")])
    color_list_path = os.path.join(output_prefix, "color_list.txt")
    write_concat_list(color_images, color_list_path)
    
    color_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", color_list_path, "-vsync", "vfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        color_output
    ]
    print("Running sorted color video export...")
    subprocess.run(color_cmd)

    # === DEPTH COLOR VIDEO ===
    depth_color_images = natsorted([os.path.join(depth_color_dir, f) for f in os.listdir(depth_color_dir) if f.endswith(".png")])
    depth_color_list_path = os.path.join(output_prefix, "depth_color_list.txt")
    write_concat_list(depth_color_images, depth_color_list_path)

    depth_color_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", depth_color_list_path, "-vsync", "vfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        depth_color_output
    ]
    print("Running sorted depth-color video export...")
    subprocess.run(depth_color_cmd)

    # === RAW DEPTH VIDEO ===
    depth_images = natsorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".png")])
    depth_list_path = os.path.join(output_prefix, "depth_list.txt")
    write_concat_list(depth_images, depth_list_path)

    depth_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", depth_list_path, "-vsync", "vfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        depth_output
    ]
    print("Running sorted depth video export...")
    subprocess.run(depth_cmd)


# def concat_videos(color_video_path, depth_video_path, output_path):
#     video_top = VideoFileClip(color_video_path)
#     video_bottom = VideoFileClip(depth_video_path)

#     width = min(video_top.w, video_bottom.w)
#     video_top = video_top.resize(width=width)
#     video_bottom = video_bottom.resize(width=width)

#     final_video = clips_array([[video_top], [video_bottom]])
#     final_video.write_videofile(output_path, audio = False, threads = 8)

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
    print(output_prefix)
    # Step 2: Run rs-convert to extract images from the rosbag
    intr, recorded_fps, duration, total_frames = run_rs_convert(args.bag_filepath, output_prefix, args.start_sec, args.end_sec)
    frate = total_frames * 2 / duration
    print(f"FPS->{recorded_fps}, {frate}")
    # Step 3: Convert the extracted images to color and depth videos using ffmpeg
    run_ffmpeg_convert(output_prefix, framerate = frate)

    # Step 4: Concatenate the videos vertically (color on top, depth on bottom)
    # color_video_path = f"{output_prefix}_Color.mp4"
    # depth_video_path = f"{output_prefix}_Depth.mp4"
    # depth_color_video_path = f"{output_prefix}_Depth_Color.mp4"
    # output_video_path = os.path.join(root_directory, f"{args.date}_t{args.trial}_output.mp4")
    # concat_videos(color_video_path, depth_color_video_path, output_video_path)

if __name__ == "__main__":
    main()

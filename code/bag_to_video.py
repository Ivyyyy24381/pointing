import os
import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
from natsort import natsorted
import copy
from moviepy.editor import VideoFileClip, clips_array

def run_rs_convert(bag_filepath, output_prefix, start_sec = 0, end_sec = 10000):
    color_dir = os.path.join(output_prefix, "Color/")
    depth_dir = os.path.join(output_prefix, "Depth/")
    depth_color_dir = os.path.join(output_prefix, "Depth_Color/")
    
    # Ensure the output directories exist
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(depth_color_dir, exist_ok=True)
    

    # command = f"rs-convert -i {bag_filepath} -p {color_dir} -r {depth_dir} -s {start_sec} -e {end_sec}"
    # print(f"Running command: {command}")
    # os.system(command)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_filepath, repeat_playback=False)
    profile = pipeline.start(config)
    device = profile.get_device()
    playback = device.as_playback()
    playback.set_real_time(False)
    duration = playback.get_duration()
    print(f"Bag duration: {duration.total_seconds():.2f} seconds")

    align = rs.align(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)
    intr = depth_profile.as_video_stream_profile().get_intrinsics()
    recorded_fps = depth_profile.as_video_stream_profile().fps()
    expected_total_frames = int(duration.total_seconds() * recorded_fps)
    print(f"Stream fps:{recorded_fps}")
    print("Camera Intrinsics:")
    print(f"Width: {intr.width}, Height: {intr.height}")
    print(f"Focal Length: fx={intr.fx}, fy={intr.fy}")
    print(f"Principal Point: cx={intr.ppx}, cy={intr.ppy}")
    print(f"Distortion Model: {intr.model}, Coefficients: {intr.coeffs}")


    frame_idx = 0
     # Create colorizer object
    colorizer = rs.colorizer()
    try:
        while True:
            frames = pipeline.wait_for_frames()

            if frame_idx < int(start_sec):
                frame_idx += 1
                continue
            if frames is None:
                break
            if frame_idx >= end_sec:
                print("✅ Done extracting all frames.")
                break

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            depth_raw_image = copy.deepcopy(depth_image)
            depth_image = copy.deepcopy(depth_color_image)
            depth_color_image = copy.deepcopy(depth_raw_image)
            depth_color_filename = os.path.join(depth_color_dir, f"_Depth_Color_{frame_idx:04}.png")
            
            cv2.imwrite(depth_color_filename,  cv2.cvtColor(depth_color_image, cv2.COLOR_BGR2RGB))
            depth_color_image.tofile(os.path.join(depth_color_dir, f"_Depth_Color_{frame_idx:04}.raw"))

            color_filename = os.path.join(color_dir, f"_Color_{frame_idx:04}.png")
            cv2.imwrite(color_filename,  cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            color_image.tofile(os.path.join(color_dir, f"_Color_{frame_idx:04}.raw"))

            depth_filename = os.path.join(depth_dir, f"_Depth_{frame_idx:04}.png")
            cv2.imwrite(depth_filename, cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))
            depth_image.tofile(os.path.join(depth_dir, f"_Depth_{frame_idx:04}.raw"))

            print(f" Saved frame {frame_idx} -> Time{frame_idx/recorded_fps/2/60} min")
            frame_idx += 1

    except RuntimeError as e:
        print("🎬 End of rosbag reached:", e)

    finally:
        pipeline.stop()
    total_frames = frame_idx
    return intr, recorded_fps, duration, total_frames

def run_ffmpeg_convert(output_prefix, framerate=6):
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


def concat_videos(color_video_path, depth_video_path, output_path):
    video_top = VideoFileClip(color_video_path)
    video_bottom = VideoFileClip(depth_video_path)

    width = min(video_top.w, video_bottom.w)
    video_top = video_top.resize(width=width)
    video_bottom = video_bottom.resize(width=width)

    final_video = clips_array([[video_top], [video_bottom]])
    final_video.write_videofile(output_path, audio = False, threads = 8)

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
    color_video_path = f"{output_prefix}_Color.mp4"
    depth_video_path = f"{output_prefix}_Depth.mp4"
    depth_color_video_path = f"{output_prefix}_Depth_Color.mp4"
    output_video_path = os.path.join(root_directory, f"{args.date}_t{args.trial}_output.mp4")
    concat_videos(color_video_path, depth_color_video_path, output_video_path)

if __name__ == "__main__":
    main()

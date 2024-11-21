import cv2
import os
import numpy as np
from collections import deque


def find_split_points_from_folders(color_folder, depth_folder):
    color_files = sorted([os.path.join(color_folder, f) for f in os.listdir(color_folder) if f.endswith(".png")])
    depth_files = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".png")])

    red_found = False
    frame_idx = 0
    split_points = []
    previous_frames = deque(maxlen=10)
    red_frame_count = 0
    green_frame_count = 0
    split_point_idx = []
    for color_file, depth_file in zip(color_files, depth_files):
        
        frame_color = cv2.imread(color_file)
        frame_depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)

        red_channel = np.mean(frame_color[:, :, 2])
        green_channel = np.mean(frame_color[:, :, 1])
        depth_channel = np.mean(frame_depth)
        previous_frames.append((red_channel, green_channel, depth_channel))
        avg_red = np.mean([p[0] for p in previous_frames])
        avg_green = np.mean([p[1] for p in previous_frames])
        red_threshold = avg_red + 20
        green_threshold = avg_green + 20

        if depth_channel < 5:
            if not red_found and red_channel > red_threshold and red_channel > green_channel:
                red_frame_count += 1
                if red_frame_count > 10:
                    red_found = True
                    red_frame_time = frame_idx
            elif red_found and green_channel > green_threshold and green_channel > red_channel:
                green_frame_count += 1
                if green_frame_count > 10:
                    green_frame_time = frame_idx
                    split_points.append((red_frame_time, green_frame_time))
                    red_found = False
            else:
                green_frame_count = 0
        frame_idx += 1

    return split_points


# Test the logic
if __name__ == "__main__":
    color_folder = "/Users/ivy/Downloads/Color_2/"
    depth_folder = "/Users/ivy/Downloads/Depth_2/"
    split_points = find_split_points_from_folders(color_folder, depth_folder)
    print("Split points:", split_points)
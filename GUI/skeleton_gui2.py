import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import sys
sys.path.append('visualize/')
from gesture_detection import PointingGestureDetector
import mediapipe as mp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import gridspec
from skspatial.objects import Plane, Line
from gesture_util import *
import cv2
import numpy as np

def match_fov(image, hfov_wide=87, hfov_narrow=69, vfov_wide=58, vfov_narrow=42):
    h_crop_ratio = np.tan(np.deg2rad(hfov_narrow / 2)) / np.tan(np.deg2rad(hfov_wide / 2))
    v_crop_ratio = np.tan(np.deg2rad(vfov_narrow / 2)) / np.tan(np.deg2rad(vfov_wide / 2))

    height, width = image.shape[:2]
    new_width = int(width * h_crop_ratio)
    new_height = int(height * v_crop_ratio)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    cropped_image = image[start_y:start_y + new_height, start_x:start_x + new_width]
    cropped_image = cv2.resize(cropped_image, (width, height))
    return cropped_image

class VideoPlayerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player with 3D Skeleton Visualization and Temporal Data")

        self.cap = None
        self.frame_count = 0
        self.current_frame = 0
        self.is_playing = False
        self.video_path = None
        self.downscale_factor = 0.5

        self.target_locations = []  # List to store target locations
        self.ground_plane = None
        self.wrist_history = []  # History of wrist positions for the entire video

        # Temporal data for plotting
        self.temporal_data = {
            "shoulder": {"x": [], "y": [], "z": []},
            "wrist": {"x": [], "y": [], "z": []},
            "elbow": {"x": [], "y": [], "z": []},
            "nose": {"x": [], "y": [], "z": []}
        }

        # Create GUI components
        self.load_button = tk.Button(root, text="Load Video", command=self.load_video)
        self.load_button.pack()

        self.video_canvas = tk.Canvas(root, bg="black", width=400, height=300)
        self.video_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.slider = tk.Scale(root, from_=0, to=100, orient="horizontal", command=self.update_frame, label="Frame")
        self.slider.pack(fill="x")

        self.play_button = tk.Button(root, text="Play", command=self.play_video)
        self.play_button.pack()

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_video)
        self.stop_button.pack()

        # Add target management buttons
        self.add_target_button = tk.Button(root, text="Add Target", command=self.add_target)
        self.add_target_button.pack()

        self.remove_target_button = tk.Button(root, text="Remove Target", command=self.remove_target)
        self.remove_target_button.pack()

        # Matplotlib figure for 3D skeleton visualization
        self.handedness = ""
        self.fig = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_3d.draw()
        self.canvas_3d.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Matplotlib figure for temporal data
        self.fig_temporal, self.ax_temporal = plt.subplots(3, 1, figsize=(3, 9), sharex=True)
        self.canvas_temporal = FigureCanvasTkAgg(self.fig_temporal, master=root)
        self.canvas_temporal.draw()
        self.canvas_temporal.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.gesture_detector = PointingGestureDetector()
        self.root.bind("<Configure>", self.resize_window)

    def load_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video File")
        if not self.video_path:
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.config(to=self.frame_count - 1)
        self.display_frame(0)

    def display_frame(self, frame_idx):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            height, width, _ = frame.shape
            color_orig = frame[0:int(height // 2), :, :]
            depth_orig = frame[int(height // 2):height, :, :]
            depth_orig = match_fov(depth_orig)

            # Process the video frame
            color_draw = self.gesture_detector.process_frame(color_orig)
            pose_results = self.gesture_detector.pose.process(color_orig)
            # draw landmark on depth image (do not delete)
            mp.solutions.drawing_utils.draw_landmarks(depth_orig, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
            landmarks_2d = pose_results.pose_landmarks
            if self.gesture_detector.pointing_hand_handedness != "":
                vectors_2d = self.gesture_detector.find_vectors(self.gesture_detector.pointing_hand_handedness, landmarks_2d)
                joints_2d = self.gesture_detector.find_joint_locations(self.gesture_detector.pointing_hand_handedness, landmarks_2d)
                self.gesture_detector.display_visualization(depth_orig, joints_2d, vectors_2d)

            if pose_results.pose_landmarks:
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_world_landmarks.landmark])
                vectors = self.gesture_detector.find_vectors(self.gesture_detector.pointing_hand_handedness, pose_results.pose_world_landmarks)
                joints = self.gesture_detector.find_joint_locations(self.gesture_detector.pointing_hand_handedness, pose_results.pose_world_landmarks)
                wrist_location = joints['wrist']

                # Add wrist position to history
                self.wrist_history.append([wrist_location.x, wrist_location.y, wrist_location.z])

                # Store temporal data for shoulder, wrist, elbow, and nose
                self.update_temporal_data(frame_idx, joints)

                # Evaluate pointing gestures in real-time
                eval_result = self.evaluate_gestures_live(self.ground_plane, wrist_location, vectors)

                # Update the 3D skeleton plot
                self.update_skeleton_3d(pose_results.pose_world_landmarks, eval_result['vector_intersections'], eval_result['closest_target'])

                # Update temporal data plots
                self.update_temporal_plots(frame_idx)

            color_frame = self.downscale_frame(color_orig, self.downscale_factor)
            depth_frame = self.downscale_frame(depth_orig, self.downscale_factor)

            stacked_frame = np.vstack((color_frame, depth_frame))

            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            resized_frame = self.resize_frame(stacked_frame, canvas_width, canvas_height)

            img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)

            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.video_canvas.image = img_tk
            self.slider.set(frame_idx)

    def downscale_frame(self, frame, scale_factor):
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return cv2.resize(frame, (new_width, new_height))

    def evaluate_gestures_live(self, ground_plane, wrist_location, vectors):
        """
        Evaluate pointing gestures for a single frame and calculate distances and intersections.
        """
        min_distance = float('inf')
        closest_target = None
        vector_intersections = {}
        if ground_plane is None:
            return {
            'closest_target': closest_target,
            'min_distance': min_distance,
            'vector_intersections': vector_intersections
        }
        for name, vec in vectors.items():
            if name == "wrist_to_index":
                continue
            direction = vec
            if direction is None:
                continue
            origin = [wrist_location.x, wrist_location.y, wrist_location.z]
            line = Line(point=origin, direction=direction)
            intersection = ground_plane.intersect_line(line)

            if intersection is not None:
                vector_intersections[name] = intersection
                    
        for target in self.target_locations:
            transformed_target_location = np.array(target)
            
            for name, vec in vectors.items():
                direction = vec
                if direction is None:
                    continue
                origin = [wrist_location.x, wrist_location.y, wrist_location.z]
                line = Line(point=origin, direction=direction)
                intersection = ground_plane.intersect_line(line)

                if intersection is not None:
                    vector_intersections[name] = intersection
                    distance_to_target = np.linalg.norm(intersection - transformed_target_location)
                    if distance_to_target < min_distance:
                        min_distance = distance_to_target
                        closest_target = target

        return {
            'closest_target': closest_target,
            'min_distance': min_distance,
            'vector_intersections': vector_intersections
        }

    def update_skeleton_3d(self, pose_landmarks, vector_intersections, closest_target):
        self.ax.cla()

        # Plot skeleton landmarks
        x = [lm.x for lm in pose_landmarks.landmark]
        y = [lm.y for lm in pose_landmarks.landmark]
        z = [lm.z for lm in pose_landmarks.landmark]

        self.ax.scatter(x, y, z, c='blue', label='Landmarks', s=50)
        mp_pose = mp.solutions.pose
        connections = [
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
        ]
    
        if self.gesture_detector.pointing_hand_handedness == 'Left':
            i = mp_pose.PoseLandmark.LEFT_WRIST
        else:
            i = mp_pose.PoseLandmark.RIGHT_WRIST
            
        for conn in connections:
            self.ax.plot([x[conn[0]], x[conn[1]]],
                         [y[conn[0]], y[conn[1]]],
                         [z[conn[0]], z[conn[1]]], c='b')

        # Plot ground plane
        self.ground_plane = find_ground_plane(pose_landmarks.landmark)
        self.ground_plane.plot_3d(self.ax, alpha=0.2)

        # Plot vector-ground intersections
        for name, intersection in vector_intersections.items():
            self.ax.scatter(intersection[0], intersection[1], intersection[2], marker='o', label=f'{name} intersection')
            self.ax.plot([x[i], intersection[0]], [y[i], intersection[1]], [z[i], intersection[2]], color='yellow', linestyle='-')

        # Plot pointing vectors and targets
        for target in self.target_locations:
            self.ax.scatter(target[0], target[1], target[2], c='red', s=100, label='Target', alpha=0.9)
            self.ax.plot([x[0], target[0]], [y[0], target[1]], [z[0], target[2]], color='orange', linestyle='--')

        # Plot wrist motion history
        wrist_history_np = np.array(self.wrist_history)
        if len(wrist_history_np) > 0:
            self.ax.plot(wrist_history_np[:, 0], wrist_history_np[:, 1], wrist_history_np[:, 2], c='cyan', linestyle='--', label="Wrist Trajectory")

        self.ax.set_xlabel('X Axis[m]')
        self.ax.set_ylabel('Y Axis[m]')
        self.ax.set_zlabel('Z Axis[m]')
        self.ax.legend(fontsize="x-small")
        self.ax.view_init(elev=-45, azim=-90, roll=0)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)

        self.canvas_3d.draw()

    def update_temporal_data(self, frame_idx, joints):
        """
        Store the current frame's x, y, z values for the shoulder, wrist, elbow, and nose.
        """
        for part in ["shoulder", "wrist", "elbow", "nose"]:
            self.temporal_data[part]["x"].append([frame_idx, getattr(joints[part], "x")])
            self.temporal_data[part]["y"].append([frame_idx, getattr(joints[part], "y")])
            self.temporal_data[part]["z"].append([frame_idx, getattr(joints[part], "z")])

    def update_temporal_plots(self, frame_idx):
        """
        Update the temporal data plots (x, y, z values over frames) with the current frame highlighted.
        """
        parts = ["shoulder", "wrist", "elbow", "nose"]
        labels = ["Shoulder", "Wrist", "Elbow", "Nose"]

        for i, axis in enumerate(["x", "y", "z"]):
            self.ax_temporal[i].cla()
            for j, part in enumerate(parts):
                data = np.array(self.temporal_data[part][axis])
                if axis == "x":
                    self.ax_temporal[i].scatter(data[:, 0],data[:, 1], label=f'{labels[j]} {axis.upper()}', s = 2)
                else:
                    self.ax_temporal[i].scatter(data[:, 0],data[:, 1], s = 2)
                self.ax_temporal[i].axvline(frame_idx, color='r', linestyle='--')  # Highlight current frame
            self.ax_temporal[i].legend(fontsize="x-small")

        self.ax_temporal[2].set_xlabel("Frame Number")
        self.ax_temporal[0].set_ylabel("X Axis[m]")
        self.ax_temporal[1].set_ylabel("Y Axis[m]")
        self.ax_temporal[2].set_ylabel("Z Axis[m]")
        self.ax_temporal[2].set_title("joint location[m]")

        self.canvas_temporal.draw()

    def update_frame(self, value):
        frame_idx = int(value)
        self.display_frame(frame_idx)
        self.current_frame = frame_idx

    def play_video(self):
        if not self.cap or self.is_playing:
            return
        self.is_playing = True
        threading.Thread(target=self.play_video_thread).start()

    def play_video_thread(self):
        while self.is_playing and self.current_frame < self.frame_count:
            self.display_frame(self.current_frame)
            self.current_frame += 1
            time.sleep(0.03)
            self.root.update_idletasks()
        self.is_playing = False

    def stop_video(self):
        self.is_playing = False

    def resize_frame(self, frame, target_width, target_height):
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(aspect_ratio * target_height)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        return cv2.resize(frame, (new_width, new_height))

    def resize_window(self, event):
        if self.cap:
            self.display_frame(self.current_frame)

    def add_target(self):
        x = simpledialog.askfloat("Input", "Enter X coordinate of target:")
        y = simpledialog.askfloat("Input", "Enter Y coordinate of target:")
        z = simpledialog.askfloat("Input", "Enter Z coordinate of target:")
        if x is not None and y is not None and z is not None:
            self.target_locations.append([x, y, z])
            self.update_skeleton_3d(self.gesture_detector.pose.pose_world_landmarks)

    def remove_target(self):
        if self.target_locations:
            self.target_locations.pop()
            self.update_skeleton_3d(self.gesture_detector.pose.pose_world_landmarks)

# Main application
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoPlayerGUI(root)
    root.mainloop()
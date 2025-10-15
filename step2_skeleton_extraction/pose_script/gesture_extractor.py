import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import argparse
import pandas as pd
import os
import glob
from mediapipe.framework.formats.landmark_pb2 import Landmark
import json
from gesture_util import *
import copy
import pyrealsense2 as rs
# === Helper function: cartesian_to_spherical ===
def cartesian_to_spherical(vector):
    x, y, z = vector
    r = np.linalg.norm(vector)
    theta = np.arctan2(y, x)  # azimuth
    phi = np.arccos(z / r) if r != 0 else 0  # elevation
    return theta, phi

# fx = 3629.5913404959015
# fy = 3629.5913404959015
# px = 2104.0
# py = 1560.0


def find_realsense_intrinsics(bag_path):
    # === Load the bag file ===
    cfg = rs.config()
    cfg.enable_device_from_file(bag_path)

    pipeline = rs.pipeline()
    profile = pipeline.start(cfg)

    # === Wait for frames to arrive ===
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # === Get stream profile ===
    color_stream = profile.get_stream(rs.stream.color)
    video_profile = color_stream.as_video_stream_profile()

    # === Extract intrinsics ===
    intr = video_profile.get_intrinsics()

    print("Width:", intr.width)
    print("Height:", intr.height)
    print("PPX (cx):", intr.ppx)
    print("PPY (cy):", intr.ppy)
    print("FX:", intr.fx)
    print("FY:", intr.fy)
    print("Distortion Model:", intr.model)
    print("Coefficients:", intr.coeffs)

    # === Stop the pipeline ===
    pipeline.stop()

    return intr


class PointingGestureDetector:
    __ELBOW_WRIST_COLOR = (13, 204, 255)
    __SHOULDER_WRIST_COLOR = (38, 115, 255)
    __EYE_WRIST_COLOR = (113, 242, 189)
    __NOSE_WRIST_COLOR = (105, 38, 191)
    __WRIST_INDEX_COLOR = (255, 98, 41)

    __INDEX_COLOR = (255, 234, 14)
    __GAZE_COLOR = (122, 110, 84)
    __POINTING_CONF_THRESHOLD = 0.4

    

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.5)
        self.pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence = 0.5,  model_complexity = 2)
        self.human_presence = False
        self.pointing = False
        self.frame = 0
        self.gesture_start_time = time.time()
        self.previous_time = time.time() 
        self.gesture_duration = 0.0
        self.pointing_hand_handedness = ''
        self.pointing_confidence = 0
        # 2d data
        self.vectors = None 
        self.vectors_conf = None
        self.origin = None
        self.joints = None
        self.gesture_conf = 0
        
        # Initialize other attributes
        self.previous_pointing = False
        self.previous_pointing_confidence = 0
        self.previous_gesture_conf = 0
        self.previous_pointing_hand_handedness = None
        self.previous_left_arm_points = None
        self.previous_right_arm_points = None
        self.previous_gray = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3, 
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01))
        
        # Best frame storage
        self.best_frame_color = None
        self.best_frame_depth = None
        self.best_frame_transformation = None
        self.best_3d_cone = None
        

        # Data storage (using pandas DataFrame)
        self.data = pd.DataFrame(columns=[
            'frame', 'gesture_duration', 'pointing_arm',
            'eye_to_wrist','eye_to_wrist_conf', 'shoulder_to_wrist', 'shoulder_to_wrist_conf', 'elbow_to_wrist','elbow_to_wrist_conf', 'nose_to_wrist','nose_to_wrist_conf','wrist_to_index','wrist_to_conf','index_finger',
            'wrist_location', 'landmarks', 'landmarks_3d'
        ])

        
    def clear_values(self):
        self.pointing = False
        self.gesture_start_time = time.time()
        self.previous_time = time.time() 
        self.gesture_duration = 0.0
        self.pointing_confidence = 0
        self.previous_pointing_confidence = 0
        self.previous_pointing_hand_handedness = None
        self.previous_gesture_conf = 0.0
        self.previous_gray = None
        self.human_presence = False
        self.best_frame_color = None
        self.best_frame_depth = None
        self.best_frame_transformation = None
        self.best_3d_cone = None
        

    
    


    def update_optimal_vector(self, image, depth_image=None, transformation_matrix=None):
        """
        Export the optimal vector and its confidence to JSON, and calculate distance to predefined targets.
        """
        self.best_frame_color = copy.deepcopy(image)
        self.best_frame_depth = copy.deepcopy(depth_image)
        self.best_frame_transformation = transformation_matrix

        # Define wrist and gaze_to_wrist before using them
        wrist = self.joints["wrist"]
        gaze_to_wrist = self.vectors["gaze"]

        shoulder_to_wrist = self.vectors['shoulder_to_wrist']
        elbow_to_wrist = self.vectors['elbow_to_wrist']
        nose_to_wrist = self.vectors['nose_to_wrist']
        eye_to_wrist = self.vectors['eye_to_wrist']

        t1, t2, t3, t4 = [
            np.array([-0.877, 0.889, 2.708]),
            np.array([-0.183, 0.763, 2.879]),
            np.array([0.595, 0.709, 2.828]),
            np.array([1.257, 0.817, 2.652])
        ]

        angle_t1_list = [
            self.angle_to_target(wrist, eye_to_wrist, t1),
            self.angle_to_target(wrist, shoulder_to_wrist, t1),
            self.angle_to_target(wrist, elbow_to_wrist, t1),
            self.angle_to_target(wrist, nose_to_wrist, t1),
            self.angle_to_target(wrist, gaze_to_wrist, t1),
        ]
        angle_eye_to_wrist_t1 = angle_t1_list[0]
        angle_shoulder_to_wrist_t1 = angle_t1_list[1]
        angle_elbow_to_wrist_t1 = angle_t1_list[2]
        angle_nose_to_wrist_t1 = angle_t1_list[3]
        angle_gaze_t1 = angle_t1_list[4]

        angle_t2_list = [
            self.angle_to_target(wrist, eye_to_wrist, t2),
            self.angle_to_target(wrist, shoulder_to_wrist, t2),
            self.angle_to_target(wrist, elbow_to_wrist, t2),
            self.angle_to_target(wrist, nose_to_wrist, t2),
            self.angle_to_target(wrist, gaze_to_wrist, t2),
        ]
        angle_eye_to_wrist_t2 = angle_t2_list[0]
        angle_shoulder_to_wrist_t2 = angle_t2_list[1]
        angle_elbow_to_wrist_t2 = angle_t2_list[2]
        angle_nose_to_wrist_t2 = angle_t2_list[3]
        angle_gaze_t2 = angle_t2_list[4]

        angle_t3_list = [
            self.angle_to_target(wrist, eye_to_wrist, t3),
            self.angle_to_target(wrist, shoulder_to_wrist, t3),
            self.angle_to_target(wrist, elbow_to_wrist, t3),
            self.angle_to_target(wrist, nose_to_wrist, t3),
            self.angle_to_target(wrist, gaze_to_wrist, t3),
        ]
        angle_eye_to_wrist_t3 = angle_t3_list[0]
        angle_shoulder_to_wrist_t3 = angle_t3_list[1]
        angle_elbow_to_wrist_t3 = angle_t3_list[2]
        angle_nose_to_wrist_t3 = angle_t3_list[3]
        angle_gaze_t3 = angle_t3_list[4]

        angle_t4_list = [
            self.angle_to_target(wrist, eye_to_wrist, t4),
            self.angle_to_target(wrist, shoulder_to_wrist, t4),
            self.angle_to_target(wrist, elbow_to_wrist, t4),
            self.angle_to_target(wrist, nose_to_wrist, t4),
            self.angle_to_target(wrist, gaze_to_wrist, t4),
        ]
        angle_eye_to_wrist_t4 = angle_t4_list[0]
        angle_shoulder_to_wrist_t4 = angle_t4_list[1]
        angle_elbow_to_wrist_t4 = angle_t4_list[2]
        angle_nose_to_wrist_t4 = angle_t4_list[3]
        angle_gaze_t4 = angle_t4_list[4]

        origin = None
        if self.origin:
            origin = [self.origin.x, self.origin.y, self.origin.z]

        # Use angle_shoulder_to_wrist_t1 as opening angle for legacy
        angle = angle_shoulder_to_wrist_t1

        # Compute spherical angles for each vector to each target
        vector_data = {
            'pointing_arm': self.pointing_hand_handedness,
            'pointing_vector_origin': origin,
            'pointing_vector_dir': (shoulder_to_wrist).tolist(),
            'pointing_classification_confidence': self.pointing_confidence,
            'pointing_vector_conf': self.gesture_conf,
            'pointing_vector_opening_angle': np.rad2deg(angle),
            # Store extracted angles for each vector and target
            'angle_eye_to_wrist_t1': angle_eye_to_wrist_t1,
            'angle_shoulder_to_wrist_t1': angle_shoulder_to_wrist_t1,
            'angle_elbow_to_wrist_t1': angle_elbow_to_wrist_t1,
            'angle_nose_to_wrist_t1': angle_nose_to_wrist_t1,
            'angle_gaze_t1': angle_gaze_t1,
            'angle_eye_to_wrist_t2': angle_eye_to_wrist_t2,
            'angle_shoulder_to_wrist_t2': angle_shoulder_to_wrist_t2,
            'angle_elbow_to_wrist_t2': angle_elbow_to_wrist_t2,
            'angle_nose_to_wrist_t2': angle_nose_to_wrist_t2,
            'angle_gaze_t2': angle_gaze_t2,
            'angle_eye_to_wrist_t3': angle_eye_to_wrist_t3,
            'angle_shoulder_to_wrist_t3': angle_shoulder_to_wrist_t3,
            'angle_elbow_to_wrist_t3': angle_elbow_to_wrist_t3,
            'angle_nose_to_wrist_t3': angle_nose_to_wrist_t3,
            'angle_gaze_t3': angle_gaze_t3,
            'angle_eye_to_wrist_t4': angle_eye_to_wrist_t4,
            'angle_shoulder_to_wrist_t4': angle_shoulder_to_wrist_t4,
            'angle_elbow_to_wrist_t4': angle_elbow_to_wrist_t4,
            'angle_nose_to_wrist_t4': angle_nose_to_wrist_t4,
            'angle_gaze_t4': angle_gaze_t4,
        }

        # Calculate spherical angles for each vector to each target
        vector_types = [
            ("eye_to_wrist", eye_to_wrist),
            ("shoulder_to_wrist", shoulder_to_wrist),
            ("elbow_to_wrist", elbow_to_wrist),
            ("nose_to_wrist", nose_to_wrist),
            ("gaze_to_wrist", gaze_to_wrist),
        ]
        targets = [("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4)]
        for vname, vvec in vector_types:
            for tname, tvec in targets:
                target_vector = tvec - wrist
                theta, phi = cartesian_to_spherical(target_vector)
                vector_data[f"theta_{vname}_{tname}"] = theta
                vector_data[f"phi_{vname}_{tname}"] = phi

        # === Gaze-to-Target Distance Calculation ===
        if origin is not None and shoulder_to_wrist is not None:
            origin_np = np.array(origin)
            direction = np.array(shoulder_to_wrist)
            direction = direction / np.linalg.norm(direction)

            self.targets_3d = [
                np.array([-0.877, 0.889, 2.708]),
                np.array([-0.183, 0.763, 2.879]),
                np.array([0.595, 0.709, 2.828]),
                np.array([1.257, 0.817, 2.652])
            ]

            distances = []
            for i, target in enumerate(self.targets_3d):
                to_target = target - origin_np
                projection_length = np.dot(to_target, direction)
                projected_point = origin_np + projection_length * direction
                distance = np.linalg.norm(projected_point - target)
                distances.append((i + 1, distance, target.tolist()))

            vector_data['target_distances'] = [{
                'target_id': tid,
                'distance': float(np.round(dist, 3)),
                'target_position': pos
            } for tid, dist, pos in distances]

            print("Distances to targets:", vector_data['target_distances'])

        self.best_3d_cone = vector_data

        # Save output
        write_json_locked(vector_data, ".tmp/gesture_vector.json")
        cv2.imwrite(".tmp/gesture_color.png", self.best_frame_color)
        if depth_image is not None:
            cv2.imwrite(".tmp/gesture_depth.png", self.best_frame_depth)
        if transformation_matrix is not None:
            write_json_locked(self.best_frame_transformation, ".tmp/gesture_transformation.json")

        print("----SAVING GESTURE IMAGES -------")
        print("updating vector output, opening angle is", np.rad2deg(angle))
        print("----waiting for new gesture-----")


    def save_gesture_data(self, frame_num, gesture_duration, pointing_arm, vectors, vectors_conf, wrist_location, landmarks_2d, landmarks_3d):
        """
        Save the detected pointing gesture data into the dataframe.
        Args:
            gesture_duration: Duration of the gesture.
            pointing_arm: The arm used for pointing (left or right).
            vectors: A dictionary of pointing vectors.
            wrist_location: The 3D location of the wrist.
        """
        import time
        # Add a new row to the dataframe with all the relevant data
        if vectors is not None and vectors_conf is not None:
            # Compute all 16 angles for the 4 targets and 4 vectors
            t1 = np.array([-0.877, 0.889, 2.708])
            t2 = np.array([-0.183, 0.763, 2.879])
            t3 = np.array([0.595, 0.709, 2.828])
            t4 = np.array([1.257, 0.817, 2.652])
            # Compute angles for each vector and target
            angle_eye_to_wrist_t1 = self.angle_to_target(wrist_location, vectors["eye_to_wrist"], t1)
            angle_shoulder_to_wrist_t1 = self.angle_to_target(wrist_location, vectors["shoulder_to_wrist"], t1)
            angle_elbow_to_wrist_t1 = self.angle_to_target(wrist_location, vectors["elbow_to_wrist"], t1)
            angle_nose_to_wrist_t1 = self.angle_to_target(wrist_location, vectors["nose_to_wrist"], t1)
            angle_gaze_t1 = self.angle_to_target(wrist_location, vectors["gaze"], t1)

            angle_eye_to_wrist_t2 = self.angle_to_target(wrist_location, vectors["eye_to_wrist"], t2)
            angle_shoulder_to_wrist_t2 = self.angle_to_target(wrist_location, vectors["shoulder_to_wrist"], t2)
            angle_elbow_to_wrist_t2 = self.angle_to_target(wrist_location, vectors["elbow_to_wrist"], t2)
            angle_nose_to_wrist_t2 = self.angle_to_target(wrist_location, vectors["nose_to_wrist"], t2)
            angle_gaze_t2 = self.angle_to_target(wrist_location, vectors["gaze"], t2)

            angle_eye_to_wrist_t3 = self.angle_to_target(wrist_location, vectors["eye_to_wrist"], t3)
            angle_shoulder_to_wrist_t3 = self.angle_to_target(wrist_location, vectors["shoulder_to_wrist"], t3)
            angle_elbow_to_wrist_t3 = self.angle_to_target(wrist_location, vectors["elbow_to_wrist"], t3)
            angle_nose_to_wrist_t3 = self.angle_to_target(wrist_location, vectors["nose_to_wrist"], t3)
            angle_gaze_t3 = self.angle_to_target(wrist_location, vectors["gaze"], t3)

            angle_eye_to_wrist_t4 = self.angle_to_target(wrist_location, vectors["eye_to_wrist"], t4)
            angle_shoulder_to_wrist_t4 = self.angle_to_target(wrist_location, vectors["shoulder_to_wrist"], t4)
            angle_elbow_to_wrist_t4 = self.angle_to_target(wrist_location, vectors["elbow_to_wrist"], t4)
            angle_nose_to_wrist_t4 = self.angle_to_target(wrist_location, vectors["nose_to_wrist"], t4)
            angle_gaze_t4 = self.angle_to_target(wrist_location, vectors["gaze"], t4)

            # Compute spherical (theta, phi) for each type and target
            vector_types = [
                ("eye_to_wrist", vectors["eye_to_wrist"]),
                ("shoulder_to_wrist", vectors["shoulder_to_wrist"]),
                ("elbow_to_wrist", vectors["elbow_to_wrist"]),
                ("nose_to_wrist", vectors["nose_to_wrist"]),
                ("gaze_to_wrist", vectors["gaze"]),
            ]
            targets = [("t1", t1), ("t2", t2), ("t3", t3), ("t4", t4)]
            sph_dict = {}
            for vname, vvec in vector_types:
                for tname, tvec in targets:
                    wrist_array = np.array([wrist_location.x, wrist_location.y, wrist_location.z])
                    target_vector = tvec - wrist_array
                    theta, phi = cartesian_to_spherical(target_vector)
                    sph_dict[f"theta_{vname}_{tname}"] = [theta]
                    sph_dict[f"phi_{vname}_{tname}"] = [phi]

            new_row = {
                'frame': frame_num,
                'time': time.time(),
                'gesture_duration': gesture_duration,
                'pointing_confidence': self.pointing_confidence,
                'pointing_arm': pointing_arm,
                'eye_to_wrist': [vectors['eye_to_wrist']],
                'eye_to_wrist_conf': [vectors_conf['eye_to_wrist']],
                'angle_eye_to_wrist_t1': [angle_eye_to_wrist_t1],
                'angle_eye_to_wrist_t2': [angle_eye_to_wrist_t2],
                'angle_eye_to_wrist_t3': [angle_eye_to_wrist_t3],
                'angle_eye_to_wrist_t4': [angle_eye_to_wrist_t4],
                'shoulder_to_wrist': [vectors['shoulder_to_wrist']],
                'shoulder_to_wrist_conf': [vectors_conf['shoulder_to_wrist']],
                'angle_shoulder_to_wrist_t1': [angle_shoulder_to_wrist_t1],
                'angle_shoulder_to_wrist_t2': [angle_shoulder_to_wrist_t2],
                'angle_shoulder_to_wrist_t3': [angle_shoulder_to_wrist_t3],
                'angle_shoulder_to_wrist_t4': [angle_shoulder_to_wrist_t4],
                'elbow_to_wrist': [vectors['elbow_to_wrist']],
                'elbow_to_wrist_conf': [vectors_conf['elbow_to_wrist']],
                'angle_elbow_to_wrist_t1': [angle_elbow_to_wrist_t1],
                'angle_elbow_to_wrist_t2': [angle_elbow_to_wrist_t2],
                'angle_elbow_to_wrist_t3': [angle_elbow_to_wrist_t3],
                'angle_elbow_to_wrist_t4': [angle_elbow_to_wrist_t4],
                'nose_to_wrist': [vectors['nose_to_wrist']],
                'nose_to_wrist_conf': [vectors_conf['nose_to_wrist']],
                'angle_nose_to_wrist_t1': [angle_nose_to_wrist_t1],
                'angle_nose_to_wrist_t2': [angle_nose_to_wrist_t2],
                'angle_nose_to_wrist_t3': [angle_nose_to_wrist_t3],
                'angle_nose_to_wrist_t4': [angle_nose_to_wrist_t4],
                'gaze': [vectors['gaze']],
                'gaze_conf': [vectors_conf['gaze']],
                'angle_gaze_t1': [angle_gaze_t1],
                'angle_gaze_t2': [angle_gaze_t2],
                'angle_gaze_t3': [angle_gaze_t3],
                'angle_gaze_t4': [angle_gaze_t4],
                'wrist_location': wrist_location,
                'landmarks': landmarks_2d,
                'landmarks_3d': landmarks_3d
            }
            # Add spherical values for all vector/target types
            new_row.update(sph_dict)
        else:
            new_row = {
                'frame': frame_num,
                'time': time.time(),
                'gesture_duration': None,
                'pointing_confidence': self.pointing_confidence,
                'pointing_arm': None,
                'eye_to_wrist': [None],
                'eye_to_wrist_conf': [None],
                'angle_eye_to_wrist_t1': [None],
                'angle_eye_to_wrist_t2': [None],
                'angle_eye_to_wrist_t3': [None],
                'angle_eye_to_wrist_t4': [None],
                'shoulder_to_wrist': [None],
                'shoulder_to_wrist_conf': [None],
                'angle_shoulder_to_wrist_t1': [None],
                'angle_shoulder_to_wrist_t2': [None],
                'angle_shoulder_to_wrist_t3': [None],
                'angle_shoulder_to_wrist_t4': [None],
                'elbow_to_wrist': [None],
                'elbow_to_wrist_conf': [None],
                'angle_elbow_to_wrist_t1': [None],
                'angle_elbow_to_wrist_t2': [None],
                'angle_elbow_to_wrist_t3': [None],
                'angle_elbow_to_wrist_t4': [None],
                'nose_to_wrist': [None],
                'nose_to_wrist_conf': [None],
                'angle_nose_to_wrist_t1': [None],
                'angle_nose_to_wrist_t2': [None],
                'angle_nose_to_wrist_t3': [None],
                'angle_nose_to_wrist_t4': [None],
                'gaze': [None],
                'gaze_conf': [None],
                'angle_gaze_t1': [None],
                'angle_gaze_t2': [None],
                'angle_gaze_t3': [None],
                'angle_gaze_t4': [None],
                'wrist_location': [None],
                'landmarks': landmarks_2d,
                'landmarks_3d': landmarks_3d
            }
        new_row = pd.DataFrame(new_row)
        self.data = pd.concat([self.data, new_row], ignore_index=True)

    def initialize_optical_flow_points(self, landmarks, w, h):
        """
        Set initial points for left and right arms based on landmarks.
        """
        self.previous_left_arm_points = np.array([
            [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x * w,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y * h],
            [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x * w,
             landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST].y * h]
        ], dtype=np.float32)

        self.previous_right_arm_points = np.array([
            [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x * w,
             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].y * h],
            [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * w,
             landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].y * h]
        ], dtype=np.float32)

    def detect_arm_movement(self, image, current_gray, landmarks):
        """
        Calculate movement for each arm using optical flow and determine which arm moves more.
        """
        if self.previous_gray is None:
            # First frame initialization
            self.previous_gray = current_gray
            h, w = current_gray.shape
            self.initialize_optical_flow_points(landmarks, w, h)
            return None  # Not enough data to determine movement yet

        # Calculate optical flow for left and right arms
        left_arm_points, st, err = cv2.calcOpticalFlowPyrLK(self.previous_gray, current_gray, self.previous_left_arm_points, None, **self.lk_params)
        right_arm_points, st, err = cv2.calcOpticalFlowPyrLK(self.previous_gray, current_gray, self.previous_right_arm_points, None, **self.lk_params)
        
        # Visualize the movement for each point in both arms
        for p, p_prev in zip(left_arm_points, self.previous_left_arm_points):
            start_point = (int(p_prev[0]), int(p_prev[1]))
            end_point = (int(p[0]), int(p[1]))
            # cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 2)
            # cv2.circle(image, end_point, 3, (0, 255, 0), -1)
            
        for p, p_prev in zip(right_arm_points, self.previous_right_arm_points):
            start_point = (int(p_prev[0]), int(p_prev[1]))
            end_point = (int(p[0]), int(p[1]))
            # cv2.arrowedLine(image, start_point, end_point, (255, 0, 0), 2)
            # cv2.circle(image, end_point, 3, (255, 0, 0), -1)
        
        # Calculate total movement for each arm
        left_movement = np.mean(np.linalg.norm(left_arm_points - self.previous_left_arm_points, axis=1))
        right_movement = np.mean(np.linalg.norm(right_arm_points - self.previous_right_arm_points, axis=1))
        print("-------")
        print(np.round(left_movement, 3), np.round(right_movement,3))
        # Update previous points and previous gray frame
        self.previous_left_arm_points, self.previous_right_arm_points = left_arm_points, right_arm_points
        self.previous_gray = current_gray

        # Determine which arm moved more
        if left_movement > right_movement:
            return "Left"
        elif left_movement < right_movement:
            return "Right"
        else:
            return None
    
    def pixel_to_camera_coords(u, v, depth_val, fx, fy, cx, cy):
        """
        Convert pixel coordinates + depth to camera coordinates.
        """
        x = (u - cx) * depth_val / fx
        y = (v - cy) * depth_val / fy
        z = depth_val
        return np.array([x, y, z])


    def get_camera_frame_point(self, wrist_landmark, depth_image, intrinsics):
        u = int(wrist_landmark.x * depth_image.shape[1])
        v = int(wrist_landmark.y * depth_image.shape[0])
        depth_val = depth_image[v, u] / 1000.0  # mm → meters

        if depth_val == 0:
            return None

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        px_to_cam_coords = self.pixel_to_camera_coords(u, v, depth_val, fx, fy, cx, cy)
    
        return px_to_cam_coords
    def process_frame(self, image, depth_image = None, transformation_matrix = None):
        process_start = time.time()
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_new = copy.copy(image)
        hand_results = self.hands.process(image_rgb)
        pose_results = self.pose.process(image_rgb)
        
        if hand_results.multi_hand_landmarks is None and pose_results.pose_landmarks is None:
            self.human_presence = False

            return image_rgb
        pointing_hand = None
        is_hand_pointing = False
        is_arm_pointing = False
        hand_handedness = None
        prev_arm = self.pointing_hand_handedness
        
        landmarks = pose_results.pose_world_landmarks
        landmarks_2d = pose_results.pose_landmarks
        self.human_presence = True
        # check if need to update previous pointing
        if self.previous_pointing:
            self.previous_gesture_conf = self.gesture_conf
            self.previous_pointing_hand_handedness = self.pointing_hand_handedness
            self.previous_pointing_confidence = self.pointing_confidence
            
        # detect moving hand
        # Detect arm movement using optical flow
        # movement_info = None
        if pose_results.pose_landmarks is None:
            return image
        movement_info = self.detect_arm_movement(image, current_gray,  pose_results.pose_landmarks.landmark)
        
        if movement_info is None:
            movement_info = self.previous_pointing_hand_handedness
        else:
            # if self.previous_pointing_hand_handedness is not None and (movement_info != self.previous_pointing_hand_handedness):
            #     movement_info = None
            self.previous_pointing_hand_handedness = movement_info
                
        print( "MOVING HAND:",movement_info)
        
        
        # draw pose landmarks
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(image_new, hand_results.multi_hand_landmarks[idx], mp.solutions.hands.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(image_new, pose_results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

        print('-----')        
        # for close proximity detect hand gestures
        if hand_results.multi_hand_world_landmarks:
            hand_pointing_confidence_list = []
            hand_handedness = None
            hand_confidence = 0
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_world_landmarks):
                handedness = hand_results.multi_handedness[idx].classification[0].label
                confidence = hand_results.multi_handedness[idx].classification[0].score
                # TODO: If use webcam or front camera, flip the handedness
                if handedness == 'Left':
                    handedness = 'Right'
                else:
                    handedness = 'Left'
                # Check if the hand is pointing
                
                hand_handedness, hand_confidence, is_hand_pointing = is_pointing_hand(hand_landmarks.landmark, handedness)
                
    
                hand_pointing_confidence_list.append((hand_handedness, hand_confidence, hand_landmarks.landmark))
        
            if len(hand_pointing_confidence_list) > 0:
            
                if movement_info in [hand_info[0] for hand_info in hand_pointing_confidence_list]:
                    hand_handedness, hand_confidence, is_hand_pointing = is_pointing_hand(hand_landmarks.landmark, movement_info)
                    
                    pointing_hand = [sublist[-1] for sublist in hand_pointing_confidence_list if sublist[0] == movement_info][0]
                    
                else:
                    hand_handedness, hand_confidence, pointing_hand= max(hand_pointing_confidence_list, key=lambda x: x[1])
            # print(f"{hand_handedness} hand confidence: {hand_confidence}")
            
        
        # Step 3: Detect human pose 
            
        # for far proximity detect arm gestures
        if landmarks:
            # Check if the arm is raised and extended
            arm_handedness, arm_confidence, is_arm_pointing = is_pointing_arm(landmarks.landmark)
            
            
            if hand_results.multi_hand_landmarks:
                self.hand_confidence = hand_confidence if hand_handedness == movement_info else 0
            else:
                self.hand_confidence = 0.5
            self.arm_confidence = arm_confidence if arm_handedness == movement_info else 0
            
            print(f" {arm_handedness} Arm Confidence: {self.arm_confidence}")
            self.pointing_confidence = self.arm_confidence * self.hand_confidence
            self.pointing_hand_handedness = movement_info
            print(f"{hand_handedness} hand confidence: {self.hand_confidence}")
            print(f"Pointing confidence, confidence: {self.pointing_confidence}")

            # Set pointing status based on confidence threshold
            self.pointing = self.pointing_confidence > 0
            if not self.pointing:
                self.pointing_hand_handedness = None
                
        else:
            self.save_gesture_data(self.frame, None, None, None, None, landmarks_2d,landmarks)
            self.frame += 1
            return image
        if not (is_hand_pointing or is_arm_pointing):
            self.pointing = False
            self.pointing_hand_handedness = None
            pointing_confidence = 0
        
        
        # if pointing is detected, update and store weighted vectors
        if self.pointing:
            
            # pointing_hand = self.pointing_hand_handedness
            
            vectors, vectors_conf = self.find_vectors(pointing_hand, landmarks)
            
            joints = self.find_joint_locations(pointing_hand, landmarks, concat = True)
            
            
            current_time = time.time()
            self.gesture_duration = time.time() - self.gesture_start_time
            self.previous_time = current_time
            prev_arm = self.pointing_hand_handedness
                
            # Save gesture data
            self.save_gesture_data(self.frame, self.gesture_duration, self.pointing_hand_handedness, vectors_conf, vectors, joints['wrist'],landmarks_2d, landmarks)
            
            vectors_2d, vectors_2d_conf = self.find_vectors(pointing_hand, landmarks_2d)
            joints_2d = self.find_joint_locations(pointing_hand, landmarks_2d)
            
            self.gesture_conf = vectors_2d_conf['shoulder_to_wrist']
            self.vectors = vectors_2d
            self.vectors_conf = vectors_2d_conf
            self.joints = joints_2d
            self.origin = joints_2d['wrist']
            
            self.display_visualization(image_new, joints_2d, vectors_2d)
            self.display_info(image_new, self.pointing_hand_handedness, arm_handedness, self.gesture_duration, vectors)
            
            
            self.previous_pointing = True
            

            # TODO: update and save pointing info
            print("pointing_conf: curr->",self.pointing_confidence,  "prev->",self.previous_pointing_confidence)
            if (self.pointing_confidence > 0.5) and (self.pointing_confidence > self.__POINTING_CONF_THRESHOLD) and (self.pointing_confidence > self.previous_pointing_confidence) and (self.gesture_conf > self.previous_gesture_conf):
                self.update_optimal_vector(image, depth_image, transformation_matrix)
        # if there is not pointing, clear all stored values
        else: 
            if self.pointing != self.previous_pointing:
                print("=======NEW GESTURE======")
                self.save_gesture_data(self.frame, None, None, None, None, None, landmarks_2d, landmarks)
                self.previous_pointing = False
                self.clear_values()
            
        self.frame += 1
        return image

    def find_vectors(self, pointing_hand, landmarks):
        """find eye-to-wrist, shoulder-to-wrist, elbow-to-wrist, and wrist-to-index vectors,
        as well as gaze vector (eye centroid to nose).
        return a dictionary of all vectors and normalized wrist location
        """
        try:
            wrist = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_WRIST if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        except (IndexError, TypeError):
            wrist = None

        try:
            elbow = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            ]
        except (IndexError, AttributeError):
            elbow = None

        try:
            shoulder = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ]
        except (IndexError, AttributeError):
            shoulder = None

        try:
            eye = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_EYE if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_EYE
            ]
        except (IndexError, AttributeError):
            eye = None
        try:
            nose = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        except (IndexError, AttributeError):
            nose = None

        # Compute the centroid of the two eyes
        try:
            left_eye = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE]
            eye_centroid = Landmark()
            eye_centroid.x = (left_eye.x + right_eye.x) / 2
            eye_centroid.y = (left_eye.y + right_eye.y) / 2
            eye_centroid.z = (left_eye.z + right_eye.z) / 2
        except (IndexError, AttributeError):
            eye_centroid = None

        eye_to_wrist_vector = calculate_vector(eye, wrist)
        shoulder_to_wrist_vector = calculate_vector(shoulder, wrist)
        elbow_to_wrist_vector = calculate_vector(elbow, wrist)
        nose_to_wrist_vector = calculate_vector(nose, wrist)
        # Gaze vector: from eye centroid to nose
        gaze_vector = calculate_vector(eye_centroid, nose)

        eye_to_wrist_vector_conf = calculate_vector_conf(eye, wrist)
        shoulder_to_wrist_vector_conf = calculate_vector_conf(shoulder, wrist)
        elbow_to_wrist_vector_conf = calculate_vector_conf(elbow, wrist)
        nose_to_wrist_vector_conf = calculate_vector_conf(nose, wrist)
        # Gaze confidence
        gaze_vector_conf = calculate_vector_conf(eye_centroid, nose)

        hand_vector_conf = self.hand_confidence
        if pointing_hand is not None:
            try:
                index_finger_tip = pointing_hand[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_mcp = pointing_hand[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
                index_finger_vector = calculate_vector(index_finger_mcp, index_finger_tip)
                wrist_to_index_vector = calculate_vector(wrist, index_finger_tip)
            except (IndexError, TypeError):
                index_finger_tip = landmarks.landmark[
                    mp.solutions.pose.PoseLandmark.LEFT_INDEX if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_INDEX
                ]
                index_finger_mcp = None
                index_finger_vector = None
                wrist_to_index_vector = calculate_vector(wrist, index_finger_tip)
        else:
            index_finger_tip = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_INDEX if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_INDEX
            ]
            index_finger_mcp = None
            index_finger_vector = None
            wrist_to_index_vector = calculate_vector(wrist, index_finger_tip)

        vectors = {
            "eye_to_wrist": eye_to_wrist_vector,
            "shoulder_to_wrist": shoulder_to_wrist_vector,
            "elbow_to_wrist": elbow_to_wrist_vector,
            "nose_to_wrist": nose_to_wrist_vector,
            "wrist_to_index": wrist_to_index_vector,
            "index_finger": index_finger_vector,
            "gaze": gaze_vector
        }

        vectors_conf = {
            "eye_to_wrist": eye_to_wrist_vector_conf,
            "shoulder_to_wrist": shoulder_to_wrist_vector_conf,
            "elbow_to_wrist": elbow_to_wrist_vector_conf,
            "nose_to_wrist": nose_to_wrist_vector_conf,
            "wrist_to_index": hand_vector_conf,
            "index_finger": hand_vector_conf,
            "gaze": gaze_vector_conf
        }
        return vectors, vectors_conf
    import numpy as np

    def angle_to_target(self, origin, vector, target):
        """
        Compute the angle (in degrees) between a pointing vector and the direction to a target.

        Args:
            origin (np.ndarray): 3D coordinates of the pointing origin (e.g. wrist) [x, y, z]
            vector (np.ndarray): 3D pointing vector direction [dx, dy, dz]
            target (np.ndarray): 3D coordinates of the target point [x, y, z]

        Returns:
            float: angle in degrees between the pointing vector and the direction to the target
        """
        if origin is None or vector is None or target is None:
            return None

        # Normalize vector
        vec_norm = vector / np.linalg.norm(vector)
        origin = np.array([origin.x, origin.y, origin.z])
        # Direction from origin to target
        target_dir = target - origin
        target_dir_norm = target_dir / np.linalg.norm(target_dir)

        # Dot product and angle
        dot_product = np.dot(vec_norm, target_dir_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)

    def landmark_to_vector(self, landmark):
        if landmark is None:
            return None
        return np.array([landmark.x, landmark.y, landmark.z])
    
    def vector_to_landmark(self, vector):
        if len(vector) != 3:
            raise ValueError("Input vector must have exactly 3 elements: [x, y, z]")
        
        return Landmark(x=vector[0], y=vector[1], z=vector[2])

    def find_joint_locations(self, pointing_hand, landmarks, concat = False):
        """find eye-to-wrist, shoulder-to-wrist, elbow-to-wrist, and wrist-to-index vectors
        return a dictionary of all vectors and normalized wrist location
        Also adds gaze_origin (midpoint between eyes) and gaze_target (nose landmark).
        """
        # if 3d location is passed in, we need to concat the hand detection onto body
        # to ensure that center is still from human body
        try:
            wrist = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_WRIST if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
            wrist_vector = np.array([wrist.x, wrist.y, wrist.z])
        except (IndexError, TypeError):
            wrist = None
        if pointing_hand is not None:
            try:
                wrist_hand = pointing_hand[mp.solutions.hands.HandLandmark.WRIST]
                wrist_hand_vector = np.array([wrist_hand.x, wrist_hand.y, wrist_hand.z])
                wrist_offset = wrist_vector - wrist_hand_vector
            except (IndexError, TypeError):
                wrist_hand = None
                wrist_offset = None
            try:
                index_finger_tip = pointing_hand[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            except (IndexError, TypeError):
                index_finger_tip = None
            try:
                index_finger_mcp = pointing_hand[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            except (IndexError, TypeError):
                index_finger_mcp = None
            if concat and wrist_offset is not None:
                index_finger_tip = self.vector_to_landmark(self.landmark_to_vector(wrist) + self.landmark_to_vector(index_finger_tip) - wrist_offset)
                index_finger_mcp = self.vector_to_landmark(self.landmark_to_vector(wrist) + self.landmark_to_vector(index_finger_mcp) - wrist_offset)
        else:
            index_finger_tip = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_INDEX if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_INDEX
            ]
            index_finger_mcp = None
        try:
            elbow = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_ELBOW if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
            ]
        except (IndexError, AttributeError):
            elbow = None

        try:
            shoulder = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_SHOULDER if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
            ]
        except (IndexError, AttributeError):
            shoulder = None

        try:
            eye = landmarks.landmark[
                mp.solutions.pose.PoseLandmark.LEFT_EYE if self.pointing_hand_handedness == "Left" else mp.solutions.pose.PoseLandmark.RIGHT_EYE
            ]
            eye_middle = (mp.solutions.pose.PoseLandmark.LEFT_EYE + mp.solutions.pose.PoseLandmark.RIGHT_EYE)/2
        except (IndexError, AttributeError):
            eye = None
        
        try:
            nose = landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
        except (IndexError, AttributeError):
            nose = None

        # Add support for gaze_origin (midpoint between left and right eyes)
        try:
            left_eye = landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_EYE]
            gaze_origin = Landmark()
            gaze_origin.x = (left_eye.x + right_eye.x) / 2
            gaze_origin.y = (left_eye.y + right_eye.y) / 2
            gaze_origin.z = (left_eye.z + right_eye.z) / 2
        except (IndexError, AttributeError):
            gaze_origin = None

        return {
            "eye": eye,
            "eye_middle": eye_middle,
            "shoulder": shoulder,
            "elbow": elbow,
            "nose": nose,
            "wrist": wrist,
            "index_finger": index_finger_tip,
            "index_finger_mcp": index_finger_mcp,
            "gaze_origin": gaze_origin,
            "gaze_target": nose
        }
        
        
    def display_visualization(self, image, joints, vectors):
        wrist = joints["wrist"]
        shoulder = joints["shoulder"]
        eye = joints["eye"]
        elbow = joints["elbow"]
        index_finger_tip = joints["index_finger"]
        # index_finger_mcp = joints["index_finger_mcp"]
            
        visualize_vector(image, wrist, vectors["shoulder_to_wrist"], self.__SHOULDER_WRIST_COLOR)
        visualize_vector(image, wrist, vectors["elbow_to_wrist"], self.__ELBOW_WRIST_COLOR)
        visualize_vector(image, wrist, vectors["eye_to_wrist"], self.__EYE_WRIST_COLOR)
        visualize_vector(image, wrist, vectors["eye_to_wrist"], self.__NOSE_WRIST_COLOR)
        
        # if index_finger_tip is not None:
        #     visualize_vector(image, index_finger_tip, vectors["wrist_to_index"], self.__WRIST_INDEX_COLOR)
        # if index_finger_mcp is not None:
        #     visualize_vector(image, index_finger_tip, vectors["index_finger"], self.__INDEX_COLOR)

    def display_info(self, image, pointing_hand_handedness, arm_handedness, gesture_duration, vectors):
        txt_c = (55, 65, 64)
        cv2.putText(image, f"Pointing Gesture Detected (Confidence: {self.pointing_confidence})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_c, 2)
        cv2.putText(image, f"Duration: {gesture_duration:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_c, 2)
        cv2.putText(image, f"Hand: {pointing_hand_handedness}  Arm: {arm_handedness}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,0.5, txt_c, 2)

        if vectors['eye_to_wrist'] is not None:
            cv2.putText(image, f"Eye-to-Wrist: {vectors['eye_to_wrist']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__EYE_WRIST_COLOR, 2)
        if vectors['shoulder_to_wrist'] is not None:
            cv2.putText(image, f"Shoulder-to-Wrist: {vectors['shoulder_to_wrist']}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__SHOULDER_WRIST_COLOR, 2)
        if vectors['elbow_to_wrist'] is not None:
            cv2.putText(image, f"Elbow-to-Wrist: {vectors['elbow_to_wrist']}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__ELBOW_WRIST_COLOR, 2)
        if vectors['nose_to_wrist'] is not None:
            cv2.putText(image, f"nose-to-Index: {vectors['nose_to_wrist']}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__NOSE_WRIST_COLOR, 2)
        # if vectors['wrist_to_index'] is not None:
        #     cv2.putText(image, f"wrist-to-index: {vectors['wrist_to_index']}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__WRIST_INDEX_COLOR, 2)
        # if vectors['index_finger'] is not None:
        #     cv2.putText(image, f"index_finger: {vectors['index_finger']}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.__INDEX_COLOR, 2)
    
        
    def run_stream(self, use_tag = False):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # comment the line below if camera is not flipped.
            image = cv2.flip(image, 1)  # 1 flips horizontally
            h, w, _= image.shape
            px = w//2
            py = h//2
            K = np.array([[fx, 0, px],[0, fy, py],[0, 0, 1]])
            processed_image = self.process_frame(image)
            cv2.imshow('Pointing Gesture Detection', processed_image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_video(self, color_path, depth_path=None):
        """Runs the gesture detection on a local video file. Optionally uses a depth video if provided."""

        color_cap = cv2.VideoCapture(color_path)
        depth_cap = cv2.VideoCapture(depth_path) if depth_path else None

        if not color_cap.isOpened():
            print(f"Error: Could not open color video {color_path}")
            return
        if depth_cap and not depth_cap.isOpened():
            print(f"Warning: Depth video {depth_path} could not be opened. Running in color-only mode.")
            depth_cap = None

        frame_num = 0
        output_path = color_path[:-4] + "_output"
        os.makedirs(output_path, exist_ok=True)

        while color_cap.isOpened():
            success_color, color_frame = color_cap.read()
            if not success_color:
                break

            if depth_cap:
                success_depth, depth_frame = depth_cap.read()
                if not success_depth:
                    print(f"Warning: Ran out of depth frames at frame {frame_num}")
                    break

                # Optional: Convert depth frame if it's not in 16-bit already
                if depth_frame.dtype != np.uint16:
                    depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
                    depth_frame = (depth_frame.astype(np.uint16) << 8)
            else:
                depth_frame = None

            h, w, _ = color_frame.shape
            self.px = w // 2
            self.py = h // 2

            processed = self.process_frame(color_frame, depth_image=depth_frame)
            cv2.imshow('Pointing Gesture Detection', processed)
            cv2.imwrite(f"{output_path}/f{frame_num:04d}.png", processed)

            frame_num += 1
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

        color_cap.release()
        if depth_cap:
            depth_cap.release()
        cv2.destroyAllWindows()


    # def run_video(self, video_path):
    #     """Runs the gesture detection on a local video file"""
            
    #     cap = cv2.VideoCapture(video_path)  # Open the video file
    #     frame_num = 0
    #     if not cap.isOpened():
    #         print(f"Error: Could not open video {video_path}")
    #         return
        
    #     while cap.isOpened():
    #         success, image = cap.read()
    #         if not success:
    #             break

    #         # Flip image horizontally if necessary, else remove the flip
    #         # image = cv2.flip(image, 1)
    #         h, w, _= image.shape
    #         px = w//2
    #         py = h//2
    #         processed_image = self.process_frame(image)
    #         cv2.imshow('Pointing Gesture Detection', processed_image)
    #         import os
    #         output_path = video_path[0:-4]
    #         # Create the directory if it does not exist
    #         if not os.path.exists(output_path):
    #             os.makedirs(output_path)

    #         cv2.imwrite(output_path+'/f%i.png'%(frame_num), processed_image)
    #         frame_num += 1
    #         if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
    #             break
    #     cap.release()
    #     cv2.destroyAllWindows()
    
    def run_image_folder(self, folder_path = '', output_csv_path = ''):
        """
        Process images and their corresponding depth images from a folder and save the results.
        
        Args:
            folder_path (str): Path to the folder containing the images.
            output_csv_path (str): Path to save the CSV with results.
        """
        # Camera intrinsic parameters
        K = np.array([[fx, 0, px],
                    [0, fy, py],
                    [0, 0, 1]])

        # Get a list of all color images in the folder (assume .jpg)
        color_images = sorted(glob.glob(os.path.join(folder_path, '*color_image_*.jpg')))
        depth_images = sorted(glob.glob(os.path.join(folder_path, '*depth_in_hand_color_frame_*.png')))

        if len(color_images) == 0 or len(depth_images) == 0:
            print("No images found in the specified folder.")
            return

        # Ensure the color and depth image lists have matching lengths
        if len(color_images) != len(depth_images):
            print("Mismatch between number of color and depth images.")
            return

        # Initialize frame counter
        frame_num = 0

        # Loop through the images
        for color_image_path, depth_image_path in zip(color_images, depth_images):
            print(f"Processing {color_image_path} and {depth_image_path}")

            # Load color and depth images
            image = cv2.imread(color_image_path)
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)

            # Process the frame
            processed_image = self.process_frame(image, depth_image,  K=K)

            # Save the processed image
            output_folder = os.path.join(folder_path, 'processed_images')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            output_image_path = os.path.join(output_folder, f'processed_frame_{frame_num}.png')
            cv2.imwrite(output_image_path, processed_image)

            # Save AprilTag and ground plane data (if available) in the DataFrame
            if hasattr(self, 'tag_centroids') and hasattr(self, 'ground_plane'):
                tag_centroids_data = self.tag_centroids if self.tag_centroids is not None else ''
                ground_plane_data = self.ground_plane if self.ground_plane is not None else ''
                self.data = pd.concat([self.data, pd.DataFrame({
                    'frame': [frame_num],
                    'tag_centroids': [tag_centroids_data],
                    'ground_plane': [ground_plane_data]
                })], ignore_index=True)

            # Increment frame number
            frame_num += 1

        # Save the data to a CSV file
        self.data.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")

# Usage
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Pointing Gesture Detection")
    parser.add_argument('--mode', type=str, choices=['live', 'video', 'image_folder', 'spot'], help="Mode to run the gesture detection. Choices: 'live', 'video', 'image_folder', 'spot'")
    parser.add_argument('--video_path', type=str, help="Path to the local video file. Required if mode is 'video'")
    parser.add_argument('--csv_path', type=str, help="Path to the saved csv file.'")

    args = parser.parse_args()

    detector = PointingGestureDetector()

    if args.mode == 'live':
        print("Running in live video mode (webcam)...")
        detector.run_stream(args.calibration_path)
        
    elif args.mode == 'video':
        if not args.video_path:
            print("Error: --video_path argument is required for 'video' mode")
        else:
            print(f"Running in local video mode with video: {args.video_path}")
            if args.calibration_path:
                detector.run_video(args.video_path, args.calibration_path, args.tag_size)
    elif args.mode == 'image_folder':
        print("Running with image folder that contains depths(png) and image(jpg) mode...")
        detector.run_image_folder(args.calibration_path)
    elif args.mode == 'spot':
        print("Running in spot video mode...")
        detector.run_spot()
    else: 
        # detector.run_stream()
        # TODO: comment out after testing
        # video_path = '/Users/ivy/Desktop/gesture_eval_right.mov'
        # video_path = '/Users/ivy/Desktop/gesture_test.mp4'
        video_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/1/Color.mp4'
        depth_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/1/Depth.mp4'
        # video_path = '/Users/ivy/Desktop/spot_gesture_eval/1003/gesture_eval_test/'
        args.csv_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/1/gesture_data.csv'
        bag_path = '/home/xhe71/Desktop/dog_data/BDL204_Waffle/BDL204_Waffles_PVP_01.bag'
        # px = 320
        # py = 240
        # fx = 552.0291012161067
        # fy = 552.0291012161067
        intr = find_realsense_intrinsics(bag_path)
        px = intr.ppx
        py = intr.ppy
        fx = intr.fx
        fy = intr.fy
        detector.run_video(video_path, depth_path)
        
    detector.data.to_csv(args.csv_path, index=False)
    print(f"Data saved to {args.csv_path}")
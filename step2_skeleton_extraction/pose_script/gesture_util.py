# gesture_util.py: helper functions for gesture detection
import mediapipe as mp
import numpy as np
import cv2
from skspatial.objects import Plane, Vector, Line
from scipy.spatial.transform import Rotation as R
import json
import pupil_apriltags as apriltag
import fcntl
import shutil
import os
import time
import cv2
import open3d as o3d
import mediapipe as mp
import numpy as np
import json
from mediapipe.python.solutions.pose import PoseLandmark
from scipy.spatial import cKDTree


def write_json_locked(data, json_path, max_retries=15, retry_delay=0.5):
    """
    Writes JSON safely with an exclusive lock and atomic file handling.

    - Uses a temporary file to prevent corruption.
    - Ensures the file is fully written before replacing the original.
    - Retries on failure to handle concurrent access.
    """
    temp_path = json_path + ".tmp"
    
    retries = 0
    while retries < max_retries:
        try:
            with open(temp_path, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the temp file
                json.dump(data, f, indent=4)  # Write JSON safely
                f.flush()
                os.fsync(f.fileno())  # Ensure data is fully written to disk
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock before closing

            # Atomically replace the original file (prevents corruption)
            shutil.move(temp_path, json_path)

            return  # Successfully written, exit function

        except OSError:
            print(f"Warning: Could not write to JSON. Retrying... ({retries+1}/{max_retries})")
            time.sleep(retry_delay)
            retries += 1

    raise RuntimeError(f"Error: Failed to write JSON file {json_path} after multiple retries.")


def read_json_locked(json_path, max_retries=15, retry_delay=0.5):
    """Reads JSON file safely with a shared lock (prevents writes but allows multiple readers)."""
    retries = 0
    while retries < max_retries:
        try:
            with open(json_path, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)  # Shared lock (allows multiple readers)
                data = json.load(f)  # Read JSON
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock BEFORE closing the file
                return data  # Return the parsed JSON
        except json.JSONDecodeError:
            print(f"Warning: JSON file is being updated. Retrying... ({retries+1}/{max_retries})")
            time.sleep(retry_delay)
            retries += 1
        except FileNotFoundError:
            print("Error: JSON file not found.")
            return None
    
    raise RuntimeError("Error: Could not read JSON file due to corruption or lock issues.")


# Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cos_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_vector(point1, point2):
    # calculate the vector between two points
    if point1 is None or point2 is None:
        return None
    if hasattr(point1, 'x'):
        return np.array([point2.x - point1.x, point2.y - point1.y, point2.z - point1.z])

    else:
        return np.array([point2['x'] - point1['x'], point2['y'] - point1['y'], point2['z'] - point1['z']])

def calculate_vector_conf(point1, point2):
    # calculate the vector between two points
    if point1 is None or point2 is None:
        return None
    if hasattr(point1, 'x'):
        return point2.visibility * point1.visibility

    else:
        return (point2['visibility'] * point1['visibility'])
   

def visualize_vector(image, start, vector, color=(0, 255, 0)):
    if start is None:
        return 
    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
    end_point = (int(start_point[0] + vector[0] * 7500),
                    int(start_point[1]+ vector[1] * 7500)  # Scaling for visibility
                    )
    
    cv2.arrowedLine(image, start_point, end_point, color, 3)
    
# fit a ground plane that is perpendicular to human's pose
def find_ground_plane(landmarks):
    mp_pose = mp.solutions.pose
    if hasattr(landmarks, 'landmark') or hasattr(landmarks[0], 'x'):
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z])
        left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].z])
        right_heel = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].z])
    else:
        left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER][0],landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER][1],landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER][2]])
        right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER][0],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER][1],landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER][2]])
        left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP][0],landmarks[mp_pose.PoseLandmark.LEFT_HIP][1],landmarks[mp_pose.PoseLandmark.LEFT_HIP][2]])
        right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP][0],landmarks[mp_pose.PoseLandmark.RIGHT_HIP][1],landmarks[mp_pose.PoseLandmark.RIGHT_HIP][2]])
        left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL][0],landmarks[mp_pose.PoseLandmark.LEFT_HEEL][1],landmarks[mp_pose.PoseLandmark.LEFT_HEEL][2]])
        right_heel = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL][0],landmarks[mp_pose.PoseLandmark.RIGHT_HEEL][1],landmarks[mp_pose.PoseLandmark.RIGHT_HEEL][2]])

    point = (left_heel + right_heel)/2
    
    # get the ground plane
    points = [left_shoulder, right_shoulder, left_hip, right_hip, right_heel, left_heel]
    normal = Line.best_fit(points).direction
    # Desired normal direction (e.g., you want the plane to point upwards in Y direction)
    desired_direction = np.array([0, -1, 0])
    plane_1 = Plane(point = point, normal = normal)
    # Check the alignment with the desired direction
    if np.dot(normal, desired_direction) < 0:
        # Flip the normal if it points in the opposite direction
        plane_1 = Plane(point=plane_1.point, normal=-normal)
    

    return plane_1

# target position conversion
def add_target(landmarks, x, y, z):
    mp_pose = mp.solutions.pose
    left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].z])
    right_heel = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].z])
    
    origin = (left_heel + right_heel)/2
    target = origin + np.array([x, y, z])
    return target

def plane_line_intersection(plane, line):
    return Plane(plane).intersect_line(Line(line))


def compute_rotation_matrix(normal_vector, target_vector):
    # Normalize the vectors
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # Calculate the rotation axis using cross product
    rotation_axis = np.cross(normal_vector, target_vector)
    
    # Calculate the angle between the two vectors using dot product
    angle = np.arccos(np.dot(normal_vector, target_vector))
    
    # Create the rotation object
    rotation = R.from_rotvec(rotation_axis * angle)
    
    return rotation

def transform_points(rotation, points):
    # Apply the rotation matrix to all points
    transformed_points = rotation.apply(points)
    return transformed_points

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to check if the person is standing still (hips are not moving)
def is_standing_still(previous_landmarks, current_landmarks, threshold=0.02):
    # Select landmarks for hips (landmark indices: 11 and 12 for hips)
    left_hip_prev = np.array([previous_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              previous_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    right_hip_prev = np.array([previous_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               previous_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    
    left_hip_curr = np.array([current_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                              current_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    right_hip_curr = np.array([current_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                               current_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

    # Calculate the displacement (movement) between frames
    left_hip_displacement = calculate_distance(left_hip_prev, left_hip_curr)
    right_hip_displacement = calculate_distance(right_hip_prev, right_hip_curr)

    # If the movement is below a threshold, consider the person standing still
    if left_hip_displacement < threshold and right_hip_displacement < threshold:
        return True  # Person is standing still
    else:
        return False  # Person is moving

# Function to check if the arm is resting on the side (shoulder, elbow, wrist aligned vertically)
def is_arm_resting(current_landmarks, threshold=0.05):
    # Get x-coordinates of the shoulder, elbow, and wrist
    left_shoulder = np.array([current_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
    left_elbow = np.array([current_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
    
    left_wrist = np.array([current_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                                current_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])
    # check x distance to determine if it is resting or not
    left_resting = abs(left_shoulder[0] - left_elbow[0]) < threshold and abs(left_elbow[0] - left_wrist[0]) < threshold
    left_forearm = calculate_distance(left_shoulder, left_elbow)
    left_upperarm = calculate_distance(left_elbow, left_wrist)

    right_shoulder = np.array([current_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
    right_elbow = np.array([current_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
    
    right_wrist = np.array([current_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                                current_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
    # check x distance to determine if it is resting or not
    right_resting = abs(left_shoulder[0] - left_elbow[0]) < threshold and abs(left_elbow[0] - left_wrist[0]) < threshold
    right_forearm = calculate_distance(right_shoulder, right_elbow)
    right_upperarm = calculate_distance(right_elbow, right_wrist)

    # Check if the x-coordinates are approximately aligned (within a small threshold)
    if left_resting or right_resting:
        if left_resting and right_resting:
            
            return True, (left_forearm + right_forearm)/2, (left_upperarm + right_upperarm)/2
        elif left_resting:
            return True, left_forearm, left_upperarm
        else:
            return True, right_forearm, right_upperarm
    else:
        return False, None, None  # No Arm is not resting
    

def is_pointing_hand(hand_landmarks, handedness):
        """
        Determines whether a hand is pointing.
        Returns:
            - handedness: 'Right' or 'Left'
            - confidence: float (confidence score of pointing)
            - is_pointing: bool (whether hand is pointing)
        """
        try:
            # Calculate the vector from the wrist to the index finger tip
            wrist = hand_landmarks[mp.solutions.hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP]
            index_finger_mcp = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            index_finger_lower = calculate_vector(index_finger_pip, index_finger_mcp)
            index_finger_higher = calculate_vector(index_finger_pip, index_finger_tip)
            index_finger_bend_angle = np.rad2deg(angle_between(index_finger_lower,index_finger_higher))
            index_finger_vector = calculate_vector(wrist, index_finger_tip)

            # Check if the index finger is extended by comparing its vector's magnitude to the others
            middle_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]

            middle_finger_vector = calculate_vector(wrist, middle_finger_tip)
            ring_finger_vector = calculate_vector(wrist, ring_finger_tip)
            pinky_vector = calculate_vector(wrist, pinky_tip)

            index_finger_extended = (
                np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(middle_finger_vector) and
                np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(ring_finger_vector) and
                np.linalg.norm(index_finger_vector) > 1.2 * np.linalg.norm(pinky_vector) and
                index_finger_bend_angle > 160
            )
            
            confidence = index_finger_bend_angle / 180 * 0.8
            is_pointing = index_finger_extended
            # print(handedness, "hand confidence",confidence)
            # # if handedness == 'Left':
            # #     handedness = 'Right'
            # # else:
            # #     handedness = 'Left'
            return handedness, confidence, is_pointing
        
        except (IndexError, TypeError, ZeroDivisionError):
            return handedness, 0.0, False

def is_pointing_arm(pose_landmarks, arm_handedness = None):
    """
    Determines whether the arm is raised and extended.
    Returns:
        - handedness: 'Right' or 'Left'
        - confidence: float (confidence score based on arm posture)
        - is_pointing: bool (whether arm is raised and extended)
    """
    # arm_handedness = 'Left'
    is_pointing = False
    try:
        # Get shoulder, elbow, and wrist landmarks based on handedness
        r_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        r_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        r_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        r_elbow = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        
        l_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        l_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        l_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        l_elbow = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
          
        # Calculate vectors & angle
        r_shoulder_to_wrist = calculate_vector(r_shoulder, r_wrist)
        r_shoulder_to_elbow =  calculate_vector(r_shoulder, r_elbow)
        r_elbow_to_wrist = calculate_vector(r_wrist, r_elbow)
        r_shoulder_to_hip = calculate_vector(r_shoulder, r_hip)
        r_arm_raise_angle = np.rad2deg(angle_between(r_shoulder_to_hip, r_shoulder_to_wrist))
        r_arm_bend_angle = np.rad2deg(angle_between(r_shoulder_to_elbow, r_elbow_to_wrist)) 
        
        l_shoulder_to_wrist = calculate_vector(l_shoulder, l_wrist)
        l_shoulder_to_elbow =  calculate_vector(l_shoulder, l_elbow)
        l_elbow_to_wrist = calculate_vector(l_wrist, l_elbow)
        l_shoulder_to_hip = calculate_vector(l_shoulder, l_hip)
        l_arm_raise_angle = np.rad2deg(angle_between(l_shoulder_to_hip, l_shoulder_to_wrist))
        l_arm_bend_angle = np.rad2deg(angle_between(l_shoulder_to_elbow, l_elbow_to_wrist)) 
        if arm_handedness == 'Left':
            confidence = l_arm_bend_angle/180
        elif arm_handedness == 'Right':
            confidence = r_arm_bend_angle/180
        else:
            if (l_arm_raise_angle > r_arm_raise_angle):
                arm_handedness = 'Left'
                # confidence is calculated by left arm to right arm raising angle ratio
                confidence = l_arm_bend_angle/180
            else:
                arm_handedness = 'Right'
                confidence = r_arm_bend_angle/180
            
        is_pointing = confidence > 0.5
        return arm_handedness, confidence, is_pointing

    except (IndexError, AttributeError):
        return arm_handedness, 0.0, False
    
# Function to calculate the depth (z) component from 2D projection using arm segment length
def calculate_depth_from_2d(arm_length_3d, arm_length_2d):
    if arm_length_3d > arm_length_2d:
        return np.sqrt(arm_length_3d**2 - arm_length_2d**2)
    else:
        return 0  # If the 2D length is greater than 3D length, we assume no forward extension.

# Function to calculate forward extension based on 2D projection and initial 3D rest position
def calculate_forward_extension(arm_length, start3d, end3d):
    # Calculate 3D arm segment lengths (constant)
    if hasattr(start3d,'x'):
        start2d = (start3d.x,start3d.y) 
        end2d = (end3d.x,end3d.y) 
        arm_length_2d = calculate_distance(start2d, end2d)
        
        # Calculate the z-depth (forward extension component) for elbow and wrist
        elbow_depth = calculate_depth_from_2d(arm_length, arm_length_2d)
        end_depth = start3d.z - elbow_depth
        # Calculate forward extension as the difference in the z-axis (depth) from rest
        # forward_extension = wrist_depth - wrist_rest_3d[2]
        # end2d.z = end_depth
        return end_depth
    else:
        start2d = start3d[:1]
        end2d = end3d[:1]
        arm_length_2d = calculate_distance(start2d, end2d)
        
        # Calculate the z-depth (forward extension component) for elbow and wrist
        elbow_depth = calculate_depth_from_2d(arm_length, arm_length_2d)
        end_depth = start3d[2] - elbow_depth
        # Calculate forward extension as the difference in the z-axis (depth) from rest
        # forward_extension = wrist_depth - wrist_rest_3d[2]
        # end3d[2] = end_depth
        return end_depth

# Smooth depth image to handle zero values
def smooth_depth_image(image, window_size=3):
    smoothed_image = np.copy(image)
    height, width = image.shape
    offset = window_size // 2
    non_zero_indices = np.argwhere(image > 0)
    
    for i, j in non_zero_indices:
        if i >= offset and i < height - offset and j >= offset and j < width - offset:
            neighborhood = image[i - offset:i + offset + 1, j - offset:j + offset + 1]
            if np.any(neighborhood == 0):
                non_zero_neighbors = neighborhood[neighborhood > 0]
                if len(non_zero_neighbors) > 0:
                    smoothed_image[i - offset:i + offset + 1, j - offset:j + offset + 1] = np.where(
                        neighborhood == 0, np.mean(non_zero_neighbors), neighborhood
                    )
    return smoothed_image

# Fit a plane using AprilTags
def find_ground_plane_apriltags(tags_3d):
    tags_3d_flat = np.vstack(tags_3d)
    plane_1 = Plane.best_fit(tags_3d_flat)
    normal = plane_1.normal
    # Desired normal direction (e.g., you want the plane to point upwards in Y direction)
    desired_direction = np.array([0, -1, 0])
    plane_1 = Plane(point = plane_1.point, normal = normal)
    # Check the alignment with the desired direction
    if np.dot(normal, desired_direction) < 0:
        # Flip the normal if it points in the opposite direction
        plane_1 = Plane(point=plane_1.point, normal=-normal)
        
    return plane_1

# Detect and calculate 3D positions of AprilTags
def find_tag_position(K, tag_size_m, image, tag_ids=None):
    """
    Detect and calculate 3D positions of AprilTags. Optionally, specify tag IDs to filter by.
    
    Args:
    - K: Intrinsic camera matrix.
    - tag_size_m: Size of the AprilTag in meters.
    - image: Image where AprilTags are detected.
    - tag_ids: List of tag IDs to filter by. Set to None to detect all tags.

    Returns:
    - tag_3d_points_from_cam: 3D positions of the specified AprilTags.
    """
    detector = apriltag.Detector(families="tag36h11",
                                 quad_decimate=1, 
                                 quad_sigma=0.5,
                                 refine_edges = 1, 
                                 decode_sharpening = 0.5)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tags = detector.detect(gray_image)
    if not tags:
        print("No AprilTags detected")
        return []

    tag_3d_points_from_cam = []
    tag_centroids = []
    half_size = tag_size_m / 2
    tag_real_world = np.array([
        [-half_size, -half_size, 0],
        [half_size, -half_size, 0],
        [half_size, half_size, 0],
        [-half_size, half_size, 0]
    ])

    for tag in tags:
        # If tag_ids is specified, only process the tags that match the specified IDs
        if tag_ids is not None:
            if isinstance(tag_ids, int):
                tag_ids = [tag_ids]
            
            if tag.tag_id not in tag_ids:
                continue  # Skip tags not in the specified list
            
        # Process the tag
        corners_2d = np.array(tag.corners, dtype='float32')
        retval, rvec, tvec = cv2.solvePnP(tag_real_world, corners_2d, K, distCoeffs=None)
        rmat, _ = cv2.Rodrigues(rvec)
        tag_3d_camera = (rmat @ tag_real_world.T + tvec).T
        tag_3d_points_from_cam.append((tag.tag_id, tag_3d_camera))

        # Optionally, print the centroid for each tag
        tag_centroid = np.mean(tag_3d_camera, axis=0)
        tag_centroids.append((tag.tag_id, tag_centroid))
        print(f"AprilTag {tag.tag_id} centroid in 3D (camera frame): {tag_centroid}")
        
    return tag_3d_points_from_cam, tag_centroids

def estimate_height(results):
    # Japanese body height ratio: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4140637
    shoulder_height_ratio = 1.24
    if results.pose_world_landmarks:
        # Extract 3D landmarks for the necessary body parts
        nose = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Compute the average hip, knee, and ankle positions
        avg_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2, (left_hip.z + right_hip.z) / 2])
        avg_knee = np.array([(left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2, (left_knee.z + right_knee.z) / 2])
        avg_ankle = np.array([(left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2, (left_ankle.z + right_ankle.z) / 2])

        # Calculate vertical distances for each body segment
        nose_to_hip = np.linalg.norm(np.array([nose.x, nose.y, nose.z]) - avg_hip)
        hip_to_knee = np.linalg.norm(avg_hip - avg_knee)
        knee_to_ankle = np.linalg.norm(avg_knee - avg_ankle)

        # Calculate total height
        total_height = (nose_to_hip + hip_to_knee + knee_to_ankle) * shoulder_height_ratio


    else:
        print("No landmarks detected.")
    if results.pose_landmarks:
        # Extract 3D landmarks for the necessary body parts
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Compute the average hip, knee, and ankle positions
        avg_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2, (left_hip.z + right_hip.z) / 2])
        avg_knee = np.array([(left_knee.x + right_knee.x) / 2, (left_knee.y + right_knee.y) / 2, (left_knee.z + right_knee.z) / 2])
        avg_ankle = np.array([(left_ankle.x + right_ankle.x) / 2, (left_ankle.y + right_ankle.y) / 2, (left_ankle.z + right_ankle.z) / 2])

        # Calculate vertical distances for each body segment
        nose_to_hip = np.linalg.norm(np.array([nose.x, nose.y, nose.z]) - avg_hip)
        hip_to_knee = np.linalg.norm(avg_hip - avg_knee)
        knee_to_ankle = np.linalg.norm(avg_knee - avg_ankle)

        # Calculate total height
        total_height_px_normalized = (nose_to_hip + hip_to_knee + knee_to_ankle) * shoulder_height_ratio

    else:
        print("No landmarks detected.")
    return total_height, total_height_px_normalized

# Project 2D to 3D using depth information
def project_2d_to_3d(u, v, Z, K):
    point_2d_homogeneous = np.array([u, v, 1])
    K_inv = np.linalg.inv(K)
    point_3d_homogeneous = Z * K_inv @ point_2d_homogeneous
    return point_3d_homogeneous[:3]

# Get the average depth of neighboring non-zero values
def get_depth(depth_image, x, y, window_size=5):
    half_size = window_size // 2
    x_min = max(0, x - half_size)
    x_max = min(depth_image.shape[1], x + half_size + 1)
    y_min = max(0, y - half_size)
    y_max = min(depth_image.shape[0], y + half_size + 1)

    window = depth_image[y_min:y_max, x_min:x_max]
    non_zero_values = window[window > 0]

    if len(non_zero_values) > 0:
        z = np.mean(non_zero_values) / 1000  # Convert to meters
    else:
        z = 0
    return z

def transform_world_landmarks_to_camera_frame(results, hip_center_3d, ground_plane_normal, target_normal=np.array([0, -1, 0])):
    """
    Transforms the world landmarks to the camera frame based on hip_center_3d, while keeping the landmark structure.
    Returns the transformed pose landmarks structure.
    """
    if not results.pose_world_landmarks:
        return None

    transformed_landmarks = []

    # Iterate through the landmarks and adjust the position relative to hip_center_3d
    for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
        # Get the 3D coordinates of the landmark
        point_3d = np.array([landmark.x, landmark.y, landmark.z])

        # Offset by the hip center to transform to camera frame
        # Align the point to the ground plane (rotate so that the ground plane normal aligns with (0, 1, 0))
        aligned_point = align_to_ground_plane(point_3d, ground_plane_normal, target_normal)
        # aligned_point = point_3d
        transformed_point = aligned_point + hip_center_3d
        # aligned_point = transformed_point
        # Create a new landmark with the transformed coordinates while keeping the original structure
        # new_landmark = mp.solutions.pose.PoseLandmark()
        landmark.x = transformed_point[0]
        landmark.y = transformed_point[1]
        landmark.z = transformed_point[2]
        transformed_landmarks.append(landmark)

    return transformed_landmarks, results

# Transform points to align AprilTags to ground plane
def align_to_ground_plane(points_3d, ground_plane_normal, target_normal=np.array([0, -1, 0])):
    """
    Aligns the 3D points to the ground plane by rotating them such that the ground plane's normal
    aligns with the target normal (defaulting to (0, -1, 0), i.e., Y-axis pointing up).

    Args:
    - points_3d: A numpy array of 3D points to be aligned.
    - ground_plane_normal: The normal vector of the current ground plane.
    - target_normal: The target normal to align to (defaults to (0, -1, 0)).

    Returns:
    - Aligned 3D points after rotation.
    """
    # Normalize the normal vectors to avoid issues with magnitude differences
    current_normal = ground_plane_normal / np.linalg.norm(ground_plane_normal)
    target_normal = target_normal / np.linalg.norm(target_normal)

    # Find the rotation matrix that aligns current_normal to target_normal
    rotation, _ = R.align_vectors([target_normal], [current_normal])

    # Apply the rotation to all the points
    aligned_points = rotation.apply(points_3d)

    return aligned_points


def load_intrinsics_and_transform(json_file_path):
    """
    Load intrinsics and transformation matrices from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file.
    
    Returns:
        tuple: Intrinsics matrix (3x3 numpy array), transformation matrix (4x4 numpy array).
    """
    data = read_json_locked(json_file_path)
    
    
    intrinsics = np.array(data["intrinsics"])
    rotation_matrix = np.array(data["transform"]["rotation_matrix"])
    translation_vector = np.array(data["transform"]["translation_vector"])
    
    # Combine into a single 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector
    
    return intrinsics, transformation_matrix

def image_pt_to_pointcloud_pt(intrinsics, transformation, x, y, depth):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    if depth > 500:
        z = depth / 1000.0  # Convert depth to meters
    else:
        z = depth
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    camera_point = np.array([x, y, z, 1])
    world_point = transformation @ camera_point
    return world_point[0:3]



# Load point cloud data
def load_pointcloud_data(pointcloud_file):
    return o3d.io.read_point_cloud(pointcloud_file)


# Load images and resize depth to match color dimensions
def load_images_and_resize_depth(color_image_path, depth_image_path):
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    h_color, w_color, _ = color_image.shape
    depth_image_resized = cv2.resize(depth_image, (w_color, h_color), interpolation=cv2.INTER_NEAREST)
    return color_image, depth_image_resized

def recreate_depth_image_from_pointcloud(color_image, point_cloud, intrinsics, color_to_world_transform):
    """
    Recreate the depth image by finding depth values in the point cloud for each color image pixel.

    Args:
        color_image (np.ndarray): Color image (HxWx3).
        point_cloud (np.ndarray): Nx3 array of 3D points in world coordinates.
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
        color_to_world_transform (np.ndarray): 4x4 transformation matrix from color image to world frame.

    Returns:
        np.ndarray: Recreated depth image (HxW) in millimeters.
    """
    h, w, _ = color_image.shape
    depth_image = np.zeros((h, w), dtype=np.float32)
    depth_buffer = np.full((h, w), np.inf)  # Buffer to store the minimum depth



    # Build a KD-tree for efficient nearest neighbor search in the point cloud
    kdtree = cKDTree(point_cloud)

    # Compute the inverse of the color image's transformation matrix
    world_to_color_transform = np.linalg.inv(color_to_world_transform)
    points_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    camera_points = (world_to_color_transform @ points_homogeneous.T).T[:, :3]

    # Project the points into the image plane
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    for point in camera_points:
        x, y, z = point
        if z <= 0:  # Skip points behind the camera
            continue

        # Map 3D points to 2D pixel coordinates
        u = int((fx * x / z) + cx)
        v = int((fy * y / z) + cy)
        # Check if the pixel is within bounds
        if 0 <= u < w and 0 <= v < h:
            # Update the depth buffer with the nearest point
            if z < depth_buffer[v, u]:
                depth_image[v, u] = z
                depth_buffer[v, u] = z

    # Convert depth to millimeters
    depth_image = (depth_image * 1000).astype(np.uint16)

    return depth_image

def visualize_color_image_in_pointcloud(point_cloud, color_image, intrinsics, color_to_world_transform):
    """
    Visualize the color image within the point cloud by projecting color image pixels into the point cloud.

    Args:
        point_cloud (np.ndarray): Nx3 array of 3D points (world coordinates).
        color_image (np.ndarray): Color image (HxWx3).
        intrinsics (np.ndarray): 3x3 intrinsic matrix of the camera.
        color_to_world_transform (np.ndarray): 4x4 transformation matrix from color image to world frame.
    """
    h, w, _ = color_image.shape
    projected_points = []

    # Compute the inverse transformation matrix
    world_to_color_transform = np.linalg.inv(color_to_world_transform)

    # Intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Loop over each pixel in the color image
    for v in range(h):
        for u in range(w):
            # Project pixel (u, v) into the camera frame
            x = (u - cx) / fx
            y = (v - cy) / fy
            z = 1.0  # Assume unit depth
            camera_point = np.array([x * z, y * z, z, 1.0])

            # Transform from camera frame to world frame
            world_point = color_to_world_transform @ camera_point
            projected_points.append(world_point[:3])

    # Convert projected points to Open3D format
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector(np.array(projected_points))

    # Assign color from the color image
    colors = np.reshape(color_image, (-1, 3)) / 255.0
    projected_pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize both point clouds
    o3d.visualization.draw_geometries([o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud)), projected_pcd])
    


# Create cone of raycasts for visualization
def create_cone_of_raycasts(origin, direction, cone_angle, num_rays=50, ray_length=2.0):
    direction = direction / np.linalg.norm(direction)
    z_axis = direction
    x_axis = np.cross(z_axis, [0, 1, 0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rays = []

    for theta in np.linspace(0, cone_angle/2 , 10):
        for phi in np.linspace(0, 2 * np.pi, num_rays):
            ray_dir = np.cos(theta) * z_axis + np.sin(theta) * (np.cos(phi) * x_axis + np.sin(phi) * y_axis)
            ray_dir /= np.linalg.norm(ray_dir)
            end_point = origin + ray_length * ray_dir
            rays.append([origin, end_point])
    return rays

# Detect human skeleton and extract 3D points
def detect_and_extract_skeleton(color_image, depth_image, intrinsics, transformation, pointing_hand = 'Left'):
    pose = mp.solutions.pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.5, model_complexity=2)
    results = pose.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
    mp_pose = mp.solutions.pose
    if not results.pose_landmarks:
        return None, None, None, None

    h, w, _ = color_image.shape
    skeleton_points = []
    depth_list = []
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        if 0 <= y_px < h and 0 <= x_px < w:  # Ensure valid indexing
            depth = depth_image[y_px, x_px]
            depth_list.append(depth)
    
    avg_depth = np.average(depth_list)
    wrist_depth = None
    direction = None
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        # to ensure unit conversion is correct
        if avg_depth < 500:
            z_gesture = results.pose_world_landmarks.landmark[idx].z
        else:
            z_gesture = results.pose_world_landmarks.landmark[idx].z * 1000
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        if 0 <= y_px < h and 0 <= x_px < w:  # Ensure valid indexing
            depth = avg_depth + z_gesture
            if pointing_hand == 'Left' and idx ==  mp_pose.PoseLandmark.LEFT_WRIST.value:
                wrist_depth = [x_px, y_px, depth]
            elif pointing_hand == 'Right' and idx ==  mp_pose.PoseLandmark.RIGHT_WRIST.value:
                wrist_depth = [x_px, y_px, depth]
            
            if pointing_hand == 'Left' and idx ==  mp_pose.PoseLandmark.LEFT_SHOULDER.value:
                direction = [x_px, y_px, depth]
            elif pointing_hand == 'Right' and idx ==  mp_pose.PoseLandmark.RIGHT_SHOULDER.value:
                direction = [x_px, y_px, depth]

            if depth > 0:  # Valid depth
                point_3d = image_pt_to_pointcloud_pt(intrinsics, transformation, x_px, y_px, depth)
                skeleton_points.append(point_3d)
            else:
                skeleton_points.append(None)
        else:
            skeleton_points.append(None)
    return skeleton_points, avg_depth, wrist_depth, direction

# Visualize skeleton, pointing direction, and cone in the point cloud
def visualize_skeleton_and_cone_in_pointcloud(pcd, skeleton_points, pose_connections, origin_3d, direction, cone_angle):
    valid_points = [point for point in skeleton_points if point is not None]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(valid_points)

    skeleton_lines = [
        [i, j] for i, j in pose_connections if i < len(valid_points) and j < len(valid_points)
    ]
    line_set.lines = o3d.utility.Vector2iVector(skeleton_lines)
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(skeleton_lines))  # Green skeleton lines

    pointing_line = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([origin_3d, origin_3d + direction]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    pointing_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red pointing line

    rays = create_cone_of_raycasts(origin_3d, direction, cone_angle)
    cone_points = [point for ray in rays for point in ray]
    cone_lines = [[i, i + 1] for i in range(0, len(cone_points), 2)]
    cone_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cone_points),
        lines=o3d.utility.Vector2iVector(cone_lines)
    )
    cone_line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(cone_lines))  # Red cone lines
    
    o3d.visualization.draw_geometries([pcd, line_set, pointing_line, cone_line_set])

def generate_pointcloud_from_depth_and_color(depth_image, color_image, intrinsics):
    """
    Generate a point cloud from the depth image and color image.

    Args:
        depth_image (np.ndarray): Depth image (HxW) in millimeters.
        color_image (np.ndarray): Color image (HxWx3).
        intrinsics (np.ndarray): 3x3 intrinsic matrix.

    Returns:
        o3d.geometry.PointCloud: Point cloud with colors.
    """
    h, w = depth_image.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # Convert depth to meters
        
    z = depth_image / 1000.0  # Convert depth to meters
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into a Nx3 array
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    # Filter out invalid points where depth is zero
    valid_mask = z > 0
    points = points[valid_mask.flatten()]

    # Map colors to valid points
    colors = color_image.reshape(-1, 3)[valid_mask.flatten()] / 255.0

    # Create the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

# Main function
def map_to_3d():
    json_file_path = ".tmp/gesture_transformation.json"
    pointing_detection_file_path = ".tmp/gesture_vector.json"

    color_image_path = ".tmp/gesture_color.png"
    depth_image_path = ".tmp/gesture_depth.png"

    intrinsics, transformation_matrix = load_intrinsics_and_transform(json_file_path)
    
    pointing_data = read_json_locked(pointing_detection_file_path)
    
    # pcd = load_pointcloud_data(pointcloud_file)
    color_image, depth_image = load_images_and_resize_depth(color_image_path, depth_image_path)
    h, w, _ = color_image.shape

    skeleton_points, avg_depth, wrist_depth, direction_shoulder = detect_and_extract_skeleton(color_image, depth_image, intrinsics, transformation_matrix, pointing_hand = pointing_data['pointing_arm'])
    
    print(avg_depth)
    if skeleton_points is None:
        print("No skeleton detected.")
        return None, None, None

    origin = np.array(pointing_data['pointing_vector_origin'])
    direction = np.array(pointing_data['pointing_vector_dir'])
    cone_angle = np.deg2rad(pointing_data['pointing_vector_opening_angle'])
    origin_x, origin_y, origin_z = wrist_depth[0], wrist_depth[1], wrist_depth[2]
    direction_x, direction_y, direction_z = direction_shoulder[0], direction_shoulder[1], direction_shoulder[2]
    # transform origin to world frame
    origin_3d = image_pt_to_pointcloud_pt(intrinsics, transformation_matrix, origin_x, origin_y, origin_z)
    # transform the direction to world frame
    direction_3d = - image_pt_to_pointcloud_pt(intrinsics, transformation_matrix, direction_x, direction_y, direction_z)
    
    pointing_data["origin_world"] = origin_3d.tolist()
    pointing_data["direction_world"] = direction_3d.tolist()
    pointing_data["intrinsics"] = intrinsics.tolist()
    pointing_data["transformation"] = transformation_matrix.tolist()
    write_json_locked(pointing_data, pointing_detection_file_path)
    return origin, direction, cone_angle

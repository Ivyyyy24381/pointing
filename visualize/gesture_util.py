# gesture_util.py: helper functions for gesture detection
import mediapipe as mp
import numpy as np
import cv2
from skspatial.objects import Plane, Vector, Line
from scipy.spatial.transform import Rotation as R

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
    return np.array([point2.x - point1.x, point2.y - point1.y, point2.z - point1.z])


def visualize_vector(image, start, vector, color=(0, 255, 0)):
    
    
    if start is None:
        return 
    # Get image dimensions
    height, width = image.shape[:2]
    
    start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
    
    # # Normalize the vector
    # vector = np.array(vector)
    # vector = vector / np.linalg.norm(vector)

    # # Calculate the scale factor to extend to the image edges
    # # Get the max scaling factor for x or y to reach the edge of the screen
    # scale_x = width / abs(vector[0]) if vector[0] != 0 else np.inf
    # scale_y = height / abs(vector[1]) if vector[1] != 0 else np.inf
    # scale_factor = min(scale_x, scale_y)

    # # Compute the end point that extends to the edge of the screen
    # end_point = (int(start_point[0] + vector[0] * scale_factor),
    #              int(start_point[1] + vector[1] * scale_factor))


    end_point = (int(start_point[0] + vector[0] * 7500),
                    int(start_point[1]+ vector[1] * 7500)  # Scaling for visibility
                    )
    
    cv2.arrowedLine(image, start_point, end_point, color, 3)
    
# fit a ground plane that is perpendicular to human's pose
def find_ground_plane(landmarks):
    mp_pose = mp.solutions.pose
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,landmarks[mp_pose.PoseLandmark.LEFT_HIP].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z])
    left_heel = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y,landmarks[mp_pose.PoseLandmark.LEFT_HEEL].z])
    right_heel = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].z])

    point = (left_heel + right_heel)/2
    
    # get the ground plane
    points = [left_shoulder, right_shoulder, left_knee, right_knee, left_hip, right_hip, right_heel, left_heel]
    normal = Line.best_fit(points).direction
    plane_1 = Plane(point = point, normal = normal)

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
    # check x distance to determin if it is resting or not
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
    # check x distance to determin if it is resting or not
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
                index_finger_bend_angle > 130
            )
            
            confidence = index_finger_bend_angle / 180  #confidence score based on bend angle
            is_pointing = index_finger_extended

            return handedness, confidence, is_pointing
        
        except (IndexError, TypeError, ZeroDivisionError):
            return handedness, 0.0, False

def is_pointing_arm(pose_landmarks):
    """
    Determines whether the arm is raised and extended.
    Returns:
        - handedness: 'Right' or 'Left'
        - confidence: float (confidence score based on arm posture)
        - is_pointing: bool (whether arm is raised and extended)
    """
    arm_handedness = 'Left'
    is_pointing = False
    try:
        # Get shoulder, elbow, and wrist landmarks based on handedness
        r_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        r_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        r_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]

        
        l_shoulder = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        l_wrist = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        l_hip = pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
                
        # Calculate vectors & angle
        r_shoulder_to_wrist = calculate_vector(r_shoulder, r_wrist)
        r_shoulder_to_hip = calculate_vector(r_shoulder, r_hip)
        r_arm_raise_angle = np.rad2deg(angle_between(r_shoulder_to_hip, r_shoulder_to_wrist))
        
        l_shoulder_to_wrist = calculate_vector(l_shoulder, l_wrist)
        l_shoulder_to_hip = calculate_vector(l_shoulder, l_hip)
        l_arm_raise_angle = np.rad2deg(angle_between(l_shoulder_to_hip, l_shoulder_to_wrist))
        
        if (l_arm_raise_angle > r_arm_raise_angle):
            arm_handedness = 'Left'
            # confidence is calculated by left arm to right arm raising angle ratio
            confidence = l_arm_raise_angle / (l_arm_raise_angle + r_arm_raise_angle)
        else:
            arm_handedness = 'Right'
            confidence = r_arm_raise_angle / (l_arm_raise_angle + r_arm_raise_angle)
            
        is_pointing = confidence > 0.5

        return arm_handedness, confidence, is_pointing

    except (IndexError, AttributeError):
        return arm_handedness, 0.0, False
    
def calibration(landmarks):
    return

import numpy as np

# Function to calculate the depth (z) component from 2D projection using arm segment length
def calculate_depth_from_2d(arm_length_3d, arm_length_2d):
    if arm_length_3d > arm_length_2d:
        return np.sqrt(arm_length_3d**2 - arm_length_2d**2)
    else:
        return 0  # If the 2D length is greater than 3D length, we assume no forward extension.

# Function to calculate forward extension based on 2D projection and initial 3D rest position
def calculate_forward_extension(shoulder_rest_3d, elbow_rest_3d, wrist_rest_3d,
                                shoulder_2d, elbow_2d, wrist_2d):
    # Calculate 3D arm segment lengths (constant)
    upper_arm_length_3d = calculate_distance(shoulder_rest_3d, elbow_rest_3d)
    lower_arm_length_3d = calculate_distance(elbow_rest_3d, wrist_rest_3d)

    # Calculate 2D arm segment lengths from projections
    upper_arm_length_2d = calculate_distance(shoulder_2d, elbow_2d)
    lower_arm_length_2d = calculate_distance(elbow_2d, wrist_2d)

    # Calculate the z-depth (forward extension component) for elbow and wrist
    elbow_depth = calculate_depth_from_2d(upper_arm_length_3d, upper_arm_length_2d)
    wrist_depth = calculate_depth_from_2d(lower_arm_length_3d, lower_arm_length_2d)
    print("elbow_depth:", elbow_depth)
    print("wrist_depth:", wrist_depth)
    # Calculate forward extension as the difference in the z-axis (depth) from rest
    forward_extension = wrist_depth - wrist_rest_3d[2]

    return forward_extension

# Example data

# 3D resting positions (shoulder, elbow, and wrist) when arm is at rest
shoulder_rest_3d = (0, 0, 0)
elbow_rest_3d = (0, -0.3, 0)
wrist_rest_3d = (0, -0.6, 0)

# 2D projection data after arm extension (shoulder, elbow, and wrist)
shoulder_2d = (0, 0)
elbow_2d = (0.1, -0.2)
wrist_2d = (0.1, -0.4)

# Calculate forward extension
forward_extension = calculate_forward_extension(shoulder_rest_3d, elbow_rest_3d, wrist_rest_3d,
                                                shoulder_2d, elbow_2d, wrist_2d)

# Output the forward extension result
print(f"Forward extension of the arm: {forward_extension:.2f} units")
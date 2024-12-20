# takes in the gesture and target, evaluate distance to the target
import sys
import os
from skspatial.objects import Plane, Vector, Line
from skspatial.plotting import plot_3d
from scipy.spatial.transform import Rotation as R
import ast
# Add the path where pose_detection is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
import argparse
from visualize.gesture_util import *
import pandas as pd
import re

import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# List of lists
# Function to convert string list to vector dictionary

class Landmark:
    def __init__(self, x, y, z, visibility):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

def dict_to_landmark(d):
    return Landmark(d['x'], d['y'], d['z'], d['visibility'])

def to_vector(vec_str):
    try:
        # Step 1: Remove brackets and split the string into individual values
        if type(vec_str) is str:
            vec_str = vec_str.strip('[]')  # Remove the brackets
            vec = list(map(float, vec_str.split()))  # Split by spaces and convert to floats
        else:
            vec = vec_str
        # Step 2: Create dictionary with 'x', 'y', 'z' keys
        landmarks = {'x': vec[0], 'y': vec[1], 'z': vec[2],'visibility': None}
        
        landmarks = dict_to_landmark(landmarks)
        return landmarks
    except Exception as e:
        print(f"Error parsing vector {vec_str}: {e}")
        return None

def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_angle(vec1, vec2):
    """
    Calculate the angle between two vectors in degrees.
    """
    dot_product = np.dot(vec1, vec2)
    magnitude_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cos_theta = np.clip(dot_product / magnitude_product, -1.0, 1.0)  # Avoid numerical instability
    return np.degrees(np.arccos(cos_theta))

def parse_landmarks(landmark_str):
    # Find all the blocks of 'landmark {...}' and extract the x, y, z, and visibility values
    landmark_blocks = re.findall(r'landmark\s*{([^}]*)}', landmark_str)
    
    landmarks = []
    for block in landmark_blocks:
        # Extract the x, y, z, and visibility values from each block
        landmark = {}
        landmark['x'] = float(re.search(r'x:\s*([-\d\.]+)', block).group(1))
        landmark['y'] = float(re.search(r'y:\s*([-\d\.]+)', block).group(1))
        landmark['z'] = float(re.search(r'z:\s*([-\d\.]+)', block).group(1))
        landmark['visibility'] = float(re.search(r'visibility:\s*([-\d\.]+)', block).group(1))
        
        landmarks.append(landmark)
    landmarks = [dict_to_landmark(landmark_dict) for landmark_dict in landmarks]
    return landmarks


def wrist_location_to_dict(wrist_str):
    try:
        # Split the string by newline characters and then split by ': ' to create key-value pairs
        wrist_data = dict(item.split(': ') for item in wrist_str.strip().split('\n') if item)
        # Convert the string values to float
        wrist_data = {k: float(v) for k, v in wrist_data.items()}
        return  dict_to_landmark(wrist_data)
    except Exception as e:
        print(f"Error converting wrist location: {e}")
        return None

def vector_to_raycast(landmark_1, landmark_2):
    return Line(point = [landmark_1.x, landmark_1.y, landmark_1.z], direction=[landmark_2.x, landmark_2.y, landmark_2.z])



# Function to visualize in 3D
def visualize_landmarks_and_target(landmarks, ground_plane_z, target_location, transformed_wrist_location, vector_intersections, frame, output_path):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot landmarks
    xs = landmarks[:, 0]
    ys =landmarks[:, 1]
    zs = landmarks[:, 2]
    
    # Plot the landmarks in 3D space
    ax.scatter(xs, ys, zs, c='blue', label='Landmarks', s=50, alpha=0.8)
 
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    # Plot the target location in 3D space
    ax.scatter(target_location[0], target_location[1], target_location[2], c='red', s=100, label='Target', alpha=0.9)
    # Draw connections between skeleton points (Mediapipe connections)
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
    
    for conn in connections:
        point1 = landmarks[conn[0]]
        point2 = landmarks[conn[1]]
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], c='b')
    
    # plot ground plane
    ground_plane_z.plot_3d(ax, alpha = 0.2)
    
    # plot target plane
    target_plane = Plane(target_location, ground_plane_z.normal)
    target_plane.point.plot_3d(ax, alpha = 0.2)
    target_plane.plot_3d(ax, alpha = 0.2)

    # Add lines connecting wrist landmark to target for better visualization
    wrist_idx = 0  # Change this index to the specific wrist landmark index
    ax.plot([landmarks[wrist_idx][0], target_location[0]],
            [landmarks[wrist_idx][1], target_location[1]],
            [landmarks[wrist_idx][2], target_location[2]], color='orange', linestyle='--', label='Wrist to Target')



    # Plot each vector (eye-to-wrist, shoulder-to-wrist, etc.)
    if hasattr(transformed_wrist_location, 'x'):
        transformed_wrist_location = np.array([[transformed_wrist_location.x, transformed_wrist_location.y, transformed_wrist_location.z]])
    for name, vec in vector_intersections.items():
        # vec = [vec['x'], vec['y'], vec['z']]
        ax.plot([transformed_wrist_location[0][0], vec[0]],
                [transformed_wrist_location[0][1], vec[1]],
                [transformed_wrist_location[0][2], vec[2]],
                label=name.replace('_', ' ').title(), linestyle='-', marker='o')

    
    # Labels
    ax.set_xlabel('X Axis[m]')
    ax.set_ylabel('Y Axis[m]')
    ax.set_zlabel('Z Axis[m]')
    ax.view_init(elev=-45, azim=-90, roll=0)

    # Title and legend
    ax.set_title("3D Visualization of Landmarks and Target Location")
    ax.legend(loc='upper right', fontsize = 'x-small')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim( -0.6, 0.8)
    ax.set_zlim(-5, 1.5)
    # Save the plot as an image
    output_file = output_path+"/plot_p%i.png"%frame
    plt.savefig(output_file, dpi=300)
    
    # Show the plot
    # plt.show()
    plt.close()
    
    
def evaluate_pointing_gestures(csv_path, video_path, target_locations, output_directory):
    # Load the cleaned data CSV file
    df = pd.read_csv(csv_path)

    # Initialize video for visualization
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Step 1: Get the landmarks from the CSV
        row = df.loc[df['frame'] == frame_idx] 
        
        if row.empty:
            frame_idx += 1
            continue

        landmarks = parse_landmarks(row['landmarks_3d'].values[0])
        pointing_count = row['pointing_count'].values[0].astype(int)
        pointing_arm = row['pointing_arm'].values[0]
        gesture_duration = row['gesture_duration'].values[0]
        wrist_location = wrist_location_to_dict(row['wrist_location'].values[0])

        # Step 2: Get the related vectors
        vectors = {
            'eye_to_wrist': to_vector(row['eye_to_wrist'].values[0]),
            'shoulder_to_wrist': to_vector(row['shoulder_to_wrist'].values[0]),
            'elbow_to_wrist': to_vector(row['elbow_to_wrist'].values[0]),
            'nose_to_wrist': to_vector(row['nose_to_wrist'].values[0]),
            
            'eye_to_index': to_vector(row['eye_to_index'].values[0]),
            'shoulder_to_index': to_vector(row['shoulder_to_index'].values[0]),
            'elbow_to_index': to_vector(row['elbow_to_index'].values[0]),
            'nose_to_index': to_vector(row['nose_to_index'].values[0]),
            
            
        }
        # vectors = {
        #     'eye_to_wrist': row['eye_to_wrist'].values[0],
        #     'shoulder_to_wrist': row['shoulder_to_wrist'].values[0],
        #     'elbow_to_wrist': row['elbow_to_wrist'].values[0],
        #     'nose_to_wrist': row['nose_to_wrist'].values[0],
            
        #     'eye_to_index': row['eye_to_index'].values[0],
        #     'shoulder_to_index': row['shoulder_to_index'].values[0],
        #     'elbow_to_index': row['elbow_to_index'].values[0],
        #     'nose_to_index': row['nose_to_index'].values[0],
            
            
        # }
        # Step 3: Calculate the ground plane
        ground_plane = find_ground_plane(landmarks)
        ground_frame = np.array(ground_plane.normal)
        ground_plane = Plane(ground_plane.point, normal = np.array([0, 1, 0]))
        target_origin = ground_plane.point
        
        # transformed_ground_plane = Plane(point = target_origin, normal = np.array([0, 1, 0]))
        # target_frame =np.array([0, 1, 0])
        # rotation_matrix = compute_rotation_matrix(ground_frame, target_frame)
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        # transformed_landmarks = transform_points(rotation_matrix, landmarks_array)
        # transformed_wrist_location = transform_points(rotation_matrix, np.array([[wrist_location.x, wrist_location.y, wrist_location.z]]))
        
        
        # Transform vectors
        # transformed_vectors = {}
        # for name, vec in vectors.items():
        #     transformed_vectors[name] = transform_points(rotation_matrix, np.array([[vec.x, vec.y, vec.z]]))[0]
        transformed_vectors = vectors
        # Step 4: Iterate over all potential target locations to find the closest one based on shoulder-to-wrist
        closest_target = None
        closest_transformed_target = None
        min_angle = float('inf')
        min_distance = float('inf')
        
        for target in target_locations:
            transformed_target_location = target_origin - np.array(target)
            target_plane = Plane(point = np.array(transformed_target_location), normal = [0, 1, 0])
            xy_plane = Plane(point = np.array(transformed_target_location), normal = [0, 0, 1])
            xz_plane = target_plane
            yz_plane = Plane(point = np.array(transformed_target_location), normal = [1, 0, 0])
            # # Step 5: Calculate the ground truth vector (target to wrist vector)
            # ground_truth_vector = calculate_vector(wrist_location, Landmark(*transformed_target_location, None))
            
            # Step 6: Distance and angle calculations for each vector
            # distances = {}
            distances_xy = {}
            distances_xz = {}
            distances_yz = {}
            xy={}
            yz={}
            xz={}
            angles = {}
            cosine_similarities = {}
            vector_intersections_xy = {}
            vector_intersections_yz = {}
            vector_intersections_xz = {}
            if pointing_arm == 'left':
                ground_truth_vector = {
                    'eye_to_wrist': calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_EYE], Landmark(*transformed_target_location, None)),
                    'shoulder_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], Landmark(*transformed_target_location, None)),
                    'elbow_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], Landmark(*transformed_target_location, None)),
                    'nose_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.NOSE], Landmark(*transformed_target_location, None)),
                    'eye_to_index': calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_EYE], Landmark(*transformed_target_location, None)),
                    'shoulder_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], Landmark(*transformed_target_location, None)),
                    'elbow_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW], Landmark(*transformed_target_location, None)),
                    'nose_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.NOSE], Landmark(*transformed_target_location, None)),
                    
                }
                shoulder_to_hip = calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                
                wrist_to_index = calculate_vector(landmarks[mp_pose.PoseLandmark.LEFT_WRIST], landmarks[mp_pose.PoseLandmark.LEFT_INDEX])
            else:
                ground_truth_vector = {
                    'eye_to_wrist': calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_EYE], Landmark(*transformed_target_location, None)),
                    'shoulder_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], Landmark(*transformed_target_location, None)),
                    'elbow_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], Landmark(*transformed_target_location, None)),
                    'nose_to_wrist':calculate_vector(landmarks[mp_pose.PoseLandmark.NOSE], Landmark(*transformed_target_location, None)),
                    'eye_to_index': calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_EYE], Landmark(*transformed_target_location, None)),
                    'shoulder_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], Landmark(*transformed_target_location, None)),
                    'elbow_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW], Landmark(*transformed_target_location, None)),
                    'nose_to_index':calculate_vector(landmarks[mp_pose.PoseLandmark.NOSE], Landmark(*transformed_target_location, None)),
                    
                }
                shoulder_to_hip =calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                wrist_to_index = calculate_vector(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST])
                
            shoulder_to_wrist_vector = np.array([vectors['shoulder_to_wrist'].x, vectors['shoulder_to_wrist'].y, vectors['shoulder_to_wrist'].z])
            shoulder_to_hip_vector = np.array(shoulder_to_hip)
            wrist_flex_angle = calculate_angle(shoulder_to_wrist_vector, wrist_to_index)
            arm_raised_angle = calculate_angle(shoulder_to_wrist_vector, shoulder_to_hip_vector)
            print("arm raised:", arm_raised_angle)
            print("wrist flexed:", wrist_flex_angle)
            for name, vec in transformed_vectors.items():
                # Step 5: Calculate the ground truth vector (target to wrist vector)
            
                # Euclidean distance
                if hasattr(vec,'x'):
                    vec = np.array([vec.x, vec.y, vec.z])
                if hasattr(wrist_location,'x'):
                    line = Line(point = np.array([wrist_location.x, wrist_location.y, wrist_location.z]), direction =  vec)
                else:
                    line = Line(point = np.array(wrist_location[0]), direction = np.array(vec))
                # plane_line_intersection = target_plane.intersect_line(line)
                
                try:
                    xy_plane_intersection = xy_plane.intersect_line(line)
                    xy[name] = np.linalg.norm(xy_plane_intersection - transformed_target_location)
                    x_xy = abs(xy_plane_intersection[0] - transformed_target_location[0])
                    y_xy = abs(xy_plane_intersection[1] - transformed_target_location[2])
                    distances_xy[name] = {'x': x_xy, 'y': y_xy, 'z': 0}
                except:
                    # if it is parallel
                    xy_plane_intersection = None
                    xy[name] = float('inf')
                    distances_xy[name] = None
                try:
                    yz_plane_intersection =  yz_plane.intersect_line(line)
                    yz[name] = np.linalg.norm(yz_plane_intersection - transformed_target_location)
                    y_yz = abs(yz_plane_intersection[1] - transformed_target_location[0])
                    z_yz = abs(yz_plane_intersection[2] - transformed_target_location[2])
                    distances_yz[name] = {'x': 0, 'y': y_yz, 'z': z_yz}
                except:
                    yz_plane_intersection = None
                    yz[name] = float('inf')
                    distances_yz[name] = None
                try:
                    xz_plane_intersection =  xz_plane.intersect_line(line)
                    xz[name] = np.linalg.norm(xz_plane_intersection - transformed_target_location)
                    x_xz = abs(xz_plane_intersection[0] - transformed_target_location[0])
                    z_xz = abs(xz_plane_intersection[2] - transformed_target_location[2])
                    distances_xz[name] = {'x': x_xz, 'y': 0,'z': z_xz}
                    
                except:
                    xz_plane_intersection = None
                    xz[name] = float('inf')
                    distances_xz[name] = None
                

                
                vector_intersections_xy[name] = xy_plane_intersection
                vector_intersections_yz[name] = yz_plane_intersection
                vector_intersections_xz[name] = xz_plane_intersection
                
                
                
                # xz[name] = np.linalg.norm(xz_plane_intersection - transformed_target_location)
                # xy[name] = np.linalg.norm(xy_plane_intersection - transformed_target_location)
                # yz[name] = np.linalg.norm(yz_plane_intersection - transformed_target_location)
                
                
                # x_xz = abs(xz_plane_intersection[0] - transformed_target_location[0])
                # z_xz = abs(xz_plane_intersection[2] - transformed_target_location[2])
                # distances_xz[name] = {'x': x_xz, 'y': 0,'z': z_xz}
                
                # x_xy = abs(xy_plane_intersection[0] - transformed_target_location[0])
                # y_xy = abs(xy_plane_intersection[1] - transformed_target_location[2])
                # distances_xy[name] = {'x': x_xy, 'y': y_xy, 'z': 0}
                
                # y_yz = abs(yz_plane_intersection[1] - transformed_target_location[0])
                # z_yz = abs(yz_plane_intersection[2] - transformed_target_location[2])
                # distances_yz[name] = {'x': 0, 'y': y_yz, 'z': z_yz}
                

                # Angle and cosine similarity
                cosine_similarity = np.dot(ground_truth_vector[name], vec) / (np.linalg.norm(ground_truth_vector[name]) * np.linalg.norm(vec))
                cosine_similarities[name] = cosine_similarity
                angle = np.degrees(np.arccos(np.clip(cosine_similarity, -1.0, 1.0)))  # Convert to degrees
                angles[name] = angle
            
            pairs = [
                    ('eye_to_wrist', 'eye_to_index'),
                    ('shoulder_to_wrist', 'shoulder_to_index'),
                    ('elbow_to_wrist', 'elbow_to_index'),
                    ('nose_to_wrist', 'nose_to_index')
                ]
            
            distance_diff = {}
            angle_diff = {}
            for pair in pairs:
                vec1_name, vec2_name = pair
                
                # Get vectors for the pair
                vec1 = transformed_vectors[vec1_name]
                vec2 = transformed_vectors[vec2_name]

                # Extend vectors to the ground plane
                intersect_1 = ground_plane.intersect_line(
        Line(point=[wrist_location.x, wrist_location.y, wrist_location.z], direction=[vec1.x, vec1.y, vec1.z])
    )
                intersect_2 = ground_plane.intersect_line(
        Line(point=[wrist_location.x, wrist_location.y, wrist_location.z], direction=[vec2.x, vec2.y, vec2.z])
    )
                intersection_vec1 = intersect_1
                intersection_vec2 = intersect_2
                
                # Compute distance difference on the ground plane
                if intersection_vec1 is not None and intersection_vec2 is not None:
                    distance_diff[vec1_name] = calculate_distance(intersection_vec1, intersection_vec2)
                else:
                    distance_diff[vec1_name] = float('inf')  # Handle parallel cases or missing intersections

                # Compute angle between vectors
                vec1_array = np.array([vec1.x, vec1.y, vec1.z])
                vec2_array = np.array([vec2.x, vec2.y, vec2.z])
                angle_diff[vec1_name] = calculate_angle(vec1_array, vec2_array)
            print("angle ->", angle_diff)
            print("dist ->",distance_diff)
            # Check if the shoulder-to-wrist vector provides the closest target
            if min(min(xz.values()), min(yz.values()), min(xy.values())) < min_distance:
                min_distance = min(min(xz.values()), min(yz.values()), min(xy.values()))
                min_angle = min(angles.values())
                closest_target = target
                closest_transformed_target = transformed_target_location
                best_cos_sim = cosine_similarities
                best_angles = angles
                best_xz_intersection = vector_intersections_xz
                best_yz_intersection = vector_intersections_yz
                best_xy_intersection = vector_intersections_xy
                # best_dist = [xz_plane_distance, yz_plane_distance, xy_plane_distance]
                best_dist_xz = distances_xz
                best_dist_yz = distances_yz
                best_dist_xy = distances_xy
            

        # Store results for the closest target
        results.append({
            'frame': frame_idx,
            'pointing_count': pointing_count,
            'gesture_duration': gesture_duration,
            'target_location': closest_target,
            'min_distance_to_target': min_distance, 
            'angle_to_target':min_angle,
            
            'arm_raised_angle':arm_raised_angle,
            'wrist_flex_angle':wrist_flex_angle,
            'distance_diff_wrist_finger-eye': distance_diff['eye_to_wrist'], 
            'angle_wrist_finger-eye': angle_diff['eye_to_wrist'],  
            'distance_diff_wrist_finger-shoulder': distance_diff['shoulder_to_wrist'], 
            'angle_wrist_finger-shoulder': angle_diff['shoulder_to_wrist'],  
            'distance_diff_wrist_finger-eblow': distance_diff['elbow_to_wrist'], 
            'angle_wrist_finger-elbow': angle_diff['elbow_to_wrist'],  
            'distance_diff_wrist_finger-nose': distance_diff['nose_to_wrist'], 
            'angle_wrist_finger-nose': angle_diff['nose_to_wrist'],  
            
            'xz_distance_to_target(ground)': min(xz.values()),
            'yz_distance_to_target': min(yz.values()),
            'xy_distance_to_target': min(xy.values()),
            
            'distance_xz_eye_to_wrist': best_dist_xz['eye_to_wrist'],
            'distance_xz_eye_to_index': best_dist_xz['eye_to_index'],
            'distance_xz_shoulder_to_wrist': best_dist_xz['shoulder_to_wrist'],
            'distance_xz_shoulder_to_index': best_dist_xz['shoulder_to_index'],
            'distance_xz_elbow_to_wrist': best_dist_xz['elbow_to_wrist'],
            'distance_xz_elbow_to_index': best_dist_xz['elbow_to_index'],
            'distance_xz_nose_to_wrist': best_dist_xz['nose_to_wrist'],
            'distance_xz_nose_to_index': best_dist_xz['nose_to_index'],
            
            'distance_yz_eye_to_wrist': best_dist_yz['eye_to_wrist'],
            'distance_yz_eye_to_index': best_dist_yz['eye_to_index'],
            'distance_yz_shoulder_to_wrist': best_dist_yz['shoulder_to_wrist'],
            'distance_yz_shoulder_to_index': best_dist_yz['shoulder_to_index'],
            'distance_yz_elbow_to_wrist': best_dist_yz['elbow_to_wrist'],
            'distance_yz_elbow_to_index': best_dist_yz['elbow_to_index'],
            'distance_yz_nose_to_wrist': best_dist_yz['nose_to_wrist'],
            'distance_yz_nose_to_index': best_dist_yz['nose_to_index'],
            
            'distance_xy_eye_to_wrist': best_dist_xy['eye_to_wrist'],
            'distance_xy_eye_to_index': best_dist_xy['eye_to_index'],
            'distance_xy_shoulder_to_wrist': best_dist_xy['shoulder_to_wrist'],
            'distance_xy_shoulder_to_index': best_dist_xy['shoulder_to_index'],
            'distance_xy_elbow_to_wrist': best_dist_xy['elbow_to_wrist'],
            'distance_xy_elbow_to_index': best_dist_xy['elbow_to_index'],
            'distance_xy_nose_to_wrist': best_dist_xy['nose_to_wrist'],
            'distance_xy_nose_to_index': best_dist_xy['nose_to_index'],
            
            'angles_eye_to_wrist': best_angles['eye_to_wrist'],
            'angles_shoulder_to_wrist': best_angles['shoulder_to_wrist'],
            'angles_elbow_to_wrist': best_angles['elbow_to_wrist'],
            'angles_nose_to_wrist': best_angles['nose_to_wrist'],
            'cosine_similarities_eye_to_wrist': best_cos_sim['eye_to_wrist'],
            'cosine_similarities_shoulder_to_wrist': best_cos_sim['shoulder_to_wrist'],
            'cosine_similarities_elbow_to_wrist': best_cos_sim['elbow_to_wrist'],
            'cosine_similarities_nose_to_wrist': best_cos_sim['nose_to_wrist']
        })

        
        # Optional: Visualization for debugging
        # visualize_landmarks_and_target(transformed_landmarks, transformed_ground_plane, closest_transformed_target, transformed_wrist_location, best_intersection, frame_idx, output_directory)
        visualize_landmarks_and_target(landmarks_array, ground_plane, closest_transformed_target, wrist_location, vector_intersections_xz, frame_idx, output_directory)


        frame_idx += 1

    # Save the results to a new CSV file
    output_file = os.path.join(output_directory, 'evaluation_results.csv')
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()
    print("=====Evaluation complete =====")

def main():    
     # Parse arguments
    parser = argparse.ArgumentParser(description="Pointing Gesture Evaluation")
    parser.add_argument('--video_path', type=str, help="Path to the video file for evaluation")
    parser.add_argument('--csv_path', type=str,  help="Path to the CSV file containing gesture data")
    parser.add_argument('--target_locations', type=float, nargs='+', help="List of target locations in the format x1 y1 z1 x2 y2 z2 ...")
    args = parser.parse_args()
    if not args.video_path:
        csv_path = "/Users/ivy/Desktop/spot_gesture_eval/3m_video_updated_cleanup.csv"
        video_path = "/Users/ivy/Desktop/spot_gesture_eval/videos/3m_video.mp4"
        target_locations = [[-1, .46, 3],[1, .74, 3]]
        output_directory = "/Users/ivy/Desktop/spot_gesture_eval/videos/3m_video/"
        evaluate_pointing_gestures(csv_path, video_path, target_locations, output_directory)
    else:
        # Parse the target locations into a list of 3D points
        target_locations = [args.target_locations[i:i+3] for i in range(0, len(args.target_locations), 3)]
        # output_directory = os.path.dirname(args.csv_path) 
        output_directory = args.video_path[0:-4]+'/'
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        # Evaluate pointing gestures with multiple targets
        evaluate_pointing_gestures(args.csv_path, args.video_path, target_locations, output_directory)
        
    # left target
    t1_2m = [-1, .46, 2]
    # right target
    t2_2m = [1, .74, 2]

    
    
    # # sample run: python pointing_eval.py --video_path /Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4 --csv_path /Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv --target_locations -1 2 0.46 1 2 0.74
    # video_path = '/Users/ivy/Desktop/spot_gesture_eval/2m_video.mp4'
    # csv_path = '/Users/ivy/Desktop/spot_gesture_eval/cleaned_2m_data.csv'
    # target_location = t1_2m
    
    
    # # Evaluate pointing gestures
    # evaluate_pointing_gestures(csv_path, video_path, target_location)

if __name__ == "__main__":
    main()
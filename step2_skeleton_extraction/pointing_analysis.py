"""
Pointing gesture analysis module.

Calculates ground plane intersections, head orientation, and distances to targets.
Based on legacy gesture_data_process.py logic.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


def compute_head_orientation(landmarks_3d: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute head orientation vector from facial landmarks.

    Args:
        landmarks_3d: 33 MediaPipe landmarks in 3D [[x, y, z], ...]

    Returns:
        head_orientation_vector: Normalized 3D vector indicating head facing direction
        head_orientation_origin: 3D point at nose (origin of orientation vector)
    """
    # MediaPipe indices
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_MOUTH = 9
    RIGHT_MOUTH = 10
    NOSE = 0

    left_eye = np.array(landmarks_3d[LEFT_EYE])
    right_eye = np.array(landmarks_3d[RIGHT_EYE])
    left_mouth = np.array(landmarks_3d[LEFT_MOUTH])
    right_mouth = np.array(landmarks_3d[RIGHT_MOUTH])
    nose = np.array(landmarks_3d[NOSE])

    # Compute vectors from eyes and mouth to nose
    left_eye_vec = nose - left_eye
    right_eye_vec = nose - right_eye
    eye_vec = (left_eye_vec + right_eye_vec) / 2

    left_mouth_vec = nose - left_mouth
    right_mouth_vec = nose - right_mouth
    mouth_vec = (left_mouth_vec + right_mouth_vec) / 2

    # Average eye and mouth vectors
    head_orientation_vector = (eye_vec + mouth_vec) / 2

    # Normalize
    head_orientation_vector = head_orientation_vector / np.linalg.norm(head_orientation_vector)

    head_orientation_origin = nose

    return head_orientation_vector, head_orientation_origin


def compute_ground_intersection(origin: np.ndarray, vector: np.ndarray,
                                ground_plane_y: float = 0.0) -> Optional[np.ndarray]:
    """
    Compute intersection of a ray with the ground plane (y=0 after transformation).

    Args:
        origin: 3D point where the vector originates (e.g., wrist or nose)
        vector: 3D direction vector (normalized or not)
        ground_plane_y: Y-coordinate of ground plane (default 0.0 after transformation)

    Returns:
        intersection: 3D point where vector intersects ground plane, or None if parallel
    """
    origin = np.array(origin)
    vector = np.array(vector)

    # Check if vector is not parallel to ground (y component must be non-zero)
    if vector[1] == 0:
        return None

    # Calculate scale factor to reach y=ground_plane_y
    # origin + scale * vector = [x, ground_plane_y, z]
    # origin[1] + scale * vector[1] = ground_plane_y
    # scale = (ground_plane_y - origin[1]) / vector[1]

    # Since we want to go FROM origin in direction of vector to reach ground:
    # We need to find t such that origin - t * vector has y = 0
    # origin[1] - t * vector[1] = 0
    # t = origin[1] / vector[1]

    scale = origin[1] / vector[1]
    intersection = origin - vector * scale

    return intersection


def compute_distances_to_targets(point: np.ndarray, targets: List[Dict]) -> List[float]:
    """
    Compute Euclidean distances from a point to each target.

    Args:
        point: 3D point (e.g., ground intersection)
        targets: List of target dictionaries with 'position_m' key OR 'x', 'y', 'z' keys

    Returns:
        distances: List of distances to each target
    """
    distances = []
    point = np.array(point)

    for target in targets:
        # Support both formats: legacy 'position_m' or new 'x', 'y', 'z'
        if 'position_m' in target:
            target_pos = np.array(target['position_m'])
        elif 'x' in target and 'y' in target and 'z' in target:
            target_pos = np.array([target['x'], target['y'], target['z']])
        else:
            raise ValueError(f"Target must have either 'position_m' or 'x'/'y'/'z' keys: {target}")

        dist = np.linalg.norm(point - target_pos)
        distances.append(dist)

    return distances


def analyze_pointing_frame(result, targets: List[Dict],
                           pointing_arm: str = 'right') -> Dict:
    """
    Analyze a single frame for pointing gesture metrics.

    Args:
        result: DetectionResult object with landmarks_3d and arm_vectors
        targets: List of target dictionaries (already transformed to ground plane frame)
        pointing_arm: 'left' or 'right'

    Returns:
        analysis: Dictionary containing:
            - wrist_location: [x, y, z]
            - eye_to_wrist_ground_intersection: [x, y, z]
            - shoulder_to_wrist_ground_intersection: [x, y, z]
            - elbow_to_wrist_ground_intersection: [x, y, z]
            - nose_to_wrist_ground_intersection: [x, y, z]
            - head_orientation_ground_intersection: [x, y, z]
            - eye_to_wrist_vec: [x, y, z]
            - eye_to_wrist_dist_to_target_1-4: float
            - shoulder_to_wrist_vec: [x, y, z]
            - shoulder_to_wrist_dist_to_target_1-4: float
            - elbow_to_wrist_vec: [x, y, z]
            - elbow_to_wrist_dist_to_target_1-4: float
            - nose_to_wrist_vec: [x, y, z]
            - nose_to_wrist_dist_to_target_1-4: float
            - head_orientation_dist_to_target_1-4: float
            - head_orientation_vector: [x, y, z]
            - head_orientation_origin: [x, y, z]
    """
    analysis = {}

    if not result or not result.landmarks_3d:
        return None

    landmarks_3d = result.landmarks_3d

    # Get wrist index based on pointing arm
    if pointing_arm.lower() == 'left':
        WRIST_INDEX = 15
    else:  # right
        WRIST_INDEX = 16

    wrist_location = landmarks_3d[WRIST_INDEX]
    analysis['wrist_location'] = wrist_location

    # Process arm vectors
    arm_vectors = result.arm_vectors
    if arm_vectors:
        vector_names = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']

        for vec_name in vector_names:
            if vec_name in arm_vectors and arm_vectors[vec_name]:
                vec = np.array(arm_vectors[vec_name])

                # Store vector
                analysis[f'{vec_name}_vec'] = vec.tolist()

                # Compute ground intersection
                intersection = compute_ground_intersection(wrist_location, vec)

                if intersection is not None:
                    analysis[f'{vec_name}_ground_intersection'] = intersection.tolist()

                    # Compute distances to each target
                    distances = compute_distances_to_targets(intersection, targets)
                    for i, dist in enumerate(distances, 1):
                        analysis[f'{vec_name}_dist_to_target_{i}'] = dist
                else:
                    analysis[f'{vec_name}_ground_intersection'] = [None, None, None]
                    for i in range(1, 5):
                        analysis[f'{vec_name}_dist_to_target_{i}'] = None

    # Compute head orientation
    try:
        head_vec, head_origin = compute_head_orientation(landmarks_3d)
        analysis['head_orientation_vector'] = head_vec.tolist()
        analysis['head_orientation_origin'] = head_origin.tolist()

        # Compute head orientation ground intersection
        head_intersection = compute_ground_intersection(head_origin, head_vec)

        if head_intersection is not None:
            analysis['head_orientation_ground_intersection'] = head_intersection.tolist()

            # Compute distances to targets
            head_distances = compute_distances_to_targets(head_intersection, targets)
            for i, dist in enumerate(head_distances, 1):
                analysis[f'head_orientation_dist_to_target_{i}'] = dist
        else:
            analysis['head_orientation_ground_intersection'] = [None, None, None]
            for i in range(1, 5):
                analysis[f'head_orientation_dist_to_target_{i}'] = None
    except Exception as e:
        print(f"Warning: Could not compute head orientation: {e}")
        analysis['head_orientation_vector'] = [None, None, None]
        analysis['head_orientation_origin'] = [None, None, None]
        analysis['head_orientation_ground_intersection'] = [None, None, None]
        for i in range(1, 5):
            analysis[f'head_orientation_dist_to_target_{i}'] = None

    # Set ground_intersection to eye_to_wrist intersection (default)
    analysis['ground_intersection'] = analysis.get('eye_to_wrist_ground_intersection', [None, None, None])

    return analysis

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

    # Find t such that origin + t * vector has y = ground_plane_y
    # origin[1] + t * vector[1] = ground_plane_y
    # t = (ground_plane_y - origin[1]) / vector[1]
    scale = (ground_plane_y - origin[1]) / vector[1]
    intersection = origin + vector * scale

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


def _rotate_targets(targets: List[Dict], R: np.ndarray) -> List[Dict]:
    """Rotate target positions by R, returning new target dicts."""
    rotated = []
    for t in targets:
        t_copy = dict(t)
        if 'x' in t and 'y' in t and 'z' in t:
            pos = R @ np.array([t['x'], t['y'], t['z']])
            t_copy['x'], t_copy['y'], t_copy['z'] = pos.tolist()
        if 'position_m' in t:
            pos = R @ np.array(t['position_m'])
            t_copy['position_m'] = pos.tolist()
        rotated.append(t_copy)
    return rotated


# Outlier filtering thresholds for ground intersections
# Based on config/targets.yaml:
# - Targets arranged in CURVED ARC at X: -1.06 to +1.16, Z: ~2.6-2.9m
# - Dog at CENTER of arc (inside the curve)
# - Human on OUTSIDE of curve (pointing toward targets/dog)
# Camera frame: +X = camera right (human's left), +Y = down, +Z = depth
INTERSECTION_BOUNDS = {
    'z_min': 1.5,    # Minimum depth (meters) - human may be closer
    'z_max': 6.0,    # Maximum depth (meters) - extended for human behind targets
    'x_min': -2.5,   # Minimum X (meters) - targets from -1.06 to +1.16 + margin
    'x_max': 2.5,    # Maximum X (meters)
    'max_dist_to_nearest_target': 3.0,  # Max distance to nearest target (meters)
}


def is_valid_intersection(intersection: np.ndarray,
                          targets: List[Dict],
                          wrist_z: float,
                          bounds: Dict = None) -> bool:
    """
    Check if a ground intersection point is valid (not an outlier).

    Filters out obviously bad pointing data like:
    - Intersections too far away (> 10m depth)
    - Intersections behind the person
    - Intersections too far from any target

    Args:
        intersection: [x, y, z] ground intersection point
        targets: List of target dictionaries
        wrist_z: Z depth of the wrist (for sanity check)
        bounds: Optional custom bounds dictionary

    Returns:
        True if valid, False if outlier
    """
    if intersection is None:
        return False

    if bounds is None:
        bounds = INTERSECTION_BOUNDS

    x, y, z = intersection

    # Check Z bounds (depth)
    if z < bounds['z_min'] or z > bounds['z_max']:
        return False

    # Check X bounds (lateral position)
    if x < bounds['x_min'] or x > bounds['x_max']:
        return False

    # Check if intersection is not too far behind the wrist
    # (pointing backward would give intersection behind person)
    if z < wrist_z - 1.0:  # Allow 1m tolerance
        return False

    # Check distance to nearest target
    if targets:
        min_dist = float('inf')
        for target in targets:
            if 'x' in target and 'z' in target:
                tx, tz = target['x'], target['z']
                # Use X-Z distance (ground plane distance)
                dist = np.sqrt((x - tx)**2 + (z - tz)**2)
                min_dist = min(min_dist, dist)

        if min_dist > bounds['max_dist_to_nearest_target']:
            return False

    return True


def analyze_pointing_frame(result, targets: List[Dict],
                           pointing_arm: str = 'right',
                           ground_plane_rotation: Optional[np.ndarray] = None,
                           filter_outliers: bool = True) -> Dict:
    """
    Analyze a single frame for pointing gesture metrics.

    If ground_plane_rotation (R) is provided, all 3D data (landmarks, vectors,
    targets) are rotated into a ground-aligned frame where y=0 is the actual
    ground surface before computing ray-ground intersections and distances.

    Args:
        result: DetectionResult object with landmarks_3d and arm_vectors
        targets: List of target dictionaries with 'x', 'y', 'z' keys
        pointing_arm: 'left' or 'right'
        ground_plane_rotation: (3,3) rotation matrix from ground_plane_correction,
            or None to skip (uses raw camera coordinates)
        filter_outliers: If True, filter out obviously bad intersection points
            (e.g., > 10m depth, too far from targets)

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
            - outliers_filtered: int (count of filtered outlier intersections)
    """
    analysis = {}

    if not result or not result.landmarks_3d:
        return None

    landmarks_3d = result.landmarks_3d
    arm_vectors = result.arm_vectors

    R = ground_plane_rotation

    # Compute ground_y from target positions FIRST (before any rotation)
    # This ensures we always intersect at the correct Y level
    target_ys = []
    for t in targets:
        if 'y' in t:
            target_ys.append(t['y'])
    ground_y = float(np.mean(target_ys)) if target_ys else 0.0

    # Apply ground plane rotation so the ground surface is horizontal
    if R is not None:
        landmarks_3d = [tuple((R @ np.array(p)).tolist()) for p in landmarks_3d]
        targets = _rotate_targets(targets, R)
        if arm_vectors:
            arm_vectors = {
                k: (R @ np.array(v)).tolist() if v is not None else None
                for k, v in arm_vectors.items()
                if k != 'wrist_location'
            }
            # Rotate wrist_location as a point
            if result.arm_vectors.get('wrist_location') is not None:
                arm_vectors['wrist_location'] = (R @ np.array(result.arm_vectors['wrist_location'])).tolist()

        # Update ground_y from rotated targets (they should all be ~same y after rotation)
        target_ys = []
        for t in targets:
            if 'y' in t:
                target_ys.append(t['y'])
        if target_ys:
            ground_y = float(np.mean(target_ys))

    # Get wrist index based on pointing arm
    if pointing_arm.lower() == 'left':
        WRIST_INDEX = 15
    else:  # right
        WRIST_INDEX = 16

    wrist_location = landmarks_3d[WRIST_INDEX]
    analysis['wrist_location'] = wrist_location
    wrist_z = wrist_location[2] if len(wrist_location) > 2 else 0.0

    # Track outliers filtered
    outliers_filtered = 0

    # Process arm vectors
    if arm_vectors:
        vector_names = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']

        for vec_name in vector_names:
            if vec_name in arm_vectors and arm_vectors[vec_name]:
                vec = np.array(arm_vectors[vec_name])

                # Store vector
                analysis[f'{vec_name}_vec'] = vec.tolist()

                # Compute ground intersection
                intersection = compute_ground_intersection(wrist_location, vec, ground_plane_y=ground_y)

                # Validate intersection (filter outliers)
                is_valid = True
                if filter_outliers and intersection is not None:
                    is_valid = is_valid_intersection(intersection, targets, wrist_z)
                    if not is_valid:
                        outliers_filtered += 1

                if intersection is not None and is_valid:
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

        # head_vec and head_origin are already in the rotated frame since
        # landmarks_3d was rotated above
        analysis['head_orientation_vector'] = head_vec.tolist()
        analysis['head_orientation_origin'] = head_origin.tolist()

        # Compute head orientation ground intersection
        head_intersection = compute_ground_intersection(head_origin, head_vec, ground_plane_y=ground_y)

        # Validate head intersection (filter outliers)
        head_is_valid = True
        if filter_outliers and head_intersection is not None:
            # Use head origin Z for validation instead of wrist
            head_origin_z = head_origin[2] if len(head_origin) > 2 else 0.0
            head_is_valid = is_valid_intersection(head_intersection, targets, head_origin_z)
            if not head_is_valid:
                outliers_filtered += 1

        if head_intersection is not None and head_is_valid:
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

    # Record number of outliers filtered
    analysis['outliers_filtered'] = outliers_filtered

    return analysis

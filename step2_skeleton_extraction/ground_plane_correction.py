"""
Ground plane correction module.

Computes transformation to align the ground plane (where targets are) to horizontal.

IMPORTANT: This module should ONLY correct for camera tilt (rotation around X-axis).
It should NOT distort the X-Z arrangement of targets, which may be on a curved arc.

Experiment setup:
- Targets are arranged in a CURVED ARC (not a flat line)
- Dog is at the CENTER (inside) of the arc
- Human is on the OUTSIDE of the curve
- Camera views from outside the arc toward the targets and dog
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


def fit_plane_to_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a plane to 3D points using SVD.
    
    Args:
        points: (N, 3) array of points
        
    Returns:
        (normal, centroid): 
            normal: (3,) unit normal vector
            centroid: (3,) plane centroid
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # SVD to find plane normal (smallest singular value)
    U, S, Vt = np.linalg.svd(centered)
    normal = Vt[2, :]  # Last row of Vt is the normal
    
    # Make sure normal points UP (positive Y component in camera frame)
    # Camera Y points down, so ground normal should be negative Y
    if normal[1] > 0:
        normal = -normal
        
    return normal, centroid


def compute_rotation_to_horizontal(plane_normal: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix to align a plane to horizontal.
    
    In camera frame:
    - Horizontal plane has normal (0, -1, 0) (Y points down)
    - We want to rotate plane_normal to align with (0, -1, 0)
    
    Args:
        plane_normal: (3,) current plane normal
        
    Returns:
        R: (3, 3) rotation matrix
    """
    # Target: horizontal plane normal (Y-axis, pointing down)
    target_normal = np.array([0, -1, 0])
    
    # Normalize input
    v1 = plane_normal / np.linalg.norm(plane_normal)
    v2 = target_normal
    
    # Rotation axis: cross product
    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    
    # Check if already aligned
    if axis_norm < 1e-6:
        # Already aligned or opposite
        cos_angle = np.dot(v1, v2)
        if cos_angle > 0:
            return np.eye(3)  # Already aligned
        else:
            # 180 degree rotation - pick arbitrary perpendicular axis
            axis = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 0, 1])
            return rotation_matrix_from_axis_angle(axis, np.pi)
    
    axis = axis / axis_norm
    
    # Rotation angle
    cos_angle = np.dot(v1, v2)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return rotation_matrix_from_axis_angle(axis, angle)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute rotation matrix from axis-angle representation (Rodrigues' formula).
    
    Args:
        axis: (3,) unit rotation axis
        angle: rotation angle in radians
        
    Returns:
        R: (3, 3) rotation matrix
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def apply_transform_to_points(points: List[Tuple], rotation: np.ndarray) -> List[Tuple]:
    """
    Apply rotation to list of 3D points.
    
    Args:
        points: List of (x, y, z) tuples
        rotation: (3, 3) rotation matrix
        
    Returns:
        List of transformed (x, y, z) tuples
    """
    result = []
    for p in points:
        if len(p) == 3 and not (p[0] == 0 and p[1] == 0 and p[2] == 0):
            # Valid point
            p_rotated = rotation @ np.array(p)
            result.append(tuple(p_rotated.tolist()))
        else:
            # Invalid point, keep as is
            result.append(p)
    return result


def compute_ground_plane_transform(targets: List[Dict]) -> Optional[np.ndarray]:
    """
    Compute transformation to align ground plane (from targets) to horizontal.
    
    Args:
        targets: List of target dicts with 'x', 'y', 'z' keys
        
    Returns:
        R: (3, 3) rotation matrix, or None if cannot compute
    """
    if not targets or len(targets) < 3:
        return None
        
    # Extract target positions
    target_positions = []
    for t in targets:
        if 'x' in t and 'y' in t and 'z' in t:
            target_positions.append([t['x'], t['y'], t['z']])
            
    if len(target_positions) < 3:
        return None
        
    points = np.array(target_positions)
    
    # Fit plane
    normal, centroid = fit_plane_to_points(points)
    
    # Compute rotation to horizontal
    R = compute_rotation_to_horizontal(normal)
    
    return R


def get_transform_info(R: np.ndarray, plane_normal: np.ndarray) -> Dict:
    """
    Get human-readable info about the transformation.

    Args:
        R: rotation matrix
        plane_normal: original plane normal

    Returns:
        Dict with transformation info
    """
    # Calculate rotation angle
    horizontal_normal = np.array([0, -1, 0])
    cos_angle = np.dot(plane_normal, horizontal_normal)
    angle_rad = np.arccos(np.clip(np.abs(cos_angle), 0, 1))
    angle_deg = angle_rad * 180 / np.pi

    # Rotation axis
    axis = np.cross(plane_normal, horizontal_normal)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([0, 0, 0])

    return {
        'angle_deg': angle_deg,
        'angle_rad': angle_rad,
        'axis': axis.tolist(),
        'plane_normal': plane_normal.tolist(),
        'rotation_matrix': R.tolist()
    }


def compute_tilt_only_transform(targets: List[Dict]) -> Optional[np.ndarray]:
    """
    Compute a TILT-ONLY correction that only rotates around the X-axis.

    This preserves the X-Z spatial arrangement of targets (the arc shape)
    while correcting for camera tilt. This is preferred over full ground
    plane rotation when targets are arranged in a curved arc.

    The tilt angle is estimated from the average Y-coordinate variation
    with depth (Z).

    Args:
        targets: List of target dicts with 'x', 'y', 'z' keys

    Returns:
        R: (3, 3) rotation matrix (X-axis rotation only), or None
    """
    if not targets or len(targets) < 2:
        return None

    # Extract positions
    positions = []
    for t in targets:
        if 'x' in t and 'y' in t and 'z' in t:
            positions.append([t['x'], t['y'], t['z']])

    if len(positions) < 2:
        return None

    points = np.array(positions)

    # Estimate tilt angle from Y vs Z correlation
    # If camera is tilted, Y will vary linearly with Z
    z_vals = points[:, 2]
    y_vals = points[:, 1]

    # Linear regression: y = a*z + b
    # Tilt angle = arctan(a)
    z_mean = z_vals.mean()
    y_mean = y_vals.mean()

    # Covariance and variance
    cov_zy = np.sum((z_vals - z_mean) * (y_vals - y_mean))
    var_z = np.sum((z_vals - z_mean) ** 2)

    if var_z < 1e-6:
        # All targets at same depth - can't estimate tilt
        return np.eye(3)

    slope = cov_zy / var_z
    tilt_angle = np.arctan(slope)

    # If tilt is very small, don't bother
    if abs(tilt_angle) < 0.01:  # ~0.5 degrees
        return np.eye(3)

    # Rotation around X-axis to correct tilt
    # R_x(theta) rotates Y toward Z when positive
    cos_t = np.cos(-tilt_angle)
    sin_t = np.sin(-tilt_angle)

    R = np.array([
        [1, 0, 0],
        [0, cos_t, -sin_t],
        [0, sin_t, cos_t]
    ])

    return R


def is_arc_arrangement(targets: List[Dict], threshold: float = 0.15) -> bool:
    """
    Check if targets are arranged in an arc (curved) rather than a flat line.

    Targets in an arc will have varying Z (depth) values - edges closer,
    middle further (or vice versa).

    Args:
        targets: List of target dicts with 'x', 'y', 'z' keys
        threshold: Z variation threshold (meters) to consider as arc

    Returns:
        True if targets appear to be in an arc arrangement
    """
    if not targets or len(targets) < 3:
        return False

    # Extract Z values sorted by X position
    positions = []
    for t in targets:
        if 'x' in t and 'z' in t:
            positions.append((t['x'], t['z']))

    if len(positions) < 3:
        return False

    # Sort by X
    positions.sort(key=lambda p: p[0])

    z_vals = [p[1] for p in positions]

    # Check if middle Z values are different from edge Z values
    # (characteristic of an arc)
    edge_z_avg = (z_vals[0] + z_vals[-1]) / 2
    middle_z_avg = np.mean(z_vals[1:-1]) if len(z_vals) > 2 else z_vals[len(z_vals)//2]

    z_diff = abs(middle_z_avg - edge_z_avg)

    return z_diff > threshold

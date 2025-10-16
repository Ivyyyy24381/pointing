#!/usr/bin/env python3
"""
Visualize 3D skeleton data from skeleton_2d.json output.

Usage:
    python visualize_skeleton_3d.py trial_output/trial_1/single_camera/skeleton_2d.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path


def load_skeleton_data(json_path: str) -> dict:
    """Load skeleton JSON file."""
    with open(json_path) as f:
        return json.load(f)


def plot_skeleton_3d(landmarks_3d, arm_vectors=None, frame_name="", targets=None, show=True, ax=None, head_orientation=None):
    """
    Plot 3D skeleton with arm vectors, head orientation, and targets.

    Args:
        landmarks_3d: List of [x, y, z] coordinates
        arm_vectors: Dictionary of arm vectors
        frame_name: Frame identifier for title
        targets: List of target dictionaries with x, y, z, label keys
        show: Whether to call plt.show() (default True)
        ax: Optional matplotlib 3D axis to plot on. If None, creates new figure.
        head_orientation: Dictionary with 'head_orientation_vector' and 'head_orientation_origin' keys
    """
    landmarks = np.array(landmarks_3d)

    # MediaPipe Pose connections (simplified)
    connections = [
        # Face
        (0, 1), (1, 2), (2, 3),  # nose to left eye
        (0, 4), (4, 5), (5, 6),  # nose to right eye
        # Torso
        (11, 12),  # shoulders
        (11, 23), (12, 24),  # shoulders to hips
        (23, 24),  # hips
        # Left arm
        (11, 13), (13, 15),  # shoulder -> elbow -> wrist
        # Right arm
        (12, 14), (14, 16),  # shoulder -> elbow -> wrist
        # Left leg
        (23, 25), (25, 27), (27, 29), (27, 31),  # hip -> knee -> ankle -> foot
        # Right leg
        (24, 26), (26, 28), (28, 30), (28, 32),  # hip -> knee -> ankle -> foot
    ]

    # Create figure only if ax not provided
    if ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Plot skeleton connections
    for start, end in connections:
        if start < len(landmarks) and end < len(landmarks):
            start_pt = landmarks[start]
            end_pt = landmarks[end]

            # Skip if either point is invalid
            if np.all(start_pt == 0) or np.all(end_pt == 0):
                continue

            xs = [start_pt[0], end_pt[0]]
            ys = [start_pt[1], end_pt[1]]
            zs = [start_pt[2], end_pt[2]]
            ax.plot(xs, ys, zs, 'b-', linewidth=2)

    # Plot landmarks as points with color coding for left/right
    # MediaPipe landmark indices - left side landmarks
    left_indices = [1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
    # Right side landmarks
    right_indices = [4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # Center landmarks (nose, mouth)
    center_indices = [0, 9, 10]

    # Plot left side landmarks in red
    for idx in left_indices:
        if idx < len(landmarks):
            point = landmarks[idx]
            if not np.all(point == 0):
                ax.scatter([point[0]], [point[1]], [point[2]],
                          c='red', marker='o', s=80, edgecolors='darkred', linewidths=1.5,
                          label='LEFT side' if idx == left_indices[0] else '')

    # Plot right side landmarks in blue
    for idx in right_indices:
        if idx < len(landmarks):
            point = landmarks[idx]
            if not np.all(point == 0):
                ax.scatter([point[0]], [point[1]], [point[2]],
                          c='blue', marker='o', s=80, edgecolors='darkblue', linewidths=1.5,
                          label='RIGHT side' if idx == right_indices[0] else '')

    # Plot center landmarks in yellow
    for idx in center_indices:
        if idx < len(landmarks):
            point = landmarks[idx]
            if not np.all(point == 0):
                ax.scatter([point[0]], [point[1]], [point[2]],
                          c='yellow', marker='o', s=100, edgecolors='black', linewidths=2,
                          label='Center' if idx == center_indices[0] else '')

    # Plot hip center as a large magenta marker
    LEFT_HIP = 23
    RIGHT_HIP = 24
    if LEFT_HIP < len(landmarks) and RIGHT_HIP < len(landmarks):
        left_hip = landmarks[LEFT_HIP]
        right_hip = landmarks[RIGHT_HIP]
        if not (np.all(left_hip == 0) or np.all(right_hip == 0)):
            hip_center = (left_hip + right_hip) / 2.0
            ax.scatter([hip_center[0]], [hip_center[1]], [hip_center[2]],
                      c='magenta', marker='X', s=300, edgecolors='black', linewidths=3,
                      label='Hip Center (origin)', zorder=10)

    # Plot arm vectors if available (all 4 representations)
    if arm_vectors and arm_vectors.get('wrist_location'):
        wrist = np.array(arm_vectors['wrist_location'])

        print("\n" + "="*70)
        print("VISUALIZE_SKELETON_3D.PY - ARM VECTOR DEBUGGING")
        print("="*70)
        print(f"Frame: {frame_name}")
        print(f"\nWrist location from arm_vectors: [{wrist[0]:+.3f}, {wrist[1]:+.3f}, {wrist[2]:+.3f}]")

        # Get origin landmark positions for dashed lines
        # MediaPipe indices: NOSE=0, LEFT_EYE=2, RIGHT_EYE=5, LEFT_SHOULDER=11, RIGHT_SHOULDER=12,
        #                   LEFT_ELBOW=13, RIGHT_ELBOW=14
        NOSE = 0
        LEFT_EYE = 2
        RIGHT_EYE = 5
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        print(f"\nLandmark positions:")
        print(f"  LEFT wrist (15):     [{landmarks[LEFT_WRIST][0]:+.3f}, {landmarks[LEFT_WRIST][1]:+.3f}, {landmarks[LEFT_WRIST][2]:+.3f}]")
        print(f"  RIGHT wrist (16):    [{landmarks[RIGHT_WRIST][0]:+.3f}, {landmarks[RIGHT_WRIST][1]:+.3f}, {landmarks[RIGHT_WRIST][2]:+.3f}]")
        print(f"  LEFT shoulder (11):  [{landmarks[LEFT_SHOULDER][0]:+.3f}, {landmarks[LEFT_SHOULDER][1]:+.3f}, {landmarks[LEFT_SHOULDER][2]:+.3f}]")
        print(f"  RIGHT shoulder (12): [{landmarks[RIGHT_SHOULDER][0]:+.3f}, {landmarks[RIGHT_SHOULDER][1]:+.3f}, {landmarks[RIGHT_SHOULDER][2]:+.3f}]")

        # Check which wrist the arm_vectors wrist matches
        left_wrist_match = np.allclose(wrist, landmarks[LEFT_WRIST], atol=0.001)
        right_wrist_match = np.allclose(wrist, landmarks[RIGHT_WRIST], atol=0.001)

        print(f"\nWrist location matches:")
        print(f"  LEFT wrist?  {left_wrist_match}")
        print(f"  RIGHT wrist? {right_wrist_match}")

        if left_wrist_match:
            print(f"  ✓ arm_vectors computed for LEFT wrist")
        elif right_wrist_match:
            print(f"  ✗ arm_vectors computed for RIGHT wrist")
        else:
            print(f"  ??? arm_vectors don't match either wrist!")

        # Helper function to draw dashed line and vector
        def draw_vector_with_reference(origin_idx, vector_key, color, label):
            """Draw dashed line from origin to wrist, and extended vector from wrist."""
            if arm_vectors.get(vector_key) and origin_idx < len(landmarks):
                origin = landmarks[origin_idx]

                # Skip if origin is invalid
                if np.all(origin == 0):
                    return

                # Draw dashed reference line from origin to wrist
                ax.plot([origin[0], wrist[0]],
                       [origin[1], wrist[1]],
                       [origin[2], wrist[2]],
                       color=color, linestyle='--', linewidth=3, alpha=0.8)

                # Draw extended vector arrow from wrist
                vec = np.array(arm_vectors[vector_key])
                vec_scaled = vec * vec_scale
                ax.quiver(wrist[0], wrist[1], wrist[2],
                         vec_scaled[0], vec_scaled[1], vec_scaled[2],
                         color=color, arrow_length_ratio=0.2, linewidth=3,
                         label=label)

        # Vector scale for visibility
        vec_scale = 0.5  # Extend vectors further for better visualization

        # Determine which shoulder/elbow/eye to use based on which side is closer to wrist
        dist_left = np.linalg.norm(landmarks[LEFT_SHOULDER] - wrist)
        dist_right = np.linalg.norm(landmarks[RIGHT_SHOULDER] - wrist)

        print(f"\nDistance calculation for drawing reference lines:")
        print(f"  Distance from LEFT shoulder to wrist:  {dist_left:.3f} m")
        print(f"  Distance from RIGHT shoulder to wrist: {dist_right:.3f} m")

        if dist_left < dist_right:
            shoulder_idx = LEFT_SHOULDER
            elbow_idx = LEFT_ELBOW
            eye_idx = LEFT_EYE
            side_selected = "LEFT"
        else:
            shoulder_idx = RIGHT_SHOULDER
            elbow_idx = RIGHT_ELBOW
            eye_idx = RIGHT_EYE
            side_selected = "RIGHT"

        print(f"\n⚠️  VISUALIZATION DECISION:")
        print(f"  Using {side_selected} shoulder/elbow/eye for drawing reference lines")
        print(f"  Shoulder index: {shoulder_idx}")
        print(f"  Elbow index: {elbow_idx}")
        print(f"  Eye index: {eye_idx}")

        print(f"\nDrawing vectors:")
        print(f"  shoulder_to_wrist vector: {arm_vectors.get('shoulder_to_wrist')}")
        print(f"  elbow_to_wrist vector:    {arm_vectors.get('elbow_to_wrist')}")
        print(f"  eye_to_wrist vector:      {arm_vectors.get('eye_to_wrist')}")
        print(f"  nose_to_wrist vector:     {arm_vectors.get('nose_to_wrist')}")
        print("="*70 + "\n")

        # Draw all 4 vector representations with dashed reference lines
        draw_vector_with_reference(shoulder_idx, 'shoulder_to_wrist', 'green', 'shoulder→wrist')
        draw_vector_with_reference(elbow_idx, 'elbow_to_wrist', 'orange', 'elbow→wrist')
        draw_vector_with_reference(eye_idx, 'eye_to_wrist', 'purple', 'eye→wrist')
        draw_vector_with_reference(NOSE, 'nose_to_wrist', 'cyan', 'nose→wrist')

    # Draw head orientation vector if available
    if head_orientation and 'head_orientation_vector' in head_orientation and 'head_orientation_origin' in head_orientation:
        head_vec = np.array(head_orientation['head_orientation_vector'])
        head_origin = np.array(head_orientation['head_orientation_origin'])

        # Skip if any values are None
        if head_vec is not None and head_origin is not None and not np.any(np.isnan(head_vec)) and not np.any(np.isnan(head_origin)):
            # Scale for visibility (similar to arm vectors)
            head_vec_scaled = head_vec * 0.5

            # Draw head orientation vector as yellow arrow from nose
            ax.quiver(head_origin[0], head_origin[1], head_origin[2],
                     head_vec_scaled[0], head_vec_scaled[1], head_vec_scaled[2],
                     color='yellow', arrow_length_ratio=0.2, linewidth=4,
                     label='head orientation')

    # Draw human facing direction indicator
    if len(landmarks) > LEFT_SHOULDER:
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        # Get shoulder positions
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]
        nose = landmarks[NOSE]

        # Skip if invalid
        if not (np.all(left_shoulder == 0) or np.all(right_shoulder == 0) or np.all(nose == 0)):
            # Calculate shoulder midpoint
            shoulder_mid = (left_shoulder + right_shoulder) / 2

            # Calculate forward direction (perpendicular to shoulder line, in XZ plane)
            shoulder_vector = right_shoulder - left_shoulder  # left to right
            # Cross with Y-axis to get forward direction
            forward = np.cross(shoulder_vector, [0, 1, 0])
            forward_norm = np.linalg.norm(forward)

            if forward_norm > 0:
                forward = forward / forward_norm

                # Scale the forward vector for visibility
                forward_scaled = forward * 0.4  # 40cm forward indicator

                # # Draw a thick magenta arrow showing body facing direction
                # ax.quiver(shoulder_mid[0], shoulder_mid[1], shoulder_mid[2],
                #          forward_scaled[0], forward_scaled[1], forward_scaled[2],
                #          color='magenta', arrow_length_ratio=0.3, linewidth=5,
                #          label='Body facing')

    # Plot targets if available
    if targets:
        target_positions = []
        target_labels = []
        for target in targets:
            if 'x' in target and 'y' in target and 'z' in target:
                target_positions.append([target['x'], target['y'], target['z']])
                target_labels.append(target.get('label', 'target'))

        if target_positions:
            target_positions = np.array(target_positions)
            # Draw targets as larger markers with different color
            ax.scatter(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2],
                      c='gold', marker='*', s=300, edgecolors='black', linewidths=2,
                      label='Targets', zorder=10)

            # Add labels for each target
            for pos, label in zip(target_positions, target_labels):
                ax.text(pos[0], pos[1], pos[2], f'  {label}',
                       fontsize=10, fontweight='bold', color='darkgoldenrod')

    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'3D Skeleton Visualization - {frame_name}')

    # Set equal aspect ratio using all non-zero landmarks
    valid_points = landmarks[~np.all(landmarks == 0, axis=1)]
    if len(valid_points) > 0:
        max_range = np.array([
            valid_points[:, 0].max() - valid_points[:, 0].min(),
            valid_points[:, 1].max() - valid_points[:, 1].min(),
            valid_points[:, 2].max() - valid_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (valid_points[:, 0].max() + valid_points[:, 0].min()) * 0.5
        mid_y = (valid_points[:, 1].max() + valid_points[:, 1].min()) * 0.5
        mid_z = (valid_points[:, 2].max() + valid_points[:, 2].min()) * 0.5

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2.5, 0.5)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    if show:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_skeleton_3d.py <path_to_skeleton_2d.json> [frame_name] [targets_json]")
        print("Example: python visualize_skeleton_3d.py trial_output/trial_1/single_camera/skeleton_2d.json")
        print("         python visualize_skeleton_3d.py skeleton_2d.json frame_000001 target_detections_cam_frame.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Load skeleton data
    print(f"Loading skeleton data from: {json_path}")
    data = load_skeleton_data(json_path)

    # Load targets if provided (3rd argument)
    targets = None
    if len(sys.argv) > 3:
        targets_path = sys.argv[3]
        if Path(targets_path).exists():
            print(f"Loading targets from: {targets_path}")
            targets = load_skeleton_data(targets_path)
        else:
            print(f"Warning: Targets file not found: {targets_path}")
    else:
        # Try to find targets file in same directory
        skeleton_path = Path(json_path)
        targets_path = skeleton_path.parent / "target_detections_cam_frame.json"
        if targets_path.exists():
            print(f"Found targets file: {targets_path}")
            targets = load_skeleton_data(str(targets_path))
        else:
            print("No targets file found (optional)")

    # Get frame to visualize (use first frame or specific frame)
    if len(sys.argv) > 2:
        frame_name = sys.argv[2]
    else:
        frame_name = list(data.keys())[0]

    if frame_name not in data:
        print(f"Error: Frame {frame_name} not found in data")
        print(f"Available frames: {list(data.keys())[:5]}...")
        sys.exit(1)

    frame_data = data[frame_name]

    # Extract 3D landmarks and arm vectors
    if 'landmarks_3d' not in frame_data:
        print("Error: No 3D landmarks found in data")
        print("Make sure to run skeleton extraction with depth images")
        sys.exit(1)

    landmarks_3d = frame_data['landmarks_3d']
    arm_vectors = frame_data.get('arm_vectors')

    print(f"Visualizing frame: {frame_name}")
    print(f"Number of 3D landmarks: {len(landmarks_3d)}")
    if arm_vectors:
        print(f"Arm vectors: {list(arm_vectors.keys())}")
    if targets:
        print(f"Number of targets: {len(targets)}")

    # Visualize
    plot_skeleton_3d(landmarks_3d, arm_vectors, frame_name, targets)


if __name__ == "__main__":
    main()

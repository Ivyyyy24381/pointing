#!/usr/bin/env python3
"""
Baby Detection CSV Exporter

Exports baby (human) detection results to a comprehensive CSV table with:
- Frame metadata (frame_index, time_sec, local_frame_index)
- Target spatial relationships (r, theta, phi)
- Baby bounding box and confidence
- All MediaPipe keypoints in 2D (x, y, conf) and 3D (x_m, y_m, z_m)
- Arm orientation vectors
- 3D trace coordinates
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class BabyCSVExporter:
    """Exports baby detection results to CSV format."""

    # MediaPipe Pose keypoint names (33 landmarks)
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    def __init__(self, fps: float = 30.0):
        """
        Initialize exporter.

        Args:
            fps: Frames per second for time calculation
        """
        self.fps = fps

    def export_to_csv(self,
                     baby_results_path: Path,
                     targets_path: Optional[Path],
                     output_csv_path: Path,
                     start_frame_index: int = 0):
        """
        Export baby detection results to CSV.

        Args:
            baby_results_path: Path to skeleton_2d.json
            targets_path: Optional path to target_detections_cam_frame.json
            output_csv_path: Path to save CSV
            start_frame_index: Starting frame index for time calculation
        """
        # Load baby results
        with open(baby_results_path) as f:
            baby_data = json.load(f)

        if not baby_data:
            print("⚠️ No baby detection data found")
            return

        # Load targets
        targets = []
        if targets_path and targets_path.exists():
            with open(targets_path) as f:
                targets_json = json.load(f)
                if isinstance(targets_json, list):
                    targets = targets_json
                else:
                    targets = targets_json.get('targets', [])

        # Build column names
        columns = self._build_column_names(targets)

        # Write CSV
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()

            frame_keys = sorted(baby_data.keys())
            for local_idx, frame_key in enumerate(frame_keys):
                baby_result = baby_data[frame_key]
                global_frame_idx = start_frame_index + local_idx

                row = self._build_csv_row(
                    frame_key=frame_key,
                    global_frame_index=global_frame_idx,
                    local_frame_index=local_idx,
                    baby_result=baby_result,
                    targets=targets
                )

                writer.writerow(row)

        print(f"✅ Exported {len(frame_keys)} frames to {output_csv_path}")

    def _build_column_names(self, targets: List[dict]) -> List[str]:
        """Build list of CSV column names."""
        columns = [
            'frame_index',
            'time_sec',
            'local_frame_index'
        ]

        # Target spatial relationships
        for i in range(1, len(targets) + 1):
            columns.extend([
                f'target_{i}_r',
                f'target_{i}_theta',
                f'target_{i}_phi'
            ])

        # All MediaPipe keypoints (2D and 3D)
        for kp_name in self.KEYPOINT_NAMES:
            columns.extend([
                f'{kp_name}_x',
                f'{kp_name}_y',
                f'{kp_name}_conf',
                f'{kp_name}_x_m',
                f'{kp_name}_y_m',
                f'{kp_name}_z_m'
            ])

        # Arm orientation vectors
        columns.extend([
            'left_arm_orientation_x',
            'left_arm_orientation_y',
            'left_arm_orientation_z',
            'right_arm_orientation_x',
            'right_arm_orientation_y',
            'right_arm_orientation_z'
        ])

        # 3D trace
        columns.extend(['trace3d_x', 'trace3d_y', 'trace3d_z'])

        return columns

    def _build_csv_row(self,
                      frame_key: str,
                      global_frame_index: int,
                      local_frame_index: int,
                      baby_result: dict,
                      targets: List[dict]) -> dict:
        """Build a single CSV row."""
        row = {}

        # Frame metadata
        frame_num = int(frame_key.split('_')[1])
        row['frame_index'] = frame_num
        row['time_sec'] = round(local_frame_index / self.fps, 3)
        row['local_frame_index'] = local_frame_index

        # Get baby nose position for spatial calculations
        landmarks_2d = baby_result.get('landmarks_2d', [])
        landmarks_3d = baby_result.get('landmarks_3d', [])

        # Extract nose position (first landmark in MediaPipe)
        nose_3d = landmarks_3d[0] if landmarks_3d and len(landmarks_3d) > 0 else [0, 0, 0]

        # Calculate spatial relationships to targets
        self._add_spatial_relationships(row, nose_3d, targets)

        # Add all keypoints
        self._add_keypoints(row, landmarks_2d, landmarks_3d)

        # Add arm orientation vectors
        arm_vectors = baby_result.get('arm_vectors', {})
        if arm_vectors:
            left_arm = arm_vectors.get('left_arm', [0, 0, 0])
            right_arm = arm_vectors.get('right_arm', [0, 0, 0])
            row['left_arm_orientation_x'] = left_arm[0] if len(left_arm) > 0 else 0
            row['left_arm_orientation_y'] = left_arm[1] if len(left_arm) > 1 else 0
            row['left_arm_orientation_z'] = left_arm[2] if len(left_arm) > 2 else 0
            row['right_arm_orientation_x'] = right_arm[0] if len(right_arm) > 0 else 0
            row['right_arm_orientation_y'] = right_arm[1] if len(right_arm) > 1 else 0
            row['right_arm_orientation_z'] = right_arm[2] if len(right_arm) > 2 else 0
        else:
            row['left_arm_orientation_x'] = 0
            row['left_arm_orientation_y'] = 0
            row['left_arm_orientation_z'] = 0
            row['right_arm_orientation_x'] = 0
            row['right_arm_orientation_y'] = 0
            row['right_arm_orientation_z'] = 0

        # 3D trace (use nose position)
        if nose_3d and len(nose_3d) >= 3:
            row['trace3d_x'] = nose_3d[0]
            row['trace3d_y'] = nose_3d[1]
            row['trace3d_z'] = nose_3d[2]
        else:
            row['trace3d_x'] = 0
            row['trace3d_y'] = 0
            row['trace3d_z'] = 0

        return row

    def _add_spatial_relationships(self, row: dict, baby_pos_3d: List[float],
                                  targets: List[dict]):
        """Add spatial relationships (r, theta, phi) to targets."""
        for i, target in enumerate(targets, 1):
            # Targets use 'x', 'y', 'z' keys
            target_pos = [target.get('x', 0), target.get('y', 0), target.get('z', 0)]
            r, theta, phi = self._compute_spherical_coords(baby_pos_3d, target_pos)
            row[f'target_{i}_r'] = r
            row[f'target_{i}_theta'] = theta
            row[f'target_{i}_phi'] = phi

    def _compute_spherical_coords(self, origin: List[float],
                                 target: List[float]) -> Tuple[float, float, float]:
        """
        Compute spherical coordinates (r, theta, phi) from origin to target.

        Returns:
            r: Distance (NaN if depth data missing)
            theta: Azimuthal angle (horizontal, NaN if depth data missing)
            phi: Polar angle (vertical, NaN if depth data missing)
        """
        if not origin or not target or len(origin) < 3 or len(target) < 3:
            return np.nan, np.nan, np.nan

        # Check if origin has valid 3D data (not all zeros - indicates missing depth)
        if abs(origin[0]) < 0.001 and abs(origin[1]) < 0.001 and abs(origin[2]) < 0.001:
            return np.nan, np.nan, np.nan

        # Check if target has valid 3D data
        if abs(target[0]) < 0.001 and abs(target[1]) < 0.001 and abs(target[2]) < 0.001:
            return np.nan, np.nan, np.nan

        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        dz = target[2] - origin[2]

        r = np.sqrt(dx**2 + dy**2 + dz**2)

        if r < 1e-6:
            return np.nan, np.nan, np.nan

        theta = np.arctan2(dy, dx)
        phi = np.arccos(dz / r)

        return round(r, 10), round(theta, 10), round(phi, 10)

    def _add_keypoints(self, row: dict, landmarks_2d: List[List[float]],
                      landmarks_3d: Optional[List[List[float]]]):
        """Add all keypoint data to row."""
        for idx, kp_name in enumerate(self.KEYPOINT_NAMES):
            # 2D keypoint
            if landmarks_2d and idx < len(landmarks_2d):
                kp_2d = landmarks_2d[idx]
                row[f'{kp_name}_x'] = kp_2d[0] if len(kp_2d) > 0 else 0
                row[f'{kp_name}_y'] = kp_2d[1] if len(kp_2d) > 1 else 0
                row[f'{kp_name}_conf'] = kp_2d[2] if len(kp_2d) > 2 else 0
            else:
                row[f'{kp_name}_x'] = 0
                row[f'{kp_name}_y'] = 0
                row[f'{kp_name}_conf'] = 0

            # 3D keypoint
            if landmarks_3d and idx < len(landmarks_3d):
                kp_3d = landmarks_3d[idx]
                row[f'{kp_name}_x_m'] = kp_3d[0] if len(kp_3d) > 0 else 0
                row[f'{kp_name}_y_m'] = kp_3d[1] if len(kp_3d) > 1 else 0
                row[f'{kp_name}_z_m'] = kp_3d[2] if len(kp_3d) > 2 else 0
            else:
                row[f'{kp_name}_x_m'] = 0
                row[f'{kp_name}_y_m'] = 0
                row[f'{kp_name}_z_m'] = 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python baby_csv_exporter.py <baby_results_json> <output_csv> [targets_json]")
        sys.exit(1)

    baby_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    targets_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    exporter = BabyCSVExporter(fps=30.0)
    exporter.export_to_csv(
        baby_results_path=baby_path,
        targets_path=targets_path,
        output_csv_path=output_path
    )

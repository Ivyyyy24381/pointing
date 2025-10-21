#!/usr/bin/env python3
"""
Dog Detection CSV Exporter

Exports dog detection results to a comprehensive CSV table with:
- Frame metadata (frame_index, time_sec, local_frame_index)
- Human and target spatial relationships (r, theta, phi)
- Dog bounding box and confidence
- Dog direction and orientation vectors
- All keypoints in 2D (x, y, conf) and 3D (x_m, y_m, z_m)
- Head and torso orientation vectors
- 3D trace coordinates
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class DogCSVExporter:
    """Export dog detection results to CSV format."""

    # Mapping from DLC SuperAnimal Quadruped keypoints to our CSV columns
    KEYPOINT_MAPPING = {
        'nose': 'nose',
        'right_eye': 'right_eye',
        'right_ear_base': 'right_ear_base',
        'right_ear_tip': 'right_ear_tip',
        'left_eye': 'left_eye',
        'left_ear_base': 'left_ear_base',
        'left_ear_tip': 'left_ear_tip',
        'throat': 'throat',
        'neck_base': 'neck',  # Closest approximation
        'withers': 'withers',
        'tail_base': 'tail_base',
        'tail_tip': 'tail_tip'
    }

    def __init__(self, fps: float = 30.0):
        """
        Initialize exporter.

        Args:
            fps: Frame rate for time calculation
        """
        self.fps = fps

    def export_to_csv(self,
                     dog_results_path: Path,
                     human_results_path: Optional[Path],
                     targets_path: Optional[Path],
                     output_csv_path: Path,
                     start_frame_index: int = 0):
        """
        Export dog detection results to CSV.

        Args:
            dog_results_path: Path to dog_detection_results.json
            human_results_path: Optional path to human skeleton_2d.json
            targets_path: Optional path to target_detections_cam_frame.json
            output_csv_path: Path to output CSV file
            start_frame_index: Global frame index to start from (for multi-trial concatenation)
        """
        # Load dog results
        with open(dog_results_path) as f:
            dog_data = json.load(f)

        # Load human results if available
        human_data = {}
        if human_results_path and human_results_path.exists():
            with open(human_results_path) as f:
                human_data = json.load(f)

        # Load targets if available
        targets = []
        if targets_path and targets_path.exists():
            with open(targets_path) as f:
                targets_json = json.load(f)
                # Handle both list and dict formats
                if isinstance(targets_json, list):
                    targets = targets_json
                else:
                    targets = targets_json.get('targets', [])

        # Sort frame keys
        frame_keys = sorted(dog_data.keys())

        # Open CSV for writing
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self._get_csv_columns(len(targets)))
            writer.writeheader()

            for local_idx, frame_key in enumerate(frame_keys):
                dog_result = dog_data[frame_key]
                human_result = human_data.get(frame_key, {})

                # Build CSV row
                row = self._build_csv_row(
                    frame_key=frame_key,
                    global_frame_index=start_frame_index + local_idx,
                    local_frame_index=local_idx,
                    dog_result=dog_result,
                    human_result=human_result,
                    targets=targets
                )

                writer.writerow(row)

        print(f"✅ Exported {len(frame_keys)} frames to {output_csv_path}")

    def _get_csv_columns(self, num_targets: int) -> List[str]:
        """Get CSV column names."""
        columns = [
            'frame_index',
            'time_sec',
            'local_frame_index'
        ]

        # Add human and target spatial relationships
        columns.append('human_r')
        for i in range(1, num_targets + 1):
            columns.append(f'target_{i}_r')

        columns.append('human_theta')
        for i in range(1, num_targets + 1):
            columns.append(f'target_{i}_theta')

        columns.append('human_phi')
        for i in range(1, num_targets + 1):
            columns.append(f'target_{i}_phi')

        # Bounding box and confidence
        columns.extend(['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'confidence'])

        # Dog direction
        columns.append('dog_dir')

        # Keypoints (2D and 3D)
        keypoint_names = list(self.KEYPOINT_MAPPING.values())
        for kp_name in keypoint_names:
            columns.extend([
                f'{kp_name}_x',
                f'{kp_name}_y',
                f'{kp_name}_conf',
                f'{kp_name}_x_m',
                f'{kp_name}_y_m',
                f'{kp_name}_z_m'
            ])

        # Orientation vectors
        columns.extend([
            'head_orientation_x',
            'head_orientation_y',
            'torso_orientation_x',
            'torso_orientation_y',
            'head_orientation_x_m',
            'head_orientation_y_m',
            'head_orientation_z_m',
            'torso_orientation_x_m',
            'torso_orientation_y_m',
            'torso_orientation_z_m'
        ])

        # 3D trace
        columns.extend(['trace3d_x', 'trace3d_y', 'trace3d_z'])

        return columns

    def _build_csv_row(self,
                      frame_key: str,
                      global_frame_index: int,
                      local_frame_index: int,
                      dog_result: dict,
                      human_result: dict,
                      targets: List[dict]) -> dict:
        """Build a single CSV row."""
        row = {}

        # Frame metadata
        frame_num = int(frame_key.split('_')[1])
        row['frame_index'] = frame_num
        row['time_sec'] = round(local_frame_index / self.fps, 3)
        row['local_frame_index'] = local_frame_index

        # Get dog nose position for spatial calculations
        keypoints_2d = dog_result.get('keypoints_2d', [])
        keypoints_3d = dog_result.get('keypoints_3d', [])

        # Extract nose position (first keypoint in MediaPipe format)
        nose_2d = keypoints_2d[0] if len(keypoints_2d) > 0 else [0, 0, 0]
        nose_3d = keypoints_3d[0] if keypoints_3d and len(keypoints_3d) > 0 else [0, 0, 0]

        # Calculate spatial relationships to human and targets
        self._add_spatial_relationships(row, nose_3d, human_result, targets)

        # Bounding box
        bbox = dog_result.get('bbox', [0, 0, 0, 0])
        if bbox:
            row['bbox_x'] = bbox[0]
            row['bbox_y'] = bbox[1]
            row['bbox_w'] = bbox[2] - bbox[0]
            row['bbox_h'] = bbox[3] - bbox[1]
        else:
            row['bbox_x'] = 0
            row['bbox_y'] = 0
            row['bbox_w'] = 0
            row['bbox_h'] = 0

        # Overall confidence (average of all keypoints)
        confidences = [kp[2] for kp in keypoints_2d if len(kp) > 2]
        row['confidence'] = np.mean(confidences) if confidences else 0.0

        # Dog direction (computed from head orientation)
        row['dog_dir'] = ''  # Will be filled when we compute orientations

        # Add all keypoints
        self._add_keypoints(row, keypoints_2d, keypoints_3d)

        # Compute and add orientation vectors
        self._add_orientations(row, keypoints_2d, keypoints_3d)

        # 3D trace (center point for tracking)
        if keypoints_3d and len(keypoints_3d) > 0:
            # Use withers (back center) as trace point
            withers_idx = 6  # withers in MediaPipe format
            if len(keypoints_3d) > withers_idx:
                row['trace3d_x'] = keypoints_3d[withers_idx][0]
                row['trace3d_y'] = keypoints_3d[withers_idx][1]
                row['trace3d_z'] = keypoints_3d[withers_idx][2]
            else:
                row['trace3d_x'] = nose_3d[0]
                row['trace3d_y'] = nose_3d[1]
                row['trace3d_z'] = nose_3d[2]
        else:
            row['trace3d_x'] = 0
            row['trace3d_y'] = 0
            row['trace3d_z'] = 0

        return row

    def _add_spatial_relationships(self, row: dict, dog_pos_3d: List[float],
                                  human_result: dict, targets: List[dict]):
        """Add spatial relationships (r, theta, phi) to human and targets."""
        # Human relationship
        if human_result and 'landmarks_3d' in human_result:
            human_pos = human_result['landmarks_3d'][0]  # Nose
            r, theta, phi = self._compute_spherical_coords(dog_pos_3d, human_pos)
            row['human_r'] = r
            row['human_theta'] = theta
            row['human_phi'] = phi
        else:
            row['human_r'] = 0
            row['human_theta'] = 0
            row['human_phi'] = 0

        # Target relationships
        for i, target in enumerate(targets, 1):
            # Targets use 'x', 'y', 'z' keys
            target_pos = [target.get('x', 0), target.get('y', 0), target.get('z', 0)]
            r, theta, phi = self._compute_spherical_coords(dog_pos_3d, target_pos)
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

    def _add_keypoints(self, row: dict, keypoints_2d: List[List[float]],
                      keypoints_3d: Optional[List[List[float]]]):
        """Add all keypoint data to row."""
        # MediaPipe pose landmark indices we care about
        # (Mapped from DLC SuperAnimal Quadruped)
        keypoint_indices = {
            'nose': 0,
            'right_eye': 2,
            'right_ear_base': 4,  # Approximation
            'right_ear_tip': 4,   # Same as base for now
            'left_eye': 1,
            'left_ear_base': 3,   # Approximation
            'left_ear_tip': 3,    # Same as base for now
            'throat': 5,
            'neck': 5,            # Use throat as approximation
            'withers': 6,
            'tail_base': 7,
            'tail_tip': 8
        }

        for kp_name, kp_idx in keypoint_indices.items():
            # 2D coordinates
            if len(keypoints_2d) > kp_idx:
                kp_2d = keypoints_2d[kp_idx]
                row[f'{kp_name}_x'] = kp_2d[0] if len(kp_2d) > 0 else 0
                row[f'{kp_name}_y'] = kp_2d[1] if len(kp_2d) > 1 else 0
                row[f'{kp_name}_conf'] = kp_2d[2] if len(kp_2d) > 2 else 0
            else:
                row[f'{kp_name}_x'] = 0
                row[f'{kp_name}_y'] = 0
                row[f'{kp_name}_conf'] = 0

            # 3D coordinates
            if keypoints_3d and len(keypoints_3d) > kp_idx:
                kp_3d = keypoints_3d[kp_idx]
                row[f'{kp_name}_x_m'] = kp_3d[0] if len(kp_3d) > 0 else 0
                row[f'{kp_name}_y_m'] = kp_3d[1] if len(kp_3d) > 1 else 0
                row[f'{kp_name}_z_m'] = kp_3d[2] if len(kp_3d) > 2 else 0
            else:
                row[f'{kp_name}_x_m'] = 0
                row[f'{kp_name}_y_m'] = 0
                row[f'{kp_name}_z_m'] = 0

    def _add_orientations(self, row: dict, keypoints_2d: List[List[float]],
                         keypoints_3d: Optional[List[List[float]]]):
        """Compute and add head and torso orientation vectors."""
        # Head orientation: nose to midpoint between eyes (2D)
        if len(keypoints_2d) >= 3:
            nose = np.array(keypoints_2d[0][:2])
            right_eye = np.array(keypoints_2d[2][:2])
            left_eye = np.array(keypoints_2d[1][:2])
            eye_center = (right_eye + left_eye) / 2
            head_vec_2d = nose - eye_center

            # Normalize
            norm = np.linalg.norm(head_vec_2d)
            if norm > 1e-6:
                head_vec_2d = head_vec_2d / norm
                row['head_orientation_x'] = round(head_vec_2d[0], 10)
                row['head_orientation_y'] = round(head_vec_2d[1], 10)
            else:
                row['head_orientation_x'] = 0
                row['head_orientation_y'] = 0
        else:
            row['head_orientation_x'] = 0
            row['head_orientation_y'] = 0

        # Torso orientation: withers to tail_base (2D)
        if len(keypoints_2d) >= 8:
            withers = np.array(keypoints_2d[6][:2])
            tail_base = np.array(keypoints_2d[7][:2])
            torso_vec_2d = tail_base - withers

            # Normalize
            norm = np.linalg.norm(torso_vec_2d)
            if norm > 1e-6:
                torso_vec_2d = torso_vec_2d / norm
                row['torso_orientation_x'] = round(torso_vec_2d[0], 10)
                row['torso_orientation_y'] = round(torso_vec_2d[1], 10)
            else:
                row['torso_orientation_x'] = 0
                row['torso_orientation_y'] = 0
        else:
            row['torso_orientation_x'] = 0
            row['torso_orientation_y'] = 0

        # 3D orientations
        if keypoints_3d and len(keypoints_3d) >= 3:
            nose_3d = np.array(keypoints_3d[0])
            right_eye_3d = np.array(keypoints_3d[2])
            left_eye_3d = np.array(keypoints_3d[1])
            eye_center_3d = (right_eye_3d + left_eye_3d) / 2
            head_vec_3d = nose_3d - eye_center_3d

            # Normalize
            norm = np.linalg.norm(head_vec_3d)
            if norm > 1e-6:
                head_vec_3d = head_vec_3d / norm
                row['head_orientation_x_m'] = round(head_vec_3d[0], 10)
                row['head_orientation_y_m'] = round(head_vec_3d[1], 10)
                row['head_orientation_z_m'] = round(head_vec_3d[2], 10)
            else:
                row['head_orientation_x_m'] = 0
                row['head_orientation_y_m'] = 0
                row['head_orientation_z_m'] = 0
        else:
            row['head_orientation_x_m'] = 0
            row['head_orientation_y_m'] = 0
            row['head_orientation_z_m'] = 0

        if keypoints_3d and len(keypoints_3d) >= 8:
            withers_3d = np.array(keypoints_3d[6])
            tail_base_3d = np.array(keypoints_3d[7])
            torso_vec_3d = tail_base_3d - withers_3d

            # Normalize
            norm = np.linalg.norm(torso_vec_3d)
            if norm > 1e-6:
                torso_vec_3d = torso_vec_3d / norm
                row['torso_orientation_x_m'] = round(torso_vec_3d[0], 10)
                row['torso_orientation_y_m'] = round(torso_vec_3d[1], 10)
                row['torso_orientation_z_m'] = round(torso_vec_3d[2], 10)
            else:
                row['torso_orientation_x_m'] = 0
                row['torso_orientation_y_m'] = 0
                row['torso_orientation_z_m'] = 0
        else:
            row['torso_orientation_x_m'] = 0
            row['torso_orientation_y_m'] = 0
            row['torso_orientation_z_m'] = 0


def main():
    """Test the exporter."""
    import argparse

    parser = argparse.ArgumentParser(description='Export dog detection results to CSV')
    parser.add_argument('--dog-results', type=str, required=True,
                       help='Path to dog_detection_results.json')
    parser.add_argument('--human-results', type=str,
                       help='Path to skeleton_2d.json (optional)')
    parser.add_argument('--targets', type=str,
                       help='Path to target_detections_cam_frame.json (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV path')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='Frame rate (default: 30.0)')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Starting frame index (default: 0)')

    args = parser.parse_args()

    exporter = DogCSVExporter(fps=args.fps)
    exporter.export_to_csv(
        dog_results_path=Path(args.dog_results),
        human_results_path=Path(args.human_results) if args.human_results else None,
        targets_path=Path(args.targets) if args.targets else None,
        output_csv_path=Path(args.output),
        start_frame_index=args.start_frame
    )


if __name__ == '__main__':
    main()

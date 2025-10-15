"""
Calculate distances from pointing wrist to target objects.

Computes 3D Euclidean distances and determines closest target.
"""

import numpy as np
import json
from typing import List, Dict, Tuple
from pathlib import Path


def load_targets(targets_file: str) -> List[Dict]:
    """
    Load target positions from JSON file.

    Args:
        targets_file: Path to target_detections.json from Step 0

    Returns:
        List of target dictionaries with 3D positions
    """
    with open(targets_file, 'r') as f:
        targets = json.load(f)

    return targets


def compute_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Compute 3D Euclidean distance between two points.

    Args:
        point_a: [x, y, z]
        point_b: [x, y, z]

    Returns:
        Distance in meters
    """
    return float(np.linalg.norm(point_a - point_b))


def compute_distances_to_targets(wrist_position: np.ndarray,
                                 targets: List[Dict]) -> Dict[str, float]:
    """
    Compute distances from wrist to all targets.

    Args:
        wrist_position: [x, y, z] in camera frame
        targets: List of target dicts with 'x', 'y', 'z', 'label'

    Returns:
        Dict mapping target_label -> distance
    """
    distances = {}

    for target in targets:
        target_pos = np.array([target['x'], target['y'], target['z']])
        distance = compute_distance(wrist_position, target_pos)
        distances[target['label']] = distance

    return distances


def find_closest_target(distances: Dict[str, float]) -> Tuple[str, float]:
    """
    Find closest target from distances.

    Args:
        distances: Dict mapping target_label -> distance

    Returns:
        (closest_target_label, distance)
    """
    if not distances:
        return ("none", float('inf'))

    closest = min(distances.items(), key=lambda x: x[1])
    return closest


def analyze_pointing_trial(pointing_file: str,
                           targets_file: str,
                           output_file: str) -> None:
    """
    Analyze entire trial: compute distances for each frame.

    Args:
        pointing_file: Path to pointing_results.json from PointingCalculator
        targets_file: Path to target_detections.json from Step 0
        output_file: Output CSV file

    Outputs:
        CSV with columns: frame, wrist_x, wrist_y, wrist_z,
                         distance_target_1, distance_target_2, ...
                         closest_target, closest_distance
    """
    print(f"\n{'='*60}")
    print(f"Distance Analysis")
    print(f"{'='*60}")

    # Load pointing results
    print(f"📁 Loading pointing: {pointing_file}")
    with open(pointing_file, 'r') as f:
        pointing_data = json.load(f)

    # Load targets
    print(f"📁 Loading targets: {targets_file}")
    targets = load_targets(targets_file)
    print(f"✅ Loaded {len(targets)} targets")

    # Prepare output
    rows = []

    print(f"🔄 Computing distances...")

    for frame_key, frame_data in pointing_data.items():
        frame_num = frame_data['frame']
        wrist_3d = frame_data.get('wrist_3d')

        if not wrist_3d:
            continue

        wrist_pos = np.array([wrist_3d['x'], wrist_3d['y'], wrist_3d['z']])

        # Compute distances to all targets
        distances = compute_distances_to_targets(wrist_pos, targets)

        # Find closest
        closest_label, closest_dist = find_closest_target(distances)

        # Build row
        row = {
            'frame': frame_num,
            'wrist_x': wrist_3d['x'],
            'wrist_y': wrist_3d['y'],
            'wrist_z': wrist_3d['z'],
            'pointing_arm': frame_data['pointing_arm'],
            'closest_target': closest_label,
            'closest_distance': closest_dist
        }

        # Add individual target distances
        for target_label, distance in distances.items():
            row[f'distance_{target_label}'] = distance

        rows.append(row)

        if frame_num % 10 == 0:
            print(f"  Processed frame {frame_num}...", end='\r')

    print(f"\n✅ Analyzed {len(rows)} frames")

    # Save to CSV
    import csv

    if rows:
        fieldnames = list(rows[0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"💾 Saved analysis to: {output_file}")
    else:
        print("⚠️ No valid frames to save")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute distances to targets")
    parser.add_argument("--pointing", required=True, help="Path to pointing_results.json")
    parser.add_argument("--targets", required=True, help="Path to target_detections.json")
    parser.add_argument("--output", required=True, help="Output CSV file")

    args = parser.parse_args()

    analyze_pointing_trial(args.pointing, args.targets, args.output)

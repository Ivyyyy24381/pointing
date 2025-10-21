"""
Generate 2D trace and distance-to-targets plots for dog/baby subjects.

This module creates two visualization plots:
1. 2D Top-Down Trace: Shows subject movement in XZ plane (top view)
2. Distance to Targets Over Time: Shows distance from subject to each target over time
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def load_subject_results(json_path: Path) -> Dict:
    """Load subject detection results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_targets(trial_path: Path) -> Optional[List[Dict]]:
    """Load target positions from target_detections_cam_frame.json."""
    target_file = trial_path / "target_detections_cam_frame.json"
    if not target_file.exists():
        return None

    with open(target_file, 'r') as f:
        targets = json.load(f)
    return targets


def extract_subject_trajectory(results: Dict) -> Tuple[List[float], List[np.ndarray]]:
    """
    Extract subject 3D trajectory from detection results.

    Returns:
        timestamps: List of timestamps (frame numbers as floats)
        positions: List of 3D positions [x, y, z] as numpy arrays
    """
    timestamps = []
    positions = []

    # Sort frames by frame number
    frame_keys = sorted(results.keys(), key=lambda x: int(x.split('_')[1]))

    for frame_key in frame_keys:
        frame_data = results[frame_key]

        # Extract frame number as timestamp
        frame_num = int(frame_key.split('_')[1])

        # Get keypoints_3d and compute centroid (subject position)
        if 'keypoints_3d' in frame_data and frame_data['keypoints_3d']:
            keypoints_3d = np.array(frame_data['keypoints_3d'])

            # Use shoulders and hips for centroid (same as visualization)
            # MediaPipe indices: LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_HIP=23, RIGHT_HIP=24, NOSE=0
            valid_points = []
            for idx in [0, 11, 12, 23, 24]:  # nose, shoulders, hips
                if idx < len(keypoints_3d):
                    kp = keypoints_3d[idx]
                    # Check if valid (not [0,0,0])
                    if not (kp[0] == 0 and kp[1] == 0 and kp[2] == 0):
                        valid_points.append(kp)

            if len(valid_points) >= 2:
                centroid = np.mean(valid_points, axis=0)
                timestamps.append(frame_num)
                positions.append(centroid)

    return timestamps, positions


def compute_distances_to_targets(positions: List[np.ndarray], targets: List[Dict]) -> Dict[str, List[float]]:
    """
    Compute distances from subject position to each target over time.

    Returns:
        Dictionary mapping target labels to lists of distances
    """
    distances = {}

    for target in targets:
        if 'x' in target and 'y' in target and 'z' in target:
            label = target.get('label', 'target')
            target_pos = np.array([target['x'], target['y'], target['z']])

            target_distances = []
            for pos in positions:
                dist = np.linalg.norm(pos - target_pos)
                target_distances.append(dist)

            distances[label] = target_distances

    return distances


def plot_2d_trace(positions: List[np.ndarray], targets: Optional[List[Dict]],
                  subject_name: str, output_path: Path):
    """
    Generate 2D top-down trace plot (XZ plane).

    Args:
        positions: List of 3D positions
        targets: List of target dictionaries with x, y, z, label
        subject_name: Name of subject (e.g., "Dog", "Baby")
        output_path: Where to save the PNG file
    """
    positions_array = np.array(positions)

    # Create colormap for trajectory (time progression)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(positions)))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory with color gradient
    for i in range(len(positions) - 1):
        ax.plot(positions_array[i:i+2, 0], positions_array[i:i+2, 2],
               color=colors[i], linewidth=2, alpha=0.7)

    # Plot start and end markers
    ax.scatter(positions_array[0, 0], positions_array[0, 2],
              c='cyan', marker='o', s=200, edgecolors='black', linewidths=2,
              label='Start', zorder=10)
    ax.scatter(positions_array[-1, 0], positions_array[-1, 2],
              c='lime', marker='o', s=200, edgecolors='black', linewidths=2,
              label='End', zorder=10)

    # Plot targets if available
    if targets:
        target_positions = []
        target_labels = []
        for target in targets:
            if 'x' in target and 'y' in target and 'z' in target:
                target_positions.append([target['x'], target['z']])  # XZ plane
                target_labels.append(target.get('label', 'target'))

        if target_positions:
            target_positions = np.array(target_positions)
            ax.scatter(target_positions[:, 0], target_positions[:, 1],
                      c='gray', marker='s', s=300, edgecolors='black', linewidths=2,
                      label='Targets', zorder=5)

            # Add target labels
            for pos, label in zip(target_positions, target_labels):
                ax.text(pos[0], pos[1], f'  {label}',
                       fontsize=10, fontweight='bold', color='black')

    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)
    ax.set_title(f'2D Trace (Top View) - {subject_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([-2, 1])
    ax.set_ylim([0.5, 3.5])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✅ Saved 2D trace plot: {output_path}")


def plot_distance_to_targets(timestamps: List[float], distances: Dict[str, List[float]],
                             subject_name: str, output_path: Path,
                             fps: float = 30.0):
    """
    Generate distance-to-targets over time plot.

    Args:
        timestamps: List of frame numbers
        distances: Dictionary mapping target labels to distance lists
        subject_name: Name of subject (e.g., "Dog", "Baby")
        output_path: Where to save the PNG file
        fps: Frames per second for time conversion
    """
    # Convert frame numbers to time in seconds
    times = np.array(timestamps) / fps

    fig, ax = plt.subplots(figsize=(12, 6))

    # Define colors for each target
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    # Plot distance for each target
    for idx, (label, dist_values) in enumerate(sorted(distances.items())):
        color = colors[idx % len(colors)]
        ax.plot(times, dist_values, marker='o', linewidth=2,
               markersize=4, label=label, color=color, alpha=0.8)

    # Add horizontal line at threshold (e.g., 0.3m for close interaction)
    threshold = 0.3
    ax.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
              label='Threshold', alpha=0.7)

    # Mark human position if available (gray X markers)
    # This would be added if human data is also processed

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Distance (meters)', fontsize=12)
    ax.set_title(f'{subject_name} - Distance to Targets Over Time', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"✅ Saved distance plot: {output_path}")


def generate_subject_plots(trial_path: Path, subject_type: str = 'dog',
                          subject_name: str = None, fps: float = 30.0,
                          output_path_override: Path = None):
    """
    Generate both 2D trace and distance plots for a subject.

    Args:
        trial_path: Path to trial_input folder (e.g., trial_input/trial_1/cam1)
        subject_type: 'dog' or 'baby'
        subject_name: Display name for subject (default: capitalized subject_type)
        fps: Frames per second for timestamp conversion
        output_path_override: Optional override for output path (defaults to trial_output)
    """
    if subject_name is None:
        subject_name = subject_type.capitalize()

    # Determine output path (trial_output) if not provided
    if output_path_override is None:
        camera_name = trial_path.name
        trial_name = trial_path.parent.name

        if camera_name == trial_name:
            output_path = Path("trial_output") / camera_name
        else:
            output_path = Path("trial_output") / trial_name / camera_name
    else:
        output_path = output_path_override

    # Load subject results
    json_file = output_path / f"{subject_type}_detection_results.json"
    if not json_file.exists():
        print(f"⚠️ No {subject_type} results found: {json_file}")
        print(f"   Expected location: {json_file.absolute()}")
        return

    print(f"\n📊 Generating plots for {subject_name}...")
    print(f"   Reading from: {json_file}")

    try:
        results = load_subject_results(json_file)
    except Exception as e:
        print(f"❌ Failed to load results: {e}")
        return

    # Extract trajectory
    timestamps, positions = extract_subject_trajectory(results)

    if len(positions) == 0:
        print(f"⚠️ No valid {subject_type} positions found")
        return

    print(f"  Found {len(positions)} valid positions")

    # Load targets
    targets = load_targets(trial_path)
    if targets is None:
        print(f"  ⚠️ No targets file found in {trial_path}")
    else:
        print(f"  Found {len(targets)} targets")

    # Generate 2D trace plot
    trace_output = output_path / f"processed_{subject_type}_result_trace2d.png"
    plot_2d_trace(positions, targets, subject_name, trace_output)

    # Generate distance plot if targets available
    if targets:
        distances = compute_distances_to_targets(positions, targets)
        distance_output = output_path / f"processed_{subject_type}_result_distance.png"
        plot_distance_to_targets(timestamps, distances, subject_name,
                                distance_output, fps=fps)
    else:
        print(f"  ⚠️ Skipping distance plot (no targets)")

    print(f"✅ Plots generated successfully\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plots for dog/baby subject data")
    parser.add_argument("--trial", type=str, required=True,
                       help="Path to trial folder (e.g., trial_input/trial_1/cam1)")
    parser.add_argument("--subject", type=str, choices=['dog', 'baby'], required=True,
                       help="Subject type: dog or baby")
    parser.add_argument("--name", type=str, default=None,
                       help="Display name for subject (default: Dog or Baby)")
    parser.add_argument("--fps", type=float, default=30.0,
                       help="Frames per second (default: 30.0)")

    args = parser.parse_args()

    trial_path = Path(args.trial)
    generate_subject_plots(trial_path, args.subject, args.name, args.fps)

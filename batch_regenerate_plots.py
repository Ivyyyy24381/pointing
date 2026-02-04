#!/usr/bin/env python3
"""
Batch Regenerate Visualizations

Regenerates all visualization plots from existing JSON/CSV output data.
Use this to update plots with new axis ranges or styling without re-processing data.

Usage:
    # Regenerate all plots for a study
    python batch_regenerate_plots.py /path/to/study_output

    # Regenerate with custom Z range
    python batch_regenerate_plots.py /path/to/study_output --z-min 2.0 --z-max 5.0

    # Regenerate only pointing traces
    python batch_regenerate_plots.py /path/to/study_output --only pointing

    # Regenerate for specific trial/camera
    python batch_regenerate_plots.py /path/to/study_output --trial trial_6 --camera cam1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# Default axis ranges (can be overridden via CLI)
DEFAULT_X_RANGE = (-1.5, 1.5)
DEFAULT_Z_RANGE = (2.0, 5.0)


def load_targets(output_path: Path) -> list:
    """Load targets from JSON file."""
    target_file = output_path / "target_detections_cam_frame.json"
    if not target_file.exists():
        return []
    with open(target_file) as f:
        return json.load(f)


def load_skeleton_data(output_path: Path) -> dict:
    """Load skeleton detection results."""
    skeleton_file = output_path / "skeleton_2d.json"
    if not skeleton_file.exists():
        return {}
    with open(skeleton_file) as f:
        return json.load(f)


def parse_list_string(s):
    """Parse a string representation of a list like '[1.0, 2.0, 3.0]' to actual list."""
    if pd.isna(s) or s is None:
        return None
    if isinstance(s, (list, tuple)):
        return list(s)
    if isinstance(s, str):
        s = s.strip()
        # Handle None values in string
        if 'None' in s or 'nan' in s.lower():
            return None
        if s.startswith('[') and s.endswith(']'):
            try:
                import ast
                result = ast.literal_eval(s)
                if isinstance(result, (list, tuple)):
                    # Check for None values
                    if any(v is None for v in result):
                        return None
                    return [float(v) for v in result]
            except:
                pass
        # Try comma-separated
        try:
            parts = s.strip('[]()').split(',')
            vals = []
            for p in parts:
                p = p.strip()
                if p and p.lower() != 'none' and p.lower() != 'nan':
                    vals.append(float(p))
                else:
                    return None  # Has invalid value
            return vals if vals else None
        except:
            pass
    return None


def load_pointing_analyses(output_path: Path) -> dict:
    """Load pointing analysis from CSV and convert to dict format."""
    csv_file = output_path / "processed_gesture.csv"
    if not csv_file.exists():
        return {}

    df = pd.read_csv(csv_file)
    analyses = {}

    for _, row in df.iterrows():
        frame_num = int(row.get('frame', row.get('frame_number', 0)))
        frame_key = f"frame_{frame_num:06d}"

        analysis = {}

        # Extract ground intersections for each vector type
        # CSV stores as single column with list: '[x, y, z]'
        for vec_type in ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist', 'head_orientation']:
            col = f'{vec_type}_ground_intersection'

            if col in df.columns:
                val = row.get(col)
                parsed = parse_list_string(val)

                if parsed and len(parsed) >= 3:
                    # Check if values are valid (not None/NaN)
                    if parsed[0] is not None and parsed[2] is not None:
                        analysis[col] = parsed
                    else:
                        analysis[col] = [None, None, None]
                else:
                    analysis[col] = [None, None, None]

            # Extract distances
            for i in range(1, 5):
                dist_col = f'{vec_type}_dist_to_target_{i}'
                if dist_col in df.columns:
                    val = row.get(dist_col)
                    analysis[dist_col] = float(val) if pd.notna(val) else None

        if analysis:
            analyses[frame_key] = analysis

    return analyses


def load_subject_data(output_path: Path, subject_type: str = 'dog') -> dict:
    """Load subject (dog/baby) detection results."""
    subject_file = output_path / f"{subject_type}_detection_results.json"
    if not subject_file.exists():
        return {}
    with open(subject_file) as f:
        return json.load(f)


def load_human_center(output_path: Path) -> list:
    """Load human center position."""
    hc_file = output_path / "human_center.json"
    if not hc_file.exists():
        return [0, 0, 0]
    with open(hc_file) as f:
        data = json.load(f)
        return data.get('human_center', [0, 0, 0])


def regenerate_pointing_trace(output_path: Path, x_range: tuple, z_range: tuple):
    """Regenerate 2D pointing trace plot."""
    from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace

    analyses = load_pointing_analyses(output_path)
    targets = load_targets(output_path)
    human_center = load_human_center(output_path)

    if not analyses:
        print(f"    No pointing analysis data found in CSV")
        return False

    # Count valid data points
    valid_count = 0
    for frame_key, analysis in analyses.items():
        for vec_type in ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']:
            intersection = analysis.get(f'{vec_type}_ground_intersection')
            if intersection and intersection[0] is not None:
                valid_count += 1
                break  # Count frame only once

    print(f"    Loaded {len(analyses)} frames, {valid_count} with valid intersections, {len(targets)} targets")

    if valid_count == 0:
        print(f"    WARNING: No valid ground intersections found in data")

    # Get trial/camera name from path
    camera_name = output_path.name
    trial_name = output_path.parent.name

    plot_path = output_path / "2d_pointing_trace.png"
    plot_2d_pointing_trace(
        analyses, targets, human_center, plot_path,
        trial_name=f"{trial_name}_{camera_name}",
        fixed_xlim=x_range,
        fixed_zlim=z_range,
        use_fixed_axes=True
    )
    return True


def regenerate_subject_trace(output_path: Path, subject_type: str, x_range: tuple, z_range: tuple):
    """Regenerate subject (dog/baby) trace plot."""
    from step3_subject_extraction.dog_trace_visualizer import DogTraceVisualizer

    subject_file = output_path / f"{subject_type}_detection_results.json"
    if not subject_file.exists():
        return False

    targets_file = output_path / "target_detections_cam_frame.json"

    # Update the default ranges in the visualizer
    import step3_subject_extraction.dog_trace_visualizer as dtv
    dtv.DEFAULT_X_RANGE = x_range
    dtv.DEFAULT_Z_RANGE = z_range

    visualizer = DogTraceVisualizer(use_fixed_axes=True)

    # Get trial/camera name from path
    camera_name = output_path.name
    trial_name = output_path.parent.name

    trace_path = output_path / f"{subject_type}_result_trace2d.png"
    visualizer.create_trace_plot(
        dog_results_path=subject_file,
        targets_path=targets_file if targets_file.exists() else None,
        output_image_path=trace_path,
        title=f"2D Trace (Top View) - {subject_type.capitalize()} - {trial_name}/{camera_name}"
    )
    return True


def regenerate_distance_plots(output_path: Path):
    """Regenerate distance-to-target plots."""
    from step2_skeleton_extraction.plot_distance_to_targets import (
        plot_distance_to_targets,
        plot_distance_summary,
        plot_best_representation_analysis
    )

    analyses = load_pointing_analyses(output_path)
    targets = load_targets(output_path)

    if not analyses or not targets:
        print(f"    No data for distance plots (analyses: {len(analyses) if analyses else 0}, targets: {len(targets) if targets else 0})")
        return False

    # Count valid distance data points
    valid_dist_count = 0
    for frame_key, analysis in analyses.items():
        for vec_type in ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist']:
            if analysis.get(f'{vec_type}_dist_to_target_1') is not None:
                valid_dist_count += 1
                break

    print(f"    Distance data: {len(analyses)} frames, {valid_dist_count} with valid distances, {len(targets)} targets")

    camera_name = output_path.name
    trial_name = output_path.parent.name

    # Time series plot
    dist_plot_path = output_path / "distance_to_targets_timeseries.png"
    plot_distance_to_targets(
        analyses, targets, dist_plot_path,
        trial_name=f"{trial_name}_{camera_name}",
        show_filtered=True
    )

    # Summary bar chart
    dist_summary_path = output_path / "distance_to_targets_summary.png"
    plot_distance_summary(
        analyses, targets, dist_summary_path,
        trial_name=f"{trial_name}_{camera_name}"
    )

    # Best representation heatmap
    best_rep_path = output_path / "pointing_accuracy_comparison.png"
    plot_best_representation_analysis(
        analyses, targets, best_rep_path,
        trial_name=f"{trial_name}_{camera_name}"
    )

    return True


def regenerate_verification_plot(output_path: Path, cam_path: Path):
    """Regenerate 3D verification plot."""
    skeleton_data = load_skeleton_data(output_path)
    targets = load_targets(output_path)

    if not skeleton_data or not targets:
        return False

    # Load ground plane rotation
    ground_plane_rotation = None
    transform_file = output_path / "ground_plane_transform.json"
    if transform_file.exists():
        with open(transform_file) as f:
            transform_data = json.load(f)
            ground_plane_rotation = np.array(transform_data['rotation_matrix'])

    # This requires the original batch_process_study function
    # For simplicity, skip this one as it requires more complex setup
    print(f"    Skipping verification plot (requires original frame data)")
    return False


def regenerate_subject_distance_plot(output_path: Path, subject_type: str):
    """Regenerate subject distance to targets plot."""
    from step3_subject_extraction.dog_distance_plotter import DogDistancePlotter

    csv_path = output_path / f"processed_{subject_type}_result_table.csv"
    if not csv_path.exists():
        return False

    distance_path = output_path / f"{subject_type}_distance_to_targets.png"

    camera_name = output_path.name
    trial_name = output_path.parent.name

    plotter = DogDistancePlotter()
    plotter.create_distance_plot(
        csv_path=csv_path,
        output_image_path=distance_path,
        title=f"{subject_type.capitalize()} Distance to Targets - {trial_name}/{camera_name}"
    )
    return True


def process_camera_output(output_path: Path, cam_path: Path = None,
                          x_range: tuple = DEFAULT_X_RANGE,
                          z_range: tuple = DEFAULT_Z_RANGE,
                          plot_types: list = None,
                          subject_type: str = 'dog'):
    """Regenerate all plots for a single camera output folder."""
    print(f"\n  Processing: {output_path}")

    if plot_types is None:
        plot_types = ['pointing', 'subject', 'distance', 'subject_distance']

    results = {}

    # Pointing trace
    if 'pointing' in plot_types or 'all' in plot_types:
        try:
            if regenerate_pointing_trace(output_path, x_range, z_range):
                results['pointing_trace'] = 'OK'
            else:
                results['pointing_trace'] = 'SKIP (no data)'
        except Exception as e:
            results['pointing_trace'] = f'ERROR: {e}'

    # Subject trace
    if 'subject' in plot_types or 'all' in plot_types:
        try:
            if regenerate_subject_trace(output_path, subject_type, x_range, z_range):
                results['subject_trace'] = 'OK'
            else:
                results['subject_trace'] = 'SKIP (no data)'
        except Exception as e:
            results['subject_trace'] = f'ERROR: {e}'

    # Distance plots
    if 'distance' in plot_types or 'all' in plot_types:
        try:
            if regenerate_distance_plots(output_path):
                results['distance_plots'] = 'OK'
            else:
                results['distance_plots'] = 'SKIP (no data)'
        except Exception as e:
            results['distance_plots'] = f'ERROR: {e}'

    # Subject distance plot
    if 'subject_distance' in plot_types or 'all' in plot_types:
        try:
            if regenerate_subject_distance_plot(output_path, subject_type):
                results['subject_distance'] = 'OK'
            else:
                results['subject_distance'] = 'SKIP (no data)'
        except Exception as e:
            results['subject_distance'] = f'ERROR: {e}'

    # Print results
    for plot_name, status in results.items():
        print(f"    {plot_name}: {status}")

    return results


def find_camera_outputs(study_output_path: Path, trial_filter: str = None,
                        camera_filter: str = None) -> list:
    """Find all camera output folders in a study output directory."""
    camera_outputs = []

    for trial_dir in sorted(study_output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue

        if trial_filter and trial_dir.name != trial_filter:
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue

            if camera_filter and cam_dir.name != camera_filter:
                continue

            camera_outputs.append(cam_dir)

    return camera_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate visualization plots from existing output data."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to study_output folder (e.g., /path/to/study_name_output)"
    )
    parser.add_argument(
        "--trial", type=str, default=None,
        help="Process only this trial (e.g., trial_6)"
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Process only this camera (e.g., cam1)"
    )
    parser.add_argument(
        "--x-min", type=float, default=DEFAULT_X_RANGE[0],
        help=f"Minimum X axis value (default: {DEFAULT_X_RANGE[0]})"
    )
    parser.add_argument(
        "--x-max", type=float, default=DEFAULT_X_RANGE[1],
        help=f"Maximum X axis value (default: {DEFAULT_X_RANGE[1]})"
    )
    parser.add_argument(
        "--z-min", type=float, default=DEFAULT_Z_RANGE[0],
        help=f"Minimum Z axis value (default: {DEFAULT_Z_RANGE[0]})"
    )
    parser.add_argument(
        "--z-max", type=float, default=DEFAULT_Z_RANGE[1],
        help=f"Maximum Z axis value (default: {DEFAULT_Z_RANGE[1]})"
    )
    parser.add_argument(
        "--only", type=str, nargs='+', default=None,
        choices=['pointing', 'subject', 'distance', 'subject_distance', 'all'],
        help="Only regenerate specific plot types"
    )
    parser.add_argument(
        "--subject", type=str, default='dog', choices=['dog', 'baby'],
        help="Subject type for trace plots (default: dog)"
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    if not output_path.exists():
        print(f"ERROR: Output path does not exist: {output_path}")
        sys.exit(1)

    x_range = (args.x_min, args.x_max)
    z_range = (args.z_min, args.z_max)
    plot_types = args.only if args.only else ['all']

    print(f"\n{'='*60}")
    print(f"  Batch Regenerate Visualizations")
    print(f"{'='*60}")
    print(f"  Output path: {output_path}")
    print(f"  X range: {x_range}")
    print(f"  Z range: {z_range}")
    print(f"  Plot types: {plot_types}")
    print(f"  Subject: {args.subject}")
    print(f"{'='*60}")

    # Find all camera output folders
    camera_outputs = find_camera_outputs(output_path, args.trial, args.camera)

    if not camera_outputs:
        print(f"\nNo camera output folders found in {output_path}")
        sys.exit(1)

    print(f"\nFound {len(camera_outputs)} camera output folder(s)")

    # Process each camera
    total_results = {'OK': 0, 'SKIP': 0, 'ERROR': 0}

    for cam_output in camera_outputs:
        results = process_camera_output(
            cam_output,
            x_range=x_range,
            z_range=z_range,
            plot_types=plot_types,
            subject_type=args.subject
        )

        for status in results.values():
            if status == 'OK':
                total_results['OK'] += 1
            elif status.startswith('SKIP'):
                total_results['SKIP'] += 1
            else:
                total_results['ERROR'] += 1

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  OK: {total_results['OK']} | SKIP: {total_results['SKIP']} | ERROR: {total_results['ERROR']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

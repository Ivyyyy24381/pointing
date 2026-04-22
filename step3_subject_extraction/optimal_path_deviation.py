#!/usr/bin/env python3
"""
Optimal Path Deviation Calculator for Dog Trajectory

Computes how much the dog deviates from the straight-line (optimal) path
to its first-choice target. Uses 2D top-down view (X-Z plane).

Supports two input formats:
  1. Current pipeline: dog_detection_results.json + target_detections_cam_frame.json
  2. Old pipeline CSV: processed_dog_result_table.csv (or similar with trace3d_x/z, target_*_r)

Output CSV columns:
  frame_index, time_sec, dog_x, dog_z, target_x, target_z, target_label,
  optimal_path_distance, cumulative_distance_traveled, distance_to_target,
  signed_deviation (positive = right of path, negative = left)

Usage:
    # Current pipeline (auto-detects first-choice target from distance data)
    python optimal_path_deviation.py /path/to/trial/cam1/

    # Old pipeline CSV
    python optimal_path_deviation.py /path/to/processed_dog_result_table.csv

    # Specify target explicitly (1-4)
    python optimal_path_deviation.py /path/to/trial/cam1/ --target 2

    # Batch: process all trials in a study
    python optimal_path_deviation.py /path/to/study_output/ --batch

    # Custom start position (x, z in meters)
    python optimal_path_deviation.py /path/to/trial/cam1/ --start-pos 0.25 2.0
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict


def point_to_line_distance(px: float, pz: float,
                           ax: float, az: float,
                           bx: float, bz: float) -> Tuple[float, float]:
    """
    Perpendicular distance from point P to line segment A->B in 2D (X-Z plane).

    Returns:
        (unsigned_distance, signed_distance)
        signed: positive = right of A->B direction, negative = left
    """
    # Vector AB
    abx = bx - ax
    abz = bz - az
    ab_len = np.sqrt(abx**2 + abz**2)
    if ab_len < 1e-9:
        d = np.sqrt((px - ax)**2 + (pz - az)**2)
        return d, d

    # Vector AP
    apx = px - ax
    apz = pz - az

    # Cross product (AB x AP) gives signed area
    cross = abx * apz - abz * apx
    signed_dist = cross / ab_len

    return abs(signed_dist), signed_dist


def detect_first_choice_target(dog_positions: List[Dict],
                               targets: List[Dict],
                               approach_threshold: float = 0.3) -> Optional[Dict]:
    """
    Detect which target the dog approaches first.

    Strategy: find the first frame where dog is within approach_threshold
    of any target. That target is the first choice.
    If dog never reaches threshold, use the target with minimum distance
    at the frame of closest approach.
    """
    if not targets or not dog_positions:
        return None

    min_dist_overall = float('inf')
    first_choice = None

    for pos in dog_positions:
        dx, dz = pos['x'], pos['z']
        if np.isnan(dx) or np.isnan(dz):
            continue

        for t in targets:
            tx, tz = t['x'], t['z']
            dist = np.sqrt((dx - tx)**2 + (dz - tz)**2)
            if dist < approach_threshold:
                return t
            if dist < min_dist_overall:
                min_dist_overall = dist
                first_choice = t

    return first_choice


def load_current_pipeline(cam_dir: Path) -> Tuple[List[Dict], List[Dict], Optional[Tuple[float, float]]]:
    """
    Load data from current pipeline output directory.

    Returns: (dog_positions, targets, start_pos_override)
        dog_positions: list of {frame, time_sec, x, z}
        targets: list of {x, z, label}
    """
    # Load dog detection results
    dog_json = cam_dir / "dog_detection_results.json"
    dog_csv = cam_dir / "processed_dog_result_table.csv"
    targets_json = cam_dir / "target_detections_cam_frame.json"

    dog_positions = []
    targets = []

    # Prefer CSV (has time_sec and pre-computed 3D)
    if dog_csv.exists():
        import csv as csv_mod
        with open(dog_csv) as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                x = _safe_float(row.get('trace3d_x') or row.get('nose_x_m'))
                z = _safe_float(row.get('trace3d_z') or row.get('nose_z_m'))
                dog_positions.append({
                    'frame': int(float(row.get('frame_index', 0))),
                    'time_sec': _safe_float(row.get('time_sec', 0)),
                    'x': x,
                    'z': z,
                })
    elif dog_json.exists():
        with open(dog_json) as f:
            data = json.load(f)
        for frame_key in sorted(data.keys()):
            entry = data[frame_key]
            kp3d = entry.get('keypoints_3d', [])
            if kp3d and len(kp3d[0]) >= 3:
                nose = kp3d[0]
                frame_num = int(frame_key.split('_')[-1])
                dog_positions.append({
                    'frame': frame_num,
                    'time_sec': frame_num / 30.0,
                    'x': nose[0],
                    'z': nose[2],
                })

    # Load targets
    if targets_json.exists():
        with open(targets_json) as f:
            tdata = json.load(f)
        if isinstance(tdata, list):
            for t in tdata:
                targets.append({
                    'x': t['x'],
                    'z': t['z'],
                    'label': t.get('label', 'unknown'),
                })

    return dog_positions, targets, None


def load_old_pipeline_csv(csv_path: Path) -> Tuple[List[Dict], List[Dict], Optional[Tuple[float, float]]]:
    """
    Load data from old pipeline CSV format.

    Old CSV has columns: frame_index, time_sec, nose_x, nose_y,
    trace3d_x, trace3d_y, trace3d_z, target_1_r, target_2_r, ...

    Returns: (dog_positions, targets, start_pos_override)
    """
    import csv as csv_mod

    dog_positions = []
    target_distances = {1: [], 2: [], 3: [], 4: []}

    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        columns = reader.fieldnames or []

        for row in reader:
            x = _safe_float(row.get('trace3d_x') or row.get('nose_x_m'))
            z = _safe_float(row.get('trace3d_z') or row.get('nose_z_m'))
            dog_positions.append({
                'frame': int(float(row.get('frame_index', 0))),
                'time_sec': _safe_float(row.get('time_sec', 0)),
                'x': x,
                'z': z,
            })
            for i in range(1, 5):
                r = _safe_float(row.get(f'target_{i}_r'))
                if not np.isnan(r):
                    target_distances[i].append(r)

    # Try to load targets from sibling target_detections_cam_frame.json
    targets = []
    targets_json = csv_path.parent / "target_detections_cam_frame.json"
    if targets_json.exists():
        with open(targets_json) as f:
            tdata = json.load(f)
        if isinstance(tdata, list):
            for t in tdata:
                targets.append({
                    'x': t['x'],
                    'z': t['z'],
                    'label': t.get('label', 'unknown'),
                })

    return dog_positions, targets, None


def _safe_float(val) -> float:
    if val is None or val == '' or val == 'nan':
        return float('nan')
    try:
        return float(val)
    except (ValueError, TypeError):
        return float('nan')


def compute_optimal_path_deviation(
    dog_positions: List[Dict],
    target: Dict,
    start_pos: Optional[Tuple[float, float]] = None,
) -> List[Dict]:
    """
    Compute per-frame deviation from the optimal straight-line path.

    Args:
        dog_positions: list of {frame, time_sec, x, z}
        target: {x, z, label} of the first-choice target
        start_pos: (x, z) override for start position. If None, uses first valid dog position.

    Returns:
        list of dicts with deviation data per frame
    """
    # Find start position (first valid dog position)
    if start_pos is None:
        for pos in dog_positions:
            if not np.isnan(pos['x']) and not np.isnan(pos['z']):
                start_pos = (pos['x'], pos['z'])
                break
    if start_pos is None:
        return []

    sx, sz = start_pos
    tx, tz = target['x'], target['z']

    results = []
    cumulative_dist = 0.0
    prev_x, prev_z = None, None

    for pos in dog_positions:
        dx, dz = pos['x'], pos['z']

        if np.isnan(dx) or np.isnan(dz):
            results.append({
                'frame_index': pos['frame'],
                'time_sec': pos['time_sec'],
                'dog_x': float('nan'),
                'dog_z': float('nan'),
                'target_x': tx,
                'target_z': tz,
                'target_label': target['label'],
                'start_x': sx,
                'start_z': sz,
                'optimal_path_distance': float('nan'),
                'signed_deviation': float('nan'),
                'cumulative_distance_traveled': float('nan'),
                'distance_to_target': float('nan'),
            })
            continue

        # Cumulative distance traveled
        if prev_x is not None and not np.isnan(prev_x):
            step = np.sqrt((dx - prev_x)**2 + (dz - prev_z)**2)
            cumulative_dist += step
        prev_x, prev_z = dx, dz

        # Perpendicular distance to optimal path line (start -> target)
        unsigned_dev, signed_dev = point_to_line_distance(dx, dz, sx, sz, tx, tz)

        # Distance from current position to target
        dist_to_target = np.sqrt((dx - tx)**2 + (dz - tz)**2)

        results.append({
            'frame_index': pos['frame'],
            'time_sec': pos['time_sec'],
            'dog_x': dx,
            'dog_z': dz,
            'target_x': tx,
            'target_z': tz,
            'target_label': target['label'],
            'start_x': sx,
            'start_z': sz,
            'optimal_path_distance': unsigned_dev,
            'signed_deviation': signed_dev,
            'cumulative_distance_traveled': cumulative_dist,
            'distance_to_target': dist_to_target,
        })

    return results


def save_deviation_csv(results: List[Dict], output_path: Path):
    """Save deviation results to CSV."""
    if not results:
        print(f"  No results to save")
        return

    fieldnames = [
        'frame_index', 'time_sec', 'dog_x', 'dog_z',
        'target_x', 'target_z', 'target_label',
        'start_x', 'start_z',
        'optimal_path_distance', 'signed_deviation',
        'cumulative_distance_traveled', 'distance_to_target',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    valid = [r for r in results if not np.isnan(r['optimal_path_distance'])]
    if valid:
        devs = [r['optimal_path_distance'] for r in valid]
        print(f"  Saved: {output_path.name}")
        print(f"  Target: {results[0]['target_label']}")
        print(f"  Frames: {len(valid)} valid / {len(results)} total")
        print(f"  Deviation — mean: {np.mean(devs):.4f}m, "
              f"max: {np.max(devs):.4f}m, "
              f"median: {np.median(devs):.4f}m")
        print(f"  Path length: {valid[-1]['cumulative_distance_traveled']:.3f}m "
              f"(optimal: {np.sqrt((results[0]['target_x'] - results[0]['start_x'])**2 + (results[0]['target_z'] - results[0]['start_z'])**2):.3f}m)")


def process_trial(input_path: Path,
                  target_num: Optional[int] = None,
                  start_pos: Optional[Tuple[float, float]] = None,
                  output_path: Optional[Path] = None,
                  approach_threshold: float = 0.3) -> Optional[Path]:
    """
    Process a single trial.

    Args:
        input_path: Path to cam dir (current pipeline) or CSV file (old pipeline)
        target_num: Override first-choice target (1-4). None = auto-detect.
        start_pos: Override start position (x, z). None = first dog position.
        output_path: Override output CSV path.
        approach_threshold: Distance threshold for first-choice detection.

    Returns:
        Path to output CSV, or None on failure.
    """
    # Detect input format
    if input_path.is_file() and input_path.suffix == '.csv':
        print(f"  Loading old pipeline CSV: {input_path.name}")
        dog_positions, targets, _ = load_old_pipeline_csv(input_path)
    elif input_path.is_dir():
        print(f"  Loading current pipeline: {input_path}")
        dog_positions, targets, _ = load_current_pipeline(input_path)
    else:
        print(f"  ERROR: {input_path} is not a directory or CSV file")
        return None

    if not dog_positions:
        print(f"  WARNING: No dog position data found")
        return None

    valid_count = sum(1 for p in dog_positions if not np.isnan(p['x']))
    print(f"  Dog positions: {valid_count} valid / {len(dog_positions)} total")

    if not targets:
        print(f"  WARNING: No target data found")
        return None

    print(f"  Targets: {len(targets)} ({', '.join(t['label'] for t in targets)})")

    # Determine first-choice target
    if target_num is not None:
        label = f"target_{target_num}"
        target = next((t for t in targets if t['label'] == label), None)
        if target is None:
            print(f"  ERROR: Target {label} not found")
            return None
        print(f"  Target (manual): {label}")
    else:
        target = detect_first_choice_target(dog_positions, targets, approach_threshold)
        if target is None:
            print(f"  WARNING: Could not detect first-choice target")
            return None
        print(f"  Target (auto-detected first choice): {target['label']}")

    # Compute deviation
    results = compute_optimal_path_deviation(dog_positions, target, start_pos)

    # Save
    if output_path is None:
        if input_path.is_file():
            output_path = input_path.parent / "optimal_path_deviation.csv"
        else:
            output_path = input_path / "optimal_path_deviation.csv"

    save_deviation_csv(results, output_path)
    return output_path


def process_batch(study_path: Path,
                  target_num: Optional[int] = None,
                  start_pos: Optional[Tuple[float, float]] = None,
                  camera: str = "cam1",
                  approach_threshold: float = 0.3):
    """
    Process all trials in a study (or all studies in an output directory).

    Handles both:
      - study_output/trial_*/cam1/  (current pipeline)
      - subject_side/1/, 2/, ...    (old pipeline with CSVs)
    """
    study_path = Path(study_path)

    # Detect structure
    trial_dirs = sorted(study_path.glob("trial_*"))
    old_trial_dirs = sorted([d for d in study_path.iterdir()
                             if d.is_dir() and d.name.isdigit()])

    if trial_dirs:
        # Current pipeline: study_output/trial_*/cam1/
        print(f"\nBatch processing (current pipeline): {study_path.name}")
        print(f"Found {len(trial_dirs)} trials\n")
        for td in trial_dirs:
            cam_dir = td / camera
            if not cam_dir.exists():
                continue
            print(f"\n--- {td.name}/{camera} ---")
            process_trial(cam_dir, target_num=target_num,
                         start_pos=start_pos, approach_threshold=approach_threshold)
    elif old_trial_dirs:
        # Old pipeline: subject_side/1/, 2/, ...
        print(f"\nBatch processing (old pipeline): {study_path.name}")
        print(f"Found {len(old_trial_dirs)} trial folders\n")
        for td in old_trial_dirs:
            csvs = list(td.glob("processed_*result*.csv"))
            if not csvs:
                csvs = list(td.glob("*result*.csv"))
            if not csvs:
                continue
            csv_path = csvs[0]
            print(f"\n--- Trial {td.name}: {csv_path.name} ---")
            process_trial(csv_path, target_num=target_num,
                         start_pos=start_pos, approach_threshold=approach_threshold)
    else:
        # Maybe it's an output dir with multiple studies
        study_dirs = sorted([d for d in study_path.iterdir()
                             if d.is_dir() and d.name.endswith('_output')])
        if study_dirs:
            print(f"\nBatch processing all studies in: {study_path}")
            print(f"Found {len(study_dirs)} studies\n")
            for sd in study_dirs:
                print(f"\n{'='*60}")
                print(f"  Study: {sd.name}")
                print(f"{'='*60}")
                process_batch(sd, target_num=target_num, start_pos=start_pos,
                            camera=camera, approach_threshold=approach_threshold)
        else:
            print(f"No trial directories found in {study_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate dog's deviation from optimal path to first-choice target")
    parser.add_argument("input", help="Trial cam dir, CSV file, or study dir (with --batch)")
    parser.add_argument("--target", type=int, choices=[1, 2, 3, 4],
                        help="Override first-choice target (1-4). Default: auto-detect.")
    parser.add_argument("--start-pos", nargs=2, type=float, metavar=('X', 'Z'),
                        help="Override dog start position (x z) in meters")
    parser.add_argument("--batch", action="store_true",
                        help="Process all trials in the directory")
    parser.add_argument("--camera", default="cam1",
                        help="Camera name for batch mode (default: cam1)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Approach distance threshold for first-choice detection (default: 0.3m)")
    parser.add_argument("-o", "--output", help="Output CSV path (single trial mode)")
    args = parser.parse_args()

    input_path = Path(args.input)
    start_pos = tuple(args.start_pos) if args.start_pos else None
    output_path = Path(args.output) if args.output else None

    if args.batch:
        process_batch(input_path, target_num=args.target,
                     start_pos=start_pos, camera=args.camera,
                     approach_threshold=args.threshold)
    else:
        process_trial(input_path, target_num=args.target,
                     start_pos=start_pos, output_path=output_path,
                     approach_threshold=args.threshold)


if __name__ == "__main__":
    main()

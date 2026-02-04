#!/usr/bin/env python3
"""
Batch processing script for pointing gesture analysis.

Processes all trials in a study folder (or multiple studies) end-to-end:
  1. Target detection (YOLO) on the first frame of each trial/camera
  2. Skeleton extraction (MediaPipe) on all frames
  3. Pointing analysis + CSV export + 2D trace plot

Data structure expected:
  study_folder/
    trial_N/
      cam1/  (or cam2, cam3)
        color/
          frame_000001.png, ...
        depth/
          frame_000001.npy, ...

Results are saved to <study_name>_output/ alongside the study folder, and synced back to the original
trial/camera folder.

Usage:
    # Process a single study
    python batch_process_study.py /path/to/study_folder

    # Process a single trial within a study
    python batch_process_study.py /path/to/study_folder --trial trial_6

    # Process only specific camera(s)
    python batch_process_study.py /path/to/study_folder --cameras cam1

    # Skip target detection (if already done)
    python batch_process_study.py /path/to/study_folder --skip-targets

    # Use a specific frame for target detection (default: 1)
    python batch_process_study.py /path/to/study_folder --target-frame 50

    # Reprocess pointing only with arm override CSV
    python batch_process_study.py /path/to/study_folder --pointing-only --arm-csv arms.csv

    # Generate arm override CSV template
    python batch_process_study.py /path/to/study_folder --generate-arm-csv
"""

import os
import sys
import json
import shutil
import argparse
import traceback
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from step0_data_loading.target_detector import (
    TargetDetector, get_default_model_path,
    apply_known_depth_constraint, enforce_coplanarity
)
from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
from step2_skeleton_extraction.batch_processor import determine_pointing_hand_whole_trial

# Default known target depth from experiment setup (config/targets.yaml)
# Targets are typically at ~2.8m from the camera
DEFAULT_TARGET_DEPTH = 2.8

# Target ordering convention (from human's perspective facing camera):
# - Human's LEFT (camera's right, +X): target_1, target_2
# - Human's RIGHT (camera's left, -X): target_3, target_4

# SAM3 detector (lazy-loaded)
_sam3_detector = None

def get_sam3_detector():
    """Get or create singleton SAM3 detector."""
    global _sam3_detector
    if _sam3_detector is None:
        from step3_subject_extraction.sam3_detector import SAM3Detector
        _sam3_detector = SAM3Detector()
    return _sam3_detector


def get_camera_intrinsics(width, height):
    """Auto-detect camera intrinsics from image resolution."""
    if width == 640 and height == 480:
        return 615.0, 615.0, 320.0, 240.0
    elif width == 1280 and height == 720:
        return 922.5, 922.5, 640.0, 360.0
    elif width == 1920 and height == 1080:
        return 1383.75, 1383.75, 960.0, 540.0
    else:
        fx = fy = width * 0.9
        return fx, fy, width / 2.0, height / 2.0


# =============================================================================
# ARM OVERRIDE CSV FUNCTIONS
# =============================================================================

def generate_arm_csv_template(study_path: Path, output_csv: Path = None) -> Path:
    """
    Generate a CSV template for arm overrides from existing study data.

    Scans the study output folder and creates a CSV with columns:
    - study: Study/bag name
    - trial: Trial name (e.g., trial_1)
    - camera: Camera name (e.g., cam1)
    - detected_arm: Currently detected pointing arm
    - override_arm: Column for manual override (left/right/skip)
    - reprocess: Whether to reprocess this trial (yes/no)

    Args:
        study_path: Path to study folder
        output_csv: Output CSV path (default: <study_name>_arm_overrides.csv)

    Returns:
        Path to generated CSV
    """
    import pandas as pd

    study_path = Path(study_path)
    study_name = study_path.name

    # Check output folder
    output_folder = study_path.parent / f"{study_name}_output"

    rows = []

    # Scan study folder for trials
    trial_dirs = sorted([d for d in study_path.iterdir()
                        if d.is_dir() and d.name.startswith('trial_')])

    for trial_dir in trial_dirs:
        trial_name = trial_dir.name

        # Find cameras
        cam_dirs = sorted([d for d in trial_dir.iterdir()
                          if d.is_dir() and d.name.startswith('cam')])

        for cam_dir in cam_dirs:
            camera_name = cam_dir.name

            # Try to load existing pointing hand detection
            detected_arm = "unknown"
            output_cam = output_folder / trial_name / camera_name
            pointing_file = output_cam / "pointing_hand.json"

            if pointing_file.exists():
                try:
                    with open(pointing_file) as f:
                        data = json.load(f)
                        detected_arm = data.get('pointing_hand', 'unknown')
                except Exception:
                    pass

            rows.append({
                'study': study_name,
                'trial': trial_name,
                'camera': camera_name,
                'detected_arm': detected_arm,
                'override_arm': '',  # User fills this in
                'reprocess': 'no'    # User marks 'yes' to reprocess
            })

    df = pd.DataFrame(rows)

    # Default output path
    if output_csv is None:
        output_csv = study_path.parent / f"{study_name}_arm_overrides.csv"

    df.to_csv(output_csv, index=False)
    print(f"✅ Generated arm override CSV: {output_csv}")
    print(f"   {len(rows)} trial/camera combinations")
    print(f"\n   Edit the CSV to:")
    print(f"   - Set 'override_arm' to 'left' or 'right' (or 'skip' to skip)")
    print(f"   - Set 'reprocess' to 'yes' for trials to reprocess")
    print(f"\n   Then run:")
    print(f"   python batch_process_study.py {study_path} --pointing-only --arm-csv {output_csv}")

    return output_csv


def load_arm_overrides(csv_path: Path) -> dict:
    """
    Load arm overrides from CSV file.

    Args:
        csv_path: Path to arm override CSV

    Returns:
        Dict mapping (study, trial, camera) -> {
            'override_arm': 'left'/'right'/'skip'/None,
            'reprocess': bool
        }
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Arm override CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    overrides = {}
    for _, row in df.iterrows():
        study = str(row.get('study', '')).strip()
        trial = str(row.get('trial', '')).strip()
        camera = str(row.get('camera', '')).strip()

        if not trial:
            continue

        # Parse override_arm
        override_arm = str(row.get('override_arm', '')).strip().lower()
        if override_arm not in ['left', 'right', 'skip']:
            override_arm = None  # Use auto-detection

        # Parse reprocess flag
        reprocess = str(row.get('reprocess', '')).strip().lower()
        reprocess = reprocess in ['yes', 'true', '1', 'y']

        key = (study, trial, camera)
        overrides[key] = {
            'override_arm': override_arm,
            'reprocess': reprocess
        }

    return overrides


def get_arm_override(overrides: dict, study_name: str, trial_name: str, camera_name: str) -> dict:
    """
    Get arm override for a specific trial/camera.

    Args:
        overrides: Dict from load_arm_overrides()
        study_name: Study name
        trial_name: Trial name (e.g., 'trial_1')
        camera_name: Camera name (e.g., 'cam1')

    Returns:
        Dict with 'override_arm' and 'reprocess', or None if not found
    """
    # Try exact match
    key = (study_name, trial_name, camera_name)
    if key in overrides:
        return overrides[key]

    # Try without study name (for single-study CSV)
    key = ('', trial_name, camera_name)
    if key in overrides:
        return overrides[key]

    # Try with any study
    for k, v in overrides.items():
        if k[1] == trial_name and k[2] == camera_name:
            return v

    return None


def load_targets_from_config(config_path: Path = None) -> list:
    """
    Load target positions from YAML config file.

    Args:
        config_path: Path to targets.yaml. If None, looks in config/targets.yaml

    Returns:
        List of target dicts with 'id', 'x', 'y', 'z' keys, or None if not found
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "targets.yaml"

    if not config_path.exists():
        return None

    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)

        targets = []
        for t in data.get('targets', []):
            pos = t.get('position_m', [0, 0, 0])
            targets.append({
                'id': t.get('id', f"target_{len(targets)+1}"),
                'label': t.get('id', f"target_{len(targets)+1}"),
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'depth_m': t.get('depth_m', pos[2]),
                'center_px': t.get('center', [0, 0]),
            })

        print(f"    Loaded {len(targets)} targets from config: {config_path.name}")
        return targets
    except Exception as e:
        print(f"    Warning: Could not load targets config: {e}")
        return None


def reprocess_pointing_only(output_path: Path, override_pointing_arm: str,
                            ground_plane_rotation=None):
    """
    Reprocess pointing analysis using existing skeleton data with a different arm.

    This is much faster than re-running MediaPipe skeleton detection.
    It loads skeleton_2d.json, recomputes arm vectors with the new arm,
    and regenerates the pointing analysis CSV and plots.

    Args:
        output_path: Path to camera output folder (contains skeleton_2d.json)
        override_pointing_arm: 'left' or 'right' - the arm to use
        ground_plane_rotation: Optional rotation matrix for ground plane correction

    Returns:
        True on success, False on failure
    """
    from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector, DetectionResult

    skeleton_file = output_path / "skeleton_2d.json"
    target_file = output_path / "target_detections_cam_frame.json"

    if not skeleton_file.exists():
        print(f"    ERROR: No skeleton_2d.json found")
        return False

    # Load skeleton data
    with open(skeleton_file) as f:
        skeleton_data = json.load(f)

    # Load targets
    targets = None
    if target_file.exists():
        with open(target_file) as f:
            targets = json.load(f)

    if not targets:
        print(f"    ERROR: No targets found")
        return False

    print(f"    Loaded {len(skeleton_data)} frames from skeleton_2d.json")

    # Create detector for computing arm vectors
    detector = MediaPipeHumanDetector()

    pointing_hand = override_pointing_arm.lower()
    print(f"    Reprocessing with arm: {pointing_hand}")

    # Reconstruct results and recompute arm vectors
    human_results = {}
    for frame_key, data in skeleton_data.items():
        # Reconstruct DetectionResult
        result = DetectionResult(
            landmarks_2d=data.get('landmarks_2d', []),
            landmarks_3d=data.get('landmarks_3d', []),
            arm_vectors=data.get('arm_vectors', {}),
            metadata=data.get('metadata', {})
        )

        # Recompute arm vectors with new arm
        if result.landmarks_3d:
            result.arm_vectors = detector._compute_arm_vectors(
                result.landmarks_3d, pointing_hand
            )
            result.metadata['pointing_arm'] = pointing_hand
            result.metadata['pointing_hand_whole_trial'] = pointing_hand

        human_results[frame_key] = result

    # Update skeleton_2d.json with new arm vectors
    output_data = {fk: r.to_dict() for fk, r in human_results.items()}
    with open(skeleton_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Update pointing_hand.json
    with open(output_path / "pointing_hand.json", 'w') as f:
        json.dump({
            "pointing_hand": pointing_hand,
            "total_frames": len(human_results),
            "frame_distribution": {pointing_hand: len(human_results)},
            "override": True
        }, f, indent=2)

    # Rerun pointing analysis
    try:
        from step2_skeleton_extraction.pointing_analysis import analyze_pointing_frame
        from step2_skeleton_extraction.csv_exporter import export_pointing_analysis_to_csv
        from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace
        from step2_skeleton_extraction.kalman_filter import smooth_pointing_analyses
        from step2_skeleton_extraction.plot_distance_to_targets import (
            plot_distance_to_targets,
            plot_distance_summary,
            plot_best_representation_analysis
        )

        analyses = {}
        for frame_key, result in human_results.items():
            if result.landmarks_3d:
                analysis = analyze_pointing_frame(
                    result, targets,
                    pointing_arm=pointing_hand,
                    ground_plane_rotation=ground_plane_rotation
                )
                if analysis:
                    analyses[frame_key] = analysis

        # Apply Kalman filtering
        print(f"    Applying Kalman filtering to {len(analyses)} frames...")
        analyses = smooth_pointing_analyses(
            analyses,
            process_noise=0.01,
            measurement_noise=0.1
        )

        # Export CSV
        csv_path = output_path / "processed_gesture.csv"
        export_pointing_analysis_to_csv(
            analyses, csv_path,
            targets=targets
        )
        print(f"    Exported: {csv_path.name}")

        # Human center for plotting
        human_center = [0, 0, 3.5]
        hc_file = output_path / "human_center.json"
        if hc_file.exists():
            with open(hc_file) as f:
                hc_data = json.load(f)
                human_center = hc_data.get('human_center', human_center)

        # Generate plots
        plot_path = output_path / "2d_pointing_trace.png"
        trial_name = f"{output_path.parent.name}_{output_path.name}"
        plot_2d_pointing_trace(analyses, targets, human_center, plot_path,
                              trial_name=trial_name, use_fixed_axes=True)

        # Distance plots
        dist_plot = output_path / "distance_to_targets_timeseries.png"
        plot_distance_to_targets(analyses, targets, dist_plot, trial_name=trial_name)

        summary_plot = output_path / "distance_to_targets_summary.png"
        plot_distance_summary(analyses, targets, summary_plot, trial_name=trial_name)

        accuracy_plot = output_path / "pointing_accuracy_comparison.png"
        plot_best_representation_analysis(analyses, targets, accuracy_plot, trial_name=trial_name)

        print(f"    Generated pointing plots")
        return True

    except Exception as e:
        print(f"    ERROR in pointing analysis: {e}")
        traceback.print_exc()
        return False


def detect_targets_for_trial(cam_path, output_path, target_frame=1, fx=615.0, fy=615.0, cx=320.0, cy=240.0,
                              expected_target_depth: float = None):
    """
    Run YOLO target detection on a single frame and save results.

    Args:
        cam_path: Path to camera folder
        output_path: Path to save output
        target_frame: Frame number to use for detection
        fx, fy, cx, cy: Camera intrinsics
        expected_target_depth: If provided, constrain targets to this depth (meters)
            Use this when you know the target distance from your experiment setup.

    Returns the list of target dicts, or None on failure.
    """
    model_path = get_default_model_path()
    if model_path is None:
        print("    ERROR: YOLO model (best.pt) not found")
        return None

    detector = TargetDetector(model_path, confidence_threshold=0.5)

    color_path = cam_path / "color" / f"frame_{target_frame:06d}.png"
    depth_path = cam_path / "depth" / f"frame_{target_frame:06d}.npy"

    if not color_path.exists():
        # Try first available frame
        color_files = sorted((cam_path / "color").glob("frame_*.png"))
        if not color_files:
            print("    ERROR: No color frames found")
            return None
        color_path = color_files[0]
        frame_num = int(color_path.stem.split('_')[-1])
        depth_path = cam_path / "depth" / f"frame_{frame_num:06d}.npy"

    color_img = cv2.imread(str(color_path))
    if color_img is None:
        print(f"    ERROR: Could not read {color_path}")
        return None

    depth_img = None
    if depth_path.exists():
        depth_img = np.load(str(depth_path))
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 1000.0

    detections = detector.detect(color_img, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)
    print(f"    Detected {len(detections)} target(s)")

    if not detections:
        return None

    # Apply known depth constraint if provided
    if expected_target_depth is not None:
        print(f"    Applying known depth constraint: {expected_target_depth:.2f}m")
        detections = apply_known_depth_constraint(
            detections, expected_target_depth,
            fx=fx, fy=fy, cx=cx, cy=cy,
            tolerance=0.5  # Allow 50cm tolerance before warning
        )

    # Enforce coplanarity (all targets at same Y height)
    detections = enforce_coplanarity(detections, use_median_y=True)

    # Sort right to left by x-coordinate (matching UI behavior)
    sorted_dets = sorted(detections, key=lambda d: d.center[0], reverse=True)

    detections_array = []
    for i, det in enumerate(sorted_dets, start=1):
        detections_array.append({
            "bbox": [int(det.x1), int(det.y1), int(det.x2), int(det.y2)],
            "center_px": [int(det.center[0]), int(det.center[1])],
            "avg_depth_m": float(det.avg_depth) if det.avg_depth is not None else 0.0,
            "x": float(det.center_3d[0]) if det.center_3d else 0.0,
            "y": float(det.center_3d[1]) if det.center_3d else 0.0,
            "z": float(det.center_3d[2]) if det.center_3d else 0.0,
            "label": f"target_{i}"
        })

    # Save target detections
    output_path.mkdir(parents=True, exist_ok=True)
    target_file = output_path / "target_detections_cam_frame.json"
    with open(target_file, 'w') as f:
        json.dump(detections_array, f, indent=2)

    # Compute and save ground plane transform
    # NOTE: For arc arrangements, use tilt-only correction to preserve X-Z shape
    if len(detections_array) >= 3:
        try:
            from step2_skeleton_extraction.ground_plane_correction import (
                compute_ground_plane_transform,
                compute_tilt_only_transform,
                fit_plane_to_points,
                get_transform_info,
                is_arc_arrangement
            )

            # Check if targets are in an arc arrangement
            if is_arc_arrangement(detections_array):
                print("    Targets detected as ARC arrangement - using tilt-only correction")
                R = compute_tilt_only_transform(detections_array)
                transform_type = 'tilt_only'
            else:
                R = compute_ground_plane_transform(detections_array)
                transform_type = 'full_ground_plane'

            if R is not None:
                target_positions = np.array([[d['x'], d['y'], d['z']] for d in detections_array])
                normal, centroid = fit_plane_to_points(target_positions)
                transform_info = get_transform_info(R, normal)

                transform_file = output_path / "ground_plane_transform.json"
                with open(transform_file, 'w') as f:
                    json.dump({
                        'rotation_matrix': R.tolist(),
                        'info': transform_info,
                        'transform_type': transform_type,
                        'description': f'{transform_type}: Rotation matrix to correct camera tilt'
                    }, f, indent=2)
                print(f"    Ground plane tilt: {transform_info['angle_deg']:.1f} deg ({transform_type})")
        except Exception as e:
            print(f"    Warning: ground plane computation failed: {e}")

    return detections_array


def process_skeletons_for_trial(cam_path, output_path, targets=None, ground_plane_rotation=None,
                                 override_pointing_arm=None):
    """
    Run MediaPipe skeleton extraction on all frames, compute pointing analysis,
    export CSV and 2D trace plot. Mirrors the UI's process_all_frames() logic.

    Args:
        cam_path: Path to camera folder with color/ and depth/ subfolders
        output_path: Path to output folder
        targets: List of target dictionaries
        ground_plane_rotation: Optional rotation matrix for ground plane correction
        override_pointing_arm: If 'left' or 'right', override auto-detected pointing arm
    """
    color_folder = cam_path / "color"
    depth_folder = cam_path / "depth"
    color_images = sorted(color_folder.glob("frame_*.png"))

    if not color_images:
        print("    ERROR: No color frames found")
        return

    # Auto-detect intrinsics
    sample = cv2.imread(str(color_images[0]))
    h, w = sample.shape[:2]
    fx, fy, cx, cy = get_camera_intrinsics(w, h)
    print(f"    Intrinsics: {w}x{h} -> fx={fx:.1f}, cx={cx:.1f}")

    has_depth = depth_folder.exists()

    detector = MediaPipeHumanDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    # Initialize Kalman filter for landmark smoothing
    from step2_skeleton_extraction.kalman_filter import LandmarkKalmanFilter
    landmark_filter = LandmarkKalmanFilter(
        num_landmarks=33,
        process_noise=0.005,
        measurement_noise=0.05
    )

    human_results = {}

    for i, color_path in enumerate(color_images, 1):
        frame_num = int(color_path.stem.split('_')[-1])
        color_img = cv2.imread(str(color_path))
        if color_img is None:
            continue
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        depth_img = None
        if has_depth:
            try:
                depth_npy = depth_folder / f"frame_{frame_num:06d}.npy"
                depth_raw = depth_folder / f"frame_{frame_num:06d}.raw"
                if depth_npy.exists():
                    depth_img = np.load(str(depth_npy))
                elif depth_raw.exists():
                    depth_data = np.fromfile(str(depth_raw), dtype=np.uint16)
                    depth_img = depth_data.reshape((h, w)).astype(np.float32) / 1000.0
                if depth_img is not None and depth_img.dtype == np.uint16:
                    depth_img = depth_img.astype(np.float32) / 1000.0
            except Exception:
                pass

        frame_key = f"frame_{frame_num:06d}"
        result = detector.detect_frame(
            color_rgb, frame_num,
            depth_image=depth_img,
            fx=fx, fy=fy, cx=cx, cy=cy
        )
        if result:
            # Apply Kalman filtering to smooth 3D landmarks
            if result.landmarks_3d:
                filtered_landmarks = landmark_filter.update(result.landmarks_3d)
                result.landmarks_3d_raw = result.landmarks_3d  # Keep raw
                result.landmarks_3d = filtered_landmarks  # Use filtered
            human_results[frame_key] = result

        if i % 50 == 0 or i == len(color_images):
            print(f"    Processed {i}/{len(color_images)} frames...", end='\r')

    print(f"    Skeleton detected in {len(human_results)}/{len(color_images)} frames")

    if not human_results:
        print("    WARNING: No skeletons detected")
        return

    # Determine pointing hand (whole-trial voting or override)
    if override_pointing_arm and override_pointing_arm.lower() in ['left', 'right']:
        pointing_hand = override_pointing_arm.lower()
        print(f"    Pointing hand: {pointing_hand} (OVERRIDE)")
    else:
        results_list = list(human_results.values())
        pointing_hand = determine_pointing_hand_whole_trial(results_list)
        print(f"    Pointing hand: {pointing_hand} (auto-detected)")

    # Update all results with whole-trial pointing hand
    for result in human_results.values():
        result.metadata['pointing_hand_whole_trial'] = pointing_hand
        if pointing_hand in ['left', 'right'] and result.landmarks_3d:
            result.arm_vectors = detector._compute_arm_vectors(
                result.landmarks_3d, pointing_hand
            )
            result.metadata['pointing_arm'] = pointing_hand

    # Save skeleton_2d.json
    output_path.mkdir(parents=True, exist_ok=True)
    output_data = {fk: r.to_dict() for fk, r in human_results.items()}
    with open(output_path / "skeleton_2d.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    # Save pointing_hand.json
    pointing_arms = {}
    for r in human_results.values():
        arm = r.metadata.get('pointing_arm', 'unknown')
        pointing_arms[arm] = pointing_arms.get(arm, 0) + 1

    with open(output_path / "pointing_hand.json", 'w') as f:
        json.dump({
            "pointing_hand": pointing_hand,
            "total_frames": len(human_results),
            "frame_distribution": pointing_arms
        }, f, indent=2)

    # Pointing analysis + CSV + plot (only if targets available)
    if targets:
        try:
            from step2_skeleton_extraction.pointing_analysis import analyze_pointing_frame
            from step2_skeleton_extraction.csv_exporter import export_pointing_analysis_to_csv
            from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace
            from step2_skeleton_extraction.kalman_filter import smooth_pointing_analyses
            from step2_skeleton_extraction.plot_distance_to_targets import (
                plot_distance_to_targets,
                plot_distance_summary,
                plot_best_representation_analysis
            )

            analyses = {}
            for frame_key, result in human_results.items():
                if result.landmarks_3d:
                    analysis = analyze_pointing_frame(
                        result, targets,
                        pointing_arm=result.metadata.get('pointing_arm', 'right'),
                        ground_plane_rotation=ground_plane_rotation
                    )
                    if analysis:
                        analyses[frame_key] = analysis

            # Apply Kalman filtering to smooth pointing trajectories
            print(f"    Applying Kalman filtering to {len(analyses)} frames...")
            analyses = smooth_pointing_analyses(
                analyses,
                process_noise=0.01,
                measurement_noise=0.1
            )

            # CSV export
            csv_path = output_path / "processed_gesture.csv"
            export_pointing_analysis_to_csv(
                human_results, analyses, csv_path, global_start_frame=0
            )
            print(f"    Exported CSV: {csv_path.name}")

            # Compute human center (average hip center)
            human_positions = []
            for r in human_results.values():
                if r.landmarks_3d and len(r.landmarks_3d) > 24:
                    left_hip = np.array(r.landmarks_3d[23])
                    right_hip = np.array(r.landmarks_3d[24])
                    hip_center = (left_hip + right_hip) / 2.0
                    if not np.all(hip_center == 0):
                        human_positions.append(hip_center)

            # Fallback: if hip center not found, use MediaPipe segmentation mask
            # to estimate human center from the person's centroid
            if not human_positions:
                print("    Hip center not found, using MediaPipe mask centroid fallback...")
                try:
                    import mediapipe as mp
                    mp_selfie = mp.solutions.selfie_segmentation
                    segmenter = mp_selfie.SelfieSegmentation(model_selection=0)
                    frame_keys = sorted(human_results.keys())
                    mid_key = frame_keys[len(frame_keys) // 2]
                    fnum = int(mid_key.split('_')[-1])
                    img_path = cam_path / "color" / f"frame_{fnum:06d}.png"
                    dep_path = cam_path / "depth" / f"frame_{fnum:06d}.npy"
                    if img_path.exists():
                        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                        seg_result = segmenter.process(img)
                        if seg_result.segmentation_mask is not None:
                            mask = seg_result.segmentation_mask > 0.5
                            ys, xs = np.where(mask)
                            if len(xs) > 0:
                                cx_px, cy_px = int(xs.mean()), int(ys.mean())
                                dep = None
                                if dep_path.exists():
                                    dep = np.load(str(dep_path))
                                    if dep.dtype == np.uint16:
                                        dep = dep.astype(np.float32) / 1000.0
                                if dep is not None:
                                    z = dep[cy_px, cx_px]
                                    if z > 0 and not np.isnan(z):
                                        x_3d = (cx_px - cx) * z / fx
                                        y_3d = (cy_px - cy) * z / fy
                                        human_positions.append(np.array([x_3d, y_3d, z]))
                                        print(f"    MediaPipe mask human center: [{x_3d:.3f}, {y_3d:.3f}, {z:.3f}]")
                    segmenter.close()
                except Exception as e:
                    print(f"    MediaPipe mask fallback failed: {e}")

            human_center = np.mean(human_positions, axis=0).tolist() if human_positions else [0, 0, 0]

            # Save human center for potential cross-camera fallback
            human_center_info = {
                "human_center": human_center,
                "hip_detected": len(human_positions) > 0,
                "num_hip_frames": len(human_positions),
            }
            if ground_plane_rotation is not None and human_positions:
                rotated = (ground_plane_rotation @ np.array(human_center)).tolist()
                human_center_info["human_center_ground_aligned"] = rotated
            hc_path = output_path / "human_center.json"
            with open(hc_path, 'w') as f:
                json.dump(human_center_info, f, indent=2)

            # 2D pointing trace plot
            trial_name = cam_path.parent.name
            camera_name = cam_path.name
            plot_path = output_path / "2d_pointing_trace.png"
            plot_2d_pointing_trace(
                analyses, targets, human_center, plot_path,
                trial_name=f"{trial_name}_{camera_name}"
            )
            print(f"    Generated plot: {plot_path.name}")

            # Distance-to-target plots
            try:
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

                # Best representation analysis heatmap
                best_rep_path = output_path / "pointing_accuracy_comparison.png"
                plot_best_representation_analysis(
                    analyses, targets, best_rep_path,
                    trial_name=f"{trial_name}_{camera_name}"
                )

            except Exception as e:
                print(f"    Warning: distance plots failed: {e}")

        except Exception as e:
            print(f"    Warning: pointing analysis failed: {e}")
            traceback.print_exc()

    # Save sample verification visualization
    try:
        save_sample_verification_plot(
            human_results, targets, ground_plane_rotation,
            output_path, cam_path
        )
    except Exception as e:
        print(f"    Warning: verification plot failed: {e}")

    # Save summary
    with open(output_path / "detection_summary.txt", 'w') as f:
        f.write(f"Detection Summary\n{'='*40}\n")
        f.write(f"Trial: {cam_path.parent.name}\n")
        f.write(f"Camera: {cam_path.name}\n")
        f.write(f"Human frames: {len(human_results)}\n")
        f.write(f"Pointing hand: {pointing_hand}\n")
        f.write(f"\nPer-frame arm distribution:\n")
        for arm, count in sorted(pointing_arms.items()):
            pct = count / len(human_results) * 100
            f.write(f"  {arm}: {count} ({pct:.1f}%)\n")


def detect_subject_for_trial(cam_path, output_path, subject_type='dog',
                             fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    """
    Run subject detection (dog or baby) on all frames using batch video processing.
    Matches the UI's run_batch_dog_detection() logic.

    Returns the results dict, or None on failure.
    """
    from step3_subject_extraction import SubjectDetector

    color_folder = cam_path / "color"
    depth_folder = cam_path / "depth"
    color_images = sorted(color_folder.glob("frame_*.png"))

    if not color_images:
        print(f"    ERROR: No color frames for {subject_type} detection")
        return None

    # Read first frame to get dimensions
    sample = cv2.imread(str(color_images[0]))
    h, w = sample.shape[:2]

    # Load all frames
    all_frames = []
    all_frame_numbers = []
    all_depth_images = []

    for color_path in color_images:
        frame_num = int(color_path.stem.split('_')[-1])
        color_img = cv2.imread(str(color_path))
        if color_img is None:
            continue
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        depth_img = None
        if depth_folder.exists():
            try:
                depth_npy = depth_folder / f"frame_{frame_num:06d}.npy"
                depth_raw = depth_folder / f"frame_{frame_num:06d}.raw"
                if depth_npy.exists():
                    depth_img = np.load(str(depth_npy))
                elif depth_raw.exists():
                    depth_data = np.fromfile(str(depth_raw), dtype=np.uint16)
                    depth_img = depth_data.reshape((h, w)).astype(np.float32) / 1000.0
                if depth_img is not None and depth_img.dtype == np.uint16:
                    depth_img = depth_img.astype(np.float32) / 1000.0
            except Exception:
                pass

        all_frames.append(color_rgb)
        all_frame_numbers.append(frame_num)
        all_depth_images.append(depth_img)

    # Initialize detector
    detector = SubjectDetector(subject_type=subject_type, crop_ratio=0.6)

    # Run batch detection
    temp_video_path = output_path / f"{subject_type}_detection_cropped_video.mp4"
    results_dict = detector.process_batch_video(
        str(temp_video_path),
        all_frames,
        all_frame_numbers,
        depth_images=all_depth_images,
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    detected_count = len(results_dict)
    print(f"    {subject_type.capitalize()} detected in {detected_count}/{len(all_frames)} frames")

    if not results_dict:
        return None

    # Save results JSON
    json_data = {}
    for frame_key, result in results_dict.items():
        if hasattr(result, 'to_dict'):
            json_data[frame_key] = result.to_dict()
        else:
            json_data[frame_key] = {
                'subject_type': result.subject_type,
                'detection_region': result.detection_region,
                'bbox': list(result.bbox) if result.bbox else None,
                'keypoints_2d': result.keypoints_2d,
                'keypoints_3d': result.keypoints_3d if result.keypoints_3d else None
            }

    json_path = output_path / f"{subject_type}_detection_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"    Saved {subject_type} results: {json_path.name}")

    # Export CSV
    try:
        from step3_subject_extraction.dog_csv_exporter import DogCSVExporter
        csv_path = output_path / f"processed_{subject_type}_result_table.csv"

        skeleton_path = output_path / "skeleton_2d.json"
        targets_path = output_path / "target_detections_cam_frame.json"

        exporter = DogCSVExporter(fps=30.0)
        exporter.export_to_csv(
            dog_results_path=json_path,
            human_results_path=skeleton_path if skeleton_path.exists() else None,
            targets_path=targets_path if targets_path.exists() else None,
            output_csv_path=csv_path,
            start_frame_index=0
        )
        print(f"    Exported {subject_type} CSV: {csv_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} CSV export failed: {e}")

    # Generate trace visualization
    try:
        from step3_subject_extraction.dog_trace_visualizer import DogTraceVisualizer
        targets_path = output_path / "target_detections_cam_frame.json"
        trace_path = output_path / f"{subject_type}_result_trace2d.png"

        visualizer = DogTraceVisualizer()
        visualizer.create_trace_plot(
            dog_results_path=json_path,
            targets_path=targets_path if targets_path.exists() else None,
            output_image_path=trace_path,
            title=f"2D Trace (Top View) - {subject_type.capitalize()}"
        )
        print(f"    Generated {subject_type} trace plot: {trace_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} trace plot failed: {e}")

    # Generate distance plot
    try:
        from step3_subject_extraction.dog_distance_plotter import DogDistancePlotter
        csv_path = output_path / f"processed_{subject_type}_result_table.csv"
        distance_path = output_path / f"{subject_type}_distance_to_targets.png"

        plotter = DogDistancePlotter()
        plotter.create_distance_plot(
            csv_path=csv_path,
            output_image_path=distance_path,
            title=f"{subject_type.capitalize()} Distance to Targets Over Time"
        )
        print(f"    Generated distance plot: {distance_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} distance plot failed: {e}")

    return results_dict


def detect_subject_sam3(cam_path, output_path, subject_type='dog',
                        fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    """
    Run SAM3-based subject detection in a subprocess to avoid C-level
    double-free crashes from mediapipe/SAM3 CUDA conflicts.

    Returns the results dict (loaded from JSON), or None on failure.
    """
    import subprocess

    json_path = output_path / f"{subject_type}_detection_results.json"

    # Run SAM3 in isolated subprocess
    cmd = [
        sys.executable, "-m", "step3_subject_extraction.sam3_subprocess_runner",
        "--cam-path", str(cam_path),
        "--output-path", str(output_path),
        "--subject-type", subject_type,
        "--fx", str(fx), "--fy", str(fy),
        "--cx", str(cx), "--cy", str(cy),
    ]
    print(f"    Launching SAM3 in subprocess...")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)

    if result.returncode != 0:
        print(f"    WARNING: SAM3 subprocess exited with code {result.returncode} (may be CUDA cleanup crash)")
        # Don't return None — the JSON may have been saved before the crash

    if not json_path.exists():
        print(f"    ERROR: SAM3 subprocess did not produce results JSON")
        return None

    # Load results from JSON
    with open(json_path) as f:
        json_data = json.load(f)
    print(f"    Loaded SAM3 {subject_type} results: {len(json_data)} frames")

    # Export clean CSV with subject location
    _export_subject_csv(output_path, subject_type, json_path)

    # Generate trace visualization
    try:
        from step3_subject_extraction.dog_trace_visualizer import DogTraceVisualizer
        targets_path = output_path / "target_detections_cam_frame.json"
        trace_path = output_path / f"{subject_type}_result_trace2d.png"
        visualizer = DogTraceVisualizer()
        visualizer.create_trace_plot(
            dog_results_path=json_path,
            targets_path=targets_path if targets_path.exists() else None,
            output_image_path=trace_path,
            title=f"2D Trace (Top View) - {subject_type.capitalize()} (SAM3)"
        )
        print(f"    Generated {subject_type} trace plot: {trace_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} trace plot failed: {e}")

    # Generate distance plot
    try:
        from step3_subject_extraction.dog_distance_plotter import DogDistancePlotter
        csv_path = output_path / f"processed_{subject_type}_result_table.csv"
        if csv_path.exists():
            distance_path = output_path / f"{subject_type}_distance_to_targets.png"
            plotter = DogDistancePlotter()
            plotter.create_distance_plot(
                csv_path=csv_path,
                output_image_path=distance_path,
                title=f"{subject_type.capitalize()} Distance to Targets Over Time (SAM3)"
            )
            print(f"    Generated distance plot: {distance_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} distance plot failed: {e}")

    # Save combined overlay visualization (dog + skeleton + cups on frame)
    try:
        _save_subject_overlay(cam_path, output_path, subject_type, json_data)
    except Exception as e:
        print(f"    Warning: {subject_type} overlay visualization failed: {e}")

    return json_data


def _save_subject_overlay(cam_path, output_path, subject_type, subject_results):
    """
    Save a sample frame with dog/subject bbox, skeleton, and cup detections overlaid.
    Picks a frame from the middle of the trial.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Pick a middle frame that has subject detection
    sorted_keys = sorted(subject_results.keys())
    if not sorted_keys:
        return
    mid_key = sorted_keys[len(sorted_keys) // 2]
    fnum = int(mid_key.split('_')[-1])

    img_path = cam_path / "color" / f"frame_{fnum:06d}.png"
    if not img_path.exists():
        return
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img_rgb)

    # Draw subject bbox and center (supports both object and dict formats)
    subj = subject_results[mid_key]
    bbox = subj.bbox if hasattr(subj, 'bbox') else subj.get('bbox')
    kp2d = subj.keypoints_2d if hasattr(subj, 'keypoints_2d') else subj.get('keypoints_2d')
    if bbox:
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{subject_type}', color='lime', fontsize=12, fontweight='bold')
    if kp2d and len(kp2d) > 0:
        cx_s, cy_s = kp2d[0][0], kp2d[0][1]
        ax.plot(cx_s, cy_s, 'o', color='lime', markersize=10, markeredgecolor='black')

    # Draw skeleton if available
    skeleton_path = output_path / "skeleton_2d.json"
    if skeleton_path.exists():
        with open(skeleton_path) as f:
            skel_data = json.load(f)
        if mid_key in skel_data:
            frame_skel = skel_data[mid_key]
            landmarks_2d = frame_skel.get('landmarks_2d', [])
            if landmarks_2d:
                # Draw connections
                connections = [
                    (11, 12), (11, 23), (12, 24), (23, 24),
                    (11, 13), (13, 15), (12, 14), (14, 16),
                    (23, 25), (25, 27), (24, 26), (26, 28),
                    (0, 11), (0, 12),
                ]
                for s, e in connections:
                    if s < len(landmarks_2d) and e < len(landmarks_2d):
                        p1, p2 = landmarks_2d[s], landmarks_2d[e]
                        if len(p1) >= 2 and len(p2) >= 2:
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5, alpha=0.7)
                # Draw joints
                for lm in landmarks_2d:
                    if len(lm) >= 2:
                        ax.plot(lm[0], lm[1], 'o', color='dodgerblue', markersize=4, alpha=0.7)

    # Draw cup/target detections
    target_path = output_path / "target_detections_cam_frame.json"
    if target_path.exists():
        with open(target_path) as f:
            targets = json.load(f)
        for t in targets:
            if 'bbox' in t:
                bx1, by1, bx2, by2 = t['bbox']
                rect = patches.Rectangle((bx1, by1), bx2 - bx1, by2 - by1,
                                          linewidth=2, edgecolor='gold', facecolor='none')
                ax.add_patch(rect)
            if 'center_px' in t:
                cpx, cpy = t['center_px']
                ax.plot(cpx, cpy, '*', color='gold', markersize=15, markeredgecolor='black')
                label = t.get('label', '')
                ax.text(cpx + 5, cpy - 5, label, color='gold', fontsize=9, fontweight='bold')

    trial_name = cam_path.parent.name
    camera_name = cam_path.name
    ax.set_title(f'{trial_name}/{camera_name} - {mid_key}\n'
                 f'Subject: {subject_type} | Skeleton + Cups overlay')
    ax.axis('off')
    plt.tight_layout()

    out_file = output_path / f"sample_{subject_type}_overlay.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved {subject_type} overlay: {out_file.name}")


def _export_subject_csv(output_path, subject_type, json_path):
    """Export a clean CSV with subject location data."""
    try:
        from step3_subject_extraction.dog_csv_exporter import DogCSVExporter
        csv_path = output_path / f"processed_{subject_type}_result_table.csv"
        skeleton_path = output_path / "skeleton_2d.json"
        targets_path = output_path / "target_detections_cam_frame.json"

        exporter = DogCSVExporter(fps=30.0)
        exporter.export_to_csv(
            dog_results_path=json_path,
            human_results_path=skeleton_path if skeleton_path.exists() else None,
            targets_path=targets_path if targets_path.exists() else None,
            output_csv_path=csv_path,
            start_frame_index=0
        )
        print(f"    Exported {subject_type} CSV: {csv_path.name}")
    except Exception as e:
        print(f"    Warning: {subject_type} CSV export failed: {e}")


def save_sample_verification_plot(human_results, targets, ground_plane_rotation,
                                  output_path, cam_path):
    """
    Save a sample 3D visualization showing skeleton, targets, pointing rays,
    and ground plane in the rotated frame — for verifying the transform is correct.

    Picks a frame from the middle of the trial where a skeleton was detected.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not human_results or not targets:
        return

    R = ground_plane_rotation

    # Pick a frame from the middle of the trial
    frame_keys = sorted(human_results.keys())
    mid_key = frame_keys[len(frame_keys) // 2]
    result = human_results[mid_key]

    if not result.landmarks_3d:
        return

    # Transform data into ground-aligned frame
    landmarks = np.array(result.landmarks_3d)
    if R is not None:
        landmarks = (R @ landmarks.T).T

    # Transform targets
    target_pts = []
    target_labels = []
    for t in targets:
        pos = np.array([t['x'], t['y'], t['z']])
        if R is not None:
            pos = R @ pos
        target_pts.append(pos)
        target_labels.append(t.get('label', ''))
    target_pts = np.array(target_pts)
    ground_y = float(target_pts[:, 1].mean()) if len(target_pts) > 0 else 0.0

    # Transform arm vectors
    arm_vectors = result.arm_vectors or {}
    wrist_idx = 15 if result.metadata.get('pointing_arm') == 'left' else 16
    wrist = landmarks[wrist_idx]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # -- Draw ground plane as a semi-transparent surface --
    x_range = np.linspace(target_pts[:, 0].min() - 0.5, target_pts[:, 0].max() + 0.5, 10)
    z_range = np.linspace(target_pts[:, 2].min() - 1.0, target_pts[:, 2].max() + 0.5, 10)
    X, Z = np.meshgrid(x_range, z_range)
    Y = np.full_like(X, ground_y)
    ax.plot_surface(X, Y, Z, alpha=0.15, color='green', label='Ground plane')

    # -- Draw skeleton connections --
    connections = [
        (11, 12), (11, 23), (12, 24), (23, 24),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (23, 25), (25, 27), (24, 26), (26, 28),
        (0, 11), (0, 12),
    ]
    for s, e in connections:
        if s < len(landmarks) and e < len(landmarks):
            pts = landmarks[[s, e]]
            if not (np.all(pts[0] == 0) or np.all(pts[1] == 0)):
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=2)

    # Skeleton joints
    valid = landmarks[~np.all(landmarks == 0, axis=1)]
    ax.scatter(valid[:, 0], valid[:, 1], valid[:, 2],
               c='dodgerblue', s=30, alpha=0.7, label='Skeleton')

    # -- Draw targets --
    ax.scatter(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2],
               c='gold', marker='*', s=400, edgecolors='black', linewidths=2,
               label='Targets', zorder=10)
    for pos, label in zip(target_pts, target_labels):
        ax.text(pos[0], pos[1] + 0.05, pos[2], f'  {label}',
                fontsize=9, fontweight='bold', color='darkgoldenrod')

    # -- Draw pointing rays to ground --
    vec_configs = [
        ('eye_to_wrist', 'purple', 'Eye→Wrist'),
        ('shoulder_to_wrist', 'green', 'Shoulder→Wrist'),
        ('elbow_to_wrist', 'orange', 'Elbow→Wrist'),
        ('nose_to_wrist', 'cyan', 'Nose→Wrist'),
    ]
    for vec_name, color, label in vec_configs:
        vec = arm_vectors.get(vec_name)
        if vec is None:
            continue
        vec = np.array(vec)
        if R is not None:
            vec = R @ vec
        # Extend ray from wrist to ground
        if vec[1] != 0:
            t_val = (ground_y - wrist[1]) / vec[1]
            intersection = wrist + vec * t_val
            ax.plot([wrist[0], intersection[0]],
                    [wrist[1], intersection[1]],
                    [wrist[2], intersection[2]],
                    color=color, linewidth=2, linestyle='--', alpha=0.8)
            ax.scatter([intersection[0]], [intersection[1]], [intersection[2]],
                       c=color, marker='x', s=100, linewidths=2, label=label)

    # -- Wrist marker --
    ax.scatter([wrist[0]], [wrist[1]], [wrist[2]],
               c='red', marker='o', s=150, edgecolors='black', linewidths=2,
               label='Wrist', zorder=10)

    # Labels and view
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m) — ground at {:.2f}'.format(ground_y))
    ax.set_zlabel('Z (m) — depth')
    trial_name = cam_path.parent.name
    camera_name = cam_path.name
    ax.set_title(f'Verification: {trial_name}/{camera_name} — {mid_key}\n'
                 f'Ground plane at y={ground_y:.3f}m '
                 f'(tilt corrected: {"yes" if R is not None else "no"})')
    ax.view_init(elev=-55, azim=-90)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()

    out_file = output_path / "sample_verification_3d.png"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved verification plot: {out_file.name}")


def sync_results_to_source(output_path, cam_path):
    """Copy results from output folder back to the original data folder."""
    items = [
        "processed_gesture.csv",
        "2d_pointing_trace.png",
        "skeleton_2d.json",
        "pointing_hand.json",
        "target_detections_cam_frame.json",
        "ground_plane_transform.json",
        "detection_summary.txt",
        "sample_verification_3d.png",
        "dog_detection_results.json",
        "baby_detection_results.json",
        "processed_dog_result_table.csv",
        "processed_baby_result_table.csv",
        "dog_result_trace2d.png",
        "baby_result_trace2d.png",
        "dog_distance_to_targets.png",
        "baby_distance_to_targets.png",
        "processed_dog_location.csv",
        "processed_baby_location.csv",
        "sample_dog_overlay.png",
        "sample_baby_overlay.png",
        # New distance-to-target plots
        "distance_to_targets_timeseries.png",
        "distance_to_targets_summary.png",
        "pointing_accuracy_comparison.png",
    ]
    copied = 0
    for item_name in items:
        src = output_path / item_name
        dst = cam_path / item_name
        if src.exists():
            shutil.copy2(str(src), str(dst))
            copied += 1
    if copied:
        print(f"    Synced {copied} result files back to source")


def process_single_camera(cam_path, study_path, trial_name, camera_name,
                          skip_targets=False, target_frame=1,
                          subject_type='dog', use_sam3=False,
                          expected_target_depth: float = None,
                          skip_ground_rotation: bool = False,
                          pointing_only: bool = False,
                          override_pointing_arm: str = None):
    """Process a single trial/camera end-to-end.

    Args:
        expected_target_depth: If provided, constrain detected targets to this depth (meters).
            Use this when you know the target distance from your experiment setup.
        skip_ground_rotation: If True, skip ground plane rotation entirely (use raw camera coords).
            Useful when targets are on a curved arc and rotation would distort the shape.
        pointing_only: If True, only reprocess pointing analysis (skip skeleton detection and subject).
            Requires existing skeleton_2d.json.
        override_pointing_arm: If 'left' or 'right', override auto-detected pointing arm.
    """
    study_name = study_path.name
    print(f"\n{'='*60}")
    print(f"  {study_name} / {trial_name} / {camera_name}")
    print(f"{'='*60}")

    color_folder = cam_path / "color"
    if not color_folder.exists():
        print(f"    SKIP: No color folder")
        return

    num_frames = len(list(color_folder.glob("frame_*.png")))
    print(f"    Frames: {num_frames}")

    # Auto-detect intrinsics from first frame
    first_frame = sorted(color_folder.glob("frame_*.png"))[0]
    sample = cv2.imread(str(first_frame))
    h, w = sample.shape[:2]
    fx, fy, cx, cy = get_camera_intrinsics(w, h)

    # Output path: <study_parent>/<study_name>_output/<trial>/<camera>/
    output_path = study_path.parent / f"{study_name}_output" / trial_name / camera_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Target detection
    targets = None
    target_file = output_path / "target_detections_cam_frame.json"

    # First, try to load from config/targets.yaml (known ground truth positions)
    config_targets = load_targets_from_config()

    if skip_targets and target_file.exists():
        print("    [Targets] Loading existing detections...")
        with open(target_file) as f:
            targets = json.load(f)
        print(f"    [Targets] {len(targets)} targets loaded")
    elif config_targets is not None:
        # Use targets from config file
        print(f"    [Targets] Using {len(config_targets)} targets from config/targets.yaml")
        targets = config_targets
        # Save to output for consistency
        output_path.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'w') as f:
            json.dump(targets, f, indent=2)
    else:
        # Fall back to YOLO detection
        print(f"    [Targets] Detecting on frame {target_frame}...")
        # Use default target depth if not specified
        depth_to_use = expected_target_depth if expected_target_depth is not None else DEFAULT_TARGET_DEPTH
        targets = detect_targets_for_trial(cam_path, output_path,
                                           target_frame=target_frame,
                                           fx=fx, fy=fy, cx=cx, cy=cy,
                                           expected_target_depth=depth_to_use)

    # Load ground plane rotation if available (unless skipped)
    ground_plane_rotation = None
    if skip_ground_rotation:
        print(f"    [Ground plane] Skipped (using raw camera coordinates)")
    else:
        transform_file = output_path / "ground_plane_transform.json"
        if transform_file.exists():
            with open(transform_file) as f:
                transform_data = json.load(f)
                ground_plane_rotation = np.array(transform_data['rotation_matrix'])
            transform_type = transform_data.get('transform_type', 'unknown')
            print(f"    [Ground plane] Loaded {transform_type} rotation matrix")

    # POINTING-ONLY MODE: Skip skeleton detection, just reprocess pointing analysis
    if pointing_only:
        if not override_pointing_arm:
            print(f"    ERROR: --pointing-only requires --arm-csv or explicit arm override")
            return
        print(f"    [Pointing-only] Reprocessing with arm: {override_pointing_arm}")
        success = reprocess_pointing_only(output_path, override_pointing_arm, ground_plane_rotation)
        if success:
            sync_results_to_source(output_path, cam_path)
        print(f"    DONE")
        return

    # Step 2: Skeleton extraction + analysis
    print("    [Skeleton] Processing all frames...")
    process_skeletons_for_trial(cam_path, output_path, targets=targets,
                                ground_plane_rotation=ground_plane_rotation,
                                override_pointing_arm=override_pointing_arm)

    # Step 3: Subject detection (dog/baby)
    if subject_type:
        print(f"    [{subject_type.capitalize()}] Running detection ({'SAM3' if use_sam3 else 'DLC/MediaPipe'})...")
        try:
            if use_sam3:
                detect_subject_sam3(
                    cam_path, output_path, subject_type=subject_type,
                    fx=fx, fy=fy, cx=cx, cy=cy
                )
            else:
                detect_subject_for_trial(
                    cam_path, output_path, subject_type=subject_type,
                    fx=fx, fy=fy, cx=cx, cy=cy
                )
        except Exception as e:
            print(f"    Warning: {subject_type} detection failed: {e}")
            traceback.print_exc()

    # Step 4: Sync results back to source
    sync_results_to_source(output_path, cam_path)

    print(f"    DONE")


def process_study(study_path, trial_filter=None, camera_filter=None,
                  skip_targets=False, target_frame=1, subject_type='dog',
                  use_sam3=False, expected_target_depth: float = None,
                  skip_ground_rotation: bool = False,
                  pointing_only: bool = False,
                  arm_csv_path: str = None):
    """Process all trials in a study folder.

    Args:
        expected_target_depth: Known target depth in meters (e.g., 4.0).
            Use when you know the distance from your experiment setup.
        skip_ground_rotation: If True, skip ground plane rotation entirely.
            Useful when targets are on a curved arc and rotation would distort the shape.
        pointing_only: If True, only reprocess pointing analysis (skip skeleton + subject detection).
        arm_csv_path: Path to CSV with arm overrides. Columns: study, trial, camera, override_arm, reprocess.
    """
    study_path = Path(study_path)
    study_name = study_path.name

    print(f"\n{'#'*60}")
    print(f"  STUDY: {study_name}")
    print(f"  Path:  {study_path}")
    if pointing_only:
        print(f"  MODE: Pointing-only reprocessing")
    if arm_csv_path:
        print(f"  ARM CSV: {arm_csv_path}")
    print(f"{'#'*60}")

    # Load arm overrides if CSV provided
    arm_overrides = {}
    if arm_csv_path:
        try:
            arm_overrides = load_arm_overrides(Path(arm_csv_path))
            print(f"  Loaded {len(arm_overrides)} arm override entries from CSV")
        except Exception as e:
            print(f"  ERROR loading arm CSV: {e}")
            return

    # Discover trials
    trials = sorted([
        d for d in study_path.iterdir()
        if d.is_dir() and d.name.startswith("trial_")
    ], key=lambda p: int(p.name.split('_')[-1]) if p.name.split('_')[-1].isdigit() else 0)

    if trial_filter:
        trials = [t for t in trials if t.name == trial_filter]

    print(f"  Found {len(trials)} trial(s)")

    for trial_path in trials:
        trial_name = trial_path.name

        # Discover cameras
        cameras = sorted([
            d for d in trial_path.iterdir()
            if d.is_dir() and d.name.startswith("cam")
        ])

        if camera_filter:
            cameras = [c for c in cameras if c.name in camera_filter]

        for cam_path in cameras:
            camera_name = cam_path.name

            # Check arm overrides for this trial/camera
            override_info = get_arm_override(arm_overrides, study_name, trial_name, camera_name)
            override_arm = None
            should_reprocess = True  # Default: process all

            if override_info:
                override_arm = override_info.get('override_arm')
                should_reprocess = override_info.get('reprocess', True)

                # If CSV says skip, skip this trial
                if override_arm == 'skip':
                    print(f"\n  SKIP: {trial_name}/{camera_name} (marked skip in CSV)")
                    continue

                # If using CSV and reprocess=False, skip unless pointing_only
                if arm_csv_path and not should_reprocess and not trial_filter:
                    print(f"\n  SKIP: {trial_name}/{camera_name} (reprocess=no in CSV)")
                    continue

            try:
                process_single_camera(
                    cam_path, study_path, trial_name, camera_name,
                    skip_targets=skip_targets, target_frame=target_frame,
                    subject_type=subject_type if not pointing_only else None,
                    use_sam3=use_sam3,
                    expected_target_depth=expected_target_depth,
                    skip_ground_rotation=skip_ground_rotation,
                    pointing_only=pointing_only,
                    override_pointing_arm=override_arm
                )
            except Exception as e:
                print(f"\n    ERROR: {e}")
                traceback.print_exc()
                continue

        # Cross-camera hip fallback: if any camera failed hip detection,
        # log which cameras succeeded for manual review.
        # NOTE: Automatic cross-camera transfer requires camera extrinsics
        # (not just ground plane rotation) to properly align coordinate frames.
        trial_output = study_path.parent / f"{study_name}_output" / trial_name
        try:
            cam_hip_status = {}
            for cam_path in cameras:
                hc_file = trial_output / cam_path.name / "human_center.json"
                if hc_file.exists():
                    with open(hc_file) as f:
                        cam_hip_status[cam_path.name] = json.load(f)
            failed = [c for c, d in cam_hip_status.items() if not d.get('hip_detected')]
            succeeded = [c for c, d in cam_hip_status.items() if d.get('hip_detected')]
            if failed and succeeded:
                print(f"\n    [Hip Fallback] Cameras without hip: {failed}")
                print(f"    [Hip Fallback] Cameras with hip: {succeeded}")
                print(f"    [Hip Fallback] Cross-camera transfer requires extrinsics — skipped.")
        except Exception as e:
            print(f"    [Hip Fallback] Check failed: {e}")

        # After all cameras for this trial, generate trial-level visualizations from JSON
        try:
            generate_trial_visualizations_from_json(trial_output, trial_name, subject_type)
        except Exception as e:
            print(f"    Warning: trial-level visualization failed: {e}")
            traceback.print_exc()

    print(f"\n{'#'*60}")
    print(f"  STUDY COMPLETE: {study_name}")
    print(f"{'#'*60}")


def generate_trial_visualizations_from_json(trial_output, trial_name, subject_type=None):
    """
    Generate trial-level visualizations by reading saved JSON files.
    This proves the saved data is accurate — all plots come from JSON, not live detection.

    Saves outputs under <study>_output/<trial>/ (shared across cameras).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print(f"\n  [Trial Summary] Generating visualizations from JSON for {trial_name}...")

    # Find all camera output dirs
    cam_dirs = sorted([d for d in trial_output.iterdir() if d.is_dir() and d.name.startswith("cam")])
    if not cam_dirs:
        return

    for cam_dir in cam_dirs:
        cam_name = cam_dir.name

        # Load ground plane rotation for this camera (if available)
        R = None
        transform_file = cam_dir / "ground_plane_transform.json"
        if transform_file.exists():
            with open(transform_file) as f:
                R = np.array(json.load(f)['rotation_matrix'])

        # --- 1. Dog/subject trace from JSON ---
        # Check for any subject detection JSON (dog or baby), regardless of current --subject flag
        targets_json = cam_dir / "target_detections_cam_frame.json"
        gesture_csv = cam_dir / "processed_gesture.csv"

        subj_json = None
        detected_subj_type = subject_type or 'dog'
        for st in ([subject_type] if subject_type else []) + ['dog', 'baby']:
            candidate = cam_dir / f"{st}_detection_results.json"
            if candidate.exists():
                subj_json = candidate
                detected_subj_type = st
                break

        if subj_json and subj_json.exists() and targets_json.exists():
            try:
                with open(subj_json) as f:
                    subj_data = json.load(f)
                with open(targets_json) as f:
                    targets = json.load(f)

                # Extract subject center positions from JSON
                subj_positions = []
                for fk in sorted(subj_data.keys()):
                    fd = subj_data[fk]
                    kp3d = fd.get('keypoints_3d')
                    if kp3d and len(kp3d) > 0 and kp3d[0] and all(v != 0 for v in kp3d[0]):
                        pos = np.array(kp3d[0])
                        if R is not None:
                            pos = R @ pos
                        subj_positions.append((fk, pos.tolist()))

                # Apply ground plane rotation to targets
                plot_targets = targets
                if R is not None:
                    plot_targets = []
                    for t in targets:
                        tc = dict(t)
                        if 'x' in t and 'y' in t and 'z' in t:
                            rp = R @ np.array([t['x'], t['y'], t['z']])
                            tc['x'], tc['y'], tc['z'] = rp.tolist()
                        plot_targets.append(tc)

                if subj_positions and plot_targets:
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Plot subject trace (X vs Z, top-down view in ground-aligned frame)
                    xs = [p[1][0] for p in subj_positions]
                    zs = [p[1][2] for p in subj_positions]
                    ax.plot(xs, zs, '-', color='lime', linewidth=1.5, alpha=0.7)
                    ax.scatter(xs, zs, c=range(len(xs)), cmap='Greens', s=20, zorder=5)
                    ax.scatter([xs[0]], [zs[0]], c='green', s=100, marker='s', label=f'{detected_subj_type} start', zorder=10)
                    ax.scatter([xs[-1]], [zs[-1]], c='red', s=100, marker='o', label=f'{detected_subj_type} end', zorder=10)

                    # Plot targets
                    for t in plot_targets:
                        tx, tz = t['x'], t['z']
                        ax.scatter(tx, tz, c='gold', s=200, marker='*', edgecolors='black', linewidths=1.5, zorder=10)
                        ax.text(tx + 0.02, tz + 0.02, t.get('label', ''), fontsize=9, fontweight='bold', color='darkgoldenrod')

                    frame_label = 'ground-aligned' if R is not None else 'camera frame'
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Z (m) — depth')
                    ax.set_title(f'{trial_name}/{cam_name} — {detected_subj_type.capitalize()} Trace ({frame_label}, from JSON)\n'
                                 f'{len(subj_positions)} frames with detection')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                    # Fixed Z range (depth) for consistent dog trace viewing
                    x_min, x_max = np.percentile(xs, 5), np.percentile(xs, 95)
                    for t in plot_targets:
                        x_min, x_max = min(x_min, t['x']), max(x_max, t['x'])
                    x_pad = max(0.3, (x_max - x_min) * 0.15)
                    ax.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax.set_ylim(1.0, 5.0)

                    plt.tight_layout()

                    out = trial_output / f"{cam_name}_{detected_subj_type}_trace_from_json.png"
                    out_cam = cam_dir / f"{detected_subj_type}_trace_from_json.png"
                    plt.savefig(out, dpi=150, bbox_inches='tight')
                    plt.savefig(out_cam, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    [{cam_name}] Saved {detected_subj_type} trace (from JSON): {out.name}")

            except Exception as e:
                print(f"    [{cam_name}] Warning: subject trace from JSON failed: {e}")

        # --- 2. Pointing vector distances from CSV (already saved from JSON) ---
        if gesture_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(gesture_csv)

                # Plot distance-to-targets over time for each vector type
                vec_types = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist', 'nose_to_wrist', 'head_orientation']
                colors = ['purple', 'green', 'orange', 'cyan', 'magenta']

                # Find target distance columns
                target_cols = {}
                for vt in vec_types:
                    for i in range(1, 5):
                        col = f'{vt}_dist_to_target_{i}'
                        if col in df.columns:
                            if vt not in target_cols:
                                target_cols[vt] = []
                            target_cols[vt].append((i, col))

                if target_cols:
                    n_targets = max(len(v) for v in target_cols.values())
                    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 3 * n_targets), sharex=True)
                    if n_targets == 1:
                        axes = [axes]

                    x_axis = df['frame'] if 'frame' in df.columns else (df['frame_number'] if 'frame_number' in df.columns else range(len(df)))

                    for target_i in range(n_targets):
                        ax = axes[target_i]
                        for vt, color in zip(vec_types, colors):
                            col = f'{vt}_dist_to_target_{target_i + 1}'
                            if col in df.columns:
                                valid = df[col].dropna()
                                if len(valid) > 0:
                                    ax.plot(x_axis[:len(valid)], valid.values, '-', color=color, label=vt, alpha=0.7)
                        ax.set_ylabel(f'Dist to target {target_i + 1} (m)')
                        ax.legend(fontsize=7, loc='upper right')
                        ax.grid(True, alpha=0.3)

                    axes[-1].set_xlabel('Frame')
                    fig.suptitle(f'{trial_name}/{cam_name} — Pointing Vector Distance to Targets (from CSV)', fontsize=12)
                    plt.tight_layout()

                    out = trial_output / f"{cam_name}_pointing_distance_to_targets_from_json.png"
                    out_cam = cam_dir / "pointing_distance_to_targets_from_json.png"
                    plt.savefig(out, dpi=150, bbox_inches='tight')
                    plt.savefig(out_cam, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"    [{cam_name}] Saved pointing distance plot (from CSV): {out.name}")

            except Exception as e:
                print(f"    [{cam_name}] Warning: pointing distance plot failed: {e}")

    # --- 3. Save a trial-level summary under the trial root ---
    summary = {
        "trial": trial_name,
        "cameras": [d.name for d in cam_dirs],
        "subject_type": subject_type,
    }
    for cam_dir in cam_dirs:
        cam_summary = {}
        gesture_csv = cam_dir / "processed_gesture.csv"
        if gesture_csv.exists():
            cam_summary["gesture_csv"] = str(gesture_csv)
        subj_json = cam_dir / f"{subject_type}_detection_results.json" if subject_type else None
        if subj_json and subj_json.exists():
            cam_summary["subject_json"] = str(subj_json)
        ph = cam_dir / "pointing_hand.json"
        if ph.exists():
            with open(ph) as f:
                cam_summary["pointing_hand"] = json.load(f)
        summary[cam_dir.name] = cam_summary

    summary_path = trial_output / "trial_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [Trial Summary] Saved: {summary_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process pointing gesture data for a study folder."
    )
    parser.add_argument(
        "study_path",
        type=str,
        help="Path to study folder (e.g., /home/tigerli/Documents/pointing_data/BDL396_...)"
    )
    parser.add_argument(
        "--trial", type=str, default=None,
        help="Process only this trial (e.g., trial_6)"
    )
    parser.add_argument(
        "--cameras", type=str, nargs='+', default=None,
        help="Process only these cameras (e.g., cam1 cam2)"
    )
    parser.add_argument(
        "--skip-targets", action="store_true",
        help="Skip target detection if results already exist"
    )
    parser.add_argument(
        "--target-frame", type=int, default=1,
        help="Frame number to use for target detection (default: 1)"
    )
    parser.add_argument(
        "--subject", type=str, default='dog', choices=['dog', 'baby', 'none'],
        help="Subject type to detect (default: dog). Use 'none' to skip subject detection."
    )
    parser.add_argument(
        "--use-sam3", action="store_true",
        help="Use SAM3 for subject detection instead of DLC/MediaPipe"
    )
    parser.add_argument(
        "--target-depth", type=float, default=None,
        help=f"Known target depth in meters. Default from config: {DEFAULT_TARGET_DEPTH}m. "
             "Constrains detected targets to this depth for more accurate pointing analysis. "
             "If config/targets.yaml exists, targets are loaded from there instead."
    )
    parser.add_argument(
        "--skip-ground-rotation", action="store_true",
        help="Skip ground plane rotation entirely (use raw camera coordinates). "
             "Useful when targets are on a curved arc and rotation would distort the shape."
    )

    # Pointing reprocessing options
    parser.add_argument(
        "--pointing-only", action="store_true",
        help="Only reprocess pointing analysis (skip skeleton detection and subject detection). "
             "Requires existing skeleton_2d.json. Use with --arm-csv to override pointing arm."
    )
    parser.add_argument(
        "--arm-csv", type=str, default=None,
        help="Path to CSV file with arm overrides. Columns: study, trial, camera, detected_arm, "
             "override_arm (left/right/skip), reprocess (yes/no). "
             "Use --generate-arm-csv to create a template."
    )
    parser.add_argument(
        "--generate-arm-csv", action="store_true",
        help="Generate a CSV template for arm overrides from existing data, then exit. "
             "Edit the CSV to set override_arm and reprocess columns, then run with --arm-csv."
    )

    args = parser.parse_args()

    # Handle --generate-arm-csv first
    if args.generate_arm_csv:
        generate_arm_csv_template(Path(args.study_path))
        return

    subject_type = args.subject if args.subject != 'none' else None

    process_study(
        args.study_path,
        trial_filter=args.trial,
        camera_filter=args.cameras,
        skip_targets=args.skip_targets,
        target_frame=args.target_frame,
        subject_type=subject_type,
        use_sam3=args.use_sam3,
        expected_target_depth=args.target_depth,
        skip_ground_rotation=args.skip_ground_rotation,
        pointing_only=args.pointing_only,
        arm_csv_path=args.arm_csv,
    )


if __name__ == "__main__":
    main()

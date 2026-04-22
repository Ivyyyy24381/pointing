#!/usr/bin/env python3
"""
Post-Processing Pipeline for Pointing Gesture Analysis.

Works entirely from saved output data (no raw frames needed).
Use this to fix targets, override pointing arms, and reprocess pointing analysis.

Subcommands:
    scan            Show status of all trials (targets, arm, data quality)
    fix-targets     Find reference targets and apply to trials with bad/missing targets
    reprocess       Reprocess pointing analysis from saved skeleton data
    generate-csv    Generate arm override CSV template

Examples:
    # 1. See what you have
    python batch_postprocess.py scan /path/to/study_output

    # 2. Fix targets (find reference, apply to bad trials)
    python batch_postprocess.py fix-targets /path/to/study_output

    # 3. Reprocess pointing (after fixing targets or changing arm)
    python batch_postprocess.py reprocess /path/to/study_output

    # 4. Reprocess with arm override CSV
    python batch_postprocess.py reprocess /path/to/study_output --arm-csv arms.csv

    # 5. Full pipeline: fix targets + reprocess all
    python batch_postprocess.py fix-targets /path/to/study_output --reprocess

    # 6. Generate arm override CSV
    python batch_postprocess.py generate-csv /path/to/study_output
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# SCAN: Show status of all trials
# =============================================================================

def scan_study_output(output_path: Path):
    """
    Scan all trial/camera outputs and show status.

    Prints a table showing:
    - How many targets each trial detected
    - Whether target positions look reasonable
    - Which arm was detected
    - Whether skeleton data exists
    """
    output_path = Path(output_path)

    print(f"\n{'='*80}")
    print(f"  SCAN: {output_path.name}")
    print(f"{'='*80}")
    print(f"  {'Trial':<12} {'Camera':<8} {'Targets':<10} {'Depth Range':<16} {'Arm':<8} {'Skeleton':<10} {'CSV':<6}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*16} {'-'*8} {'-'*10} {'-'*6}")

    cameras_with_4 = {}  # {camera_name: [(trial, targets), ...]}
    all_entries = []

    for trial_dir in sorted(output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue

            trial_name = trial_dir.name
            camera_name = cam_dir.name

            # Target info
            target_file = cam_dir / "target_detections_cam_frame.json"
            num_targets = 0
            depth_range = "-"
            targets = []
            if target_file.exists():
                with open(target_file) as f:
                    targets = json.load(f)
                num_targets = len(targets)
                if targets:
                    depths = [t.get('z', t.get('avg_depth_m', 0)) for t in targets]
                    depth_range = f"{min(depths):.1f}-{max(depths):.1f}m"

            # Arm info
            arm = "-"
            pointing_file = cam_dir / "pointing_hand.json"
            if pointing_file.exists():
                with open(pointing_file) as f:
                    data = json.load(f)
                    arm = data.get('pointing_hand', '-')
                    if data.get('override'):
                        arm += '*'

            # Skeleton info
            skeleton_file = cam_dir / "skeleton_2d.json"
            skeleton_status = "Yes" if skeleton_file.exists() else "No"

            # CSV info
            csv_file = cam_dir / "processed_gesture.csv"
            csv_status = "Yes" if csv_file.exists() else "No"

            # Flag issues
            flag = ""
            if num_targets < 4:
                flag = " <-- MISSING TARGETS"
            elif num_targets > 4:
                flag = " <-- TOO MANY TARGETS"

            print(f"  {trial_name:<12} {camera_name:<8} {num_targets:<10} {depth_range:<16} {arm:<8} {skeleton_status:<10} {csv_status:<6}{flag}")

            entry = {
                'trial': trial_name, 'camera': camera_name,
                'num_targets': num_targets, 'targets': targets,
                'arm': arm, 'has_skeleton': skeleton_file.exists(),
                'has_csv': csv_file.exists(), 'path': cam_dir
            }
            all_entries.append(entry)

            # Track cameras with 4 targets
            if num_targets == 4:
                cameras_with_4.setdefault(camera_name, []).append((trial_name, targets))

    # Summary
    total = len(all_entries)
    missing_targets = sum(1 for e in all_entries if e['num_targets'] < 4)
    no_skeleton = sum(1 for e in all_entries if not e['has_skeleton'])
    no_csv = sum(1 for e in all_entries if not e['has_csv'])

    print(f"\n  Summary:")
    print(f"    Total trial/camera combos: {total}")
    print(f"    Missing targets (<4):      {missing_targets}")
    print(f"    No skeleton data:          {no_skeleton}")
    print(f"    No CSV output:             {no_csv}")

    for cam_name, trials_with_4 in sorted(cameras_with_4.items()):
        print(f"    {cam_name}: {len(trials_with_4)} trials with 4 targets (usable as reference)")

    return all_entries


# =============================================================================
# FIX TARGETS: Find reference and apply to bad trials
# =============================================================================

def find_reference_targets(output_path: Path, max_depth_diff: float = 1.0) -> dict:
    """
    Find reference targets for each camera by analyzing the distribution of
    target positions across ALL trials with 4 targets, then taking the median.

    This is more robust than using a single trial, as it averages out
    depth noise from individual frames.

    Args:
        output_path: Path to study output folder
        max_depth_diff: Max allowed depth difference between targets within a single
            trial (meters). Trials exceeding this are excluded as bad detections.

    Returns:
        Dict mapping camera_name -> list of 4 target dicts (median positions)
    """
    output_path = Path(output_path)

    # Collect all valid 4-target detections per camera
    # camera_name -> [trial_targets_list, ...]
    all_detections = {}

    for trial_dir in sorted(output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue

            camera_name = cam_dir.name
            target_file = cam_dir / "target_detections_cam_frame.json"
            if not target_file.exists():
                continue

            with open(target_file) as f:
                targets = json.load(f)

            if len(targets) != 4:
                continue

            # Validate: consistent depth within this trial
            depths = [t.get('z', t.get('avg_depth_m', 0)) for t in targets]
            if max(depths) - min(depths) > max_depth_diff:
                continue

            # Validate: no zero/negative depths
            if any(d <= 0.5 for d in depths):
                continue

            all_detections.setdefault(camera_name, []).append(targets)

    # Compute median reference for each camera
    reference = {}

    for camera_name, detection_list in sorted(all_detections.items()):
        n = len(detection_list)
        print(f"    {camera_name}: {n} valid trials with 4 targets")

        if n == 0:
            continue

        # Stack positions: shape (n_trials, 4_targets, 3_xyz)
        positions = np.zeros((n, 4, 3))
        pixel_centers = np.zeros((n, 4, 2))
        avg_depths = np.zeros((n, 4))

        for i, targets in enumerate(detection_list):
            for j, t in enumerate(targets):
                positions[i, j] = [t.get('x', 0), t.get('y', 0), t.get('z', 0)]
                px = t.get('center_px', [0, 0])
                pixel_centers[i, j] = px[:2] if len(px) >= 2 else [0, 0]
                avg_depths[i, j] = t.get('avg_depth_m', t.get('z', 0))

        # Use median across trials for each target
        median_positions = np.median(positions, axis=0)   # (4, 3)
        median_pixels = np.median(pixel_centers, axis=0)  # (4, 2)
        median_depths = np.median(avg_depths, axis=0)     # (4,)

        # Also compute std to show stability
        std_positions = np.std(positions, axis=0)

        # Build reference target list
        ref_targets = []
        for j in range(4):
            # Use the first trial's metadata (bbox, label) as template
            template = detection_list[0][j]
            ref_target = {
                'bbox': template.get('bbox', [0, 0, 0, 0]),
                'center_px': [int(median_pixels[j, 0]), int(median_pixels[j, 1])],
                'avg_depth_m': float(median_depths[j]),
                'x': float(median_positions[j, 0]),
                'y': float(median_positions[j, 1]),
                'z': float(median_positions[j, 2]),
                'label': f'target_{j+1}'
            }
            ref_targets.append(ref_target)

            std_xyz = std_positions[j]
            print(f"      target_{j+1}: x={median_positions[j,0]:+.3f}  "
                  f"y={median_positions[j,1]:+.3f}  z={median_positions[j,2]:.3f}m  "
                  f"(std: x={std_xyz[0]:.3f} y={std_xyz[1]:.3f} z={std_xyz[2]:.3f})")

        reference[camera_name] = ref_targets

    return reference


def is_target_quality_bad(targets: list, reference_targets: list,
                          max_position_diff: float = 1.0) -> bool:
    """
    Check if targets have bad quality compared to reference.

    Returns True if targets should be replaced with reference.

    Args:
        targets: Current target list
        reference_targets: Known good reference targets
        max_position_diff: Max allowed position difference from reference (meters)
    """
    if len(targets) != len(reference_targets):
        return True

    # Compare each target position
    for t, ref in zip(targets, reference_targets):
        t_pos = np.array([t.get('x', 0), t.get('y', 0), t.get('z', 0)])
        ref_pos = np.array([ref.get('x', 0), ref.get('y', 0), ref.get('z', 0)])
        diff = np.linalg.norm(t_pos - ref_pos)
        if diff > max_position_diff:
            return True

    return False


def fix_targets(output_path: Path, max_depth_diff: float = 1.0,
                max_position_diff: float = 1.0, dry_run: bool = False,
                fallback_targets: dict = None) -> list:
    """
    Fix targets for all trials using reference targets.

    1. Finds reference targets (first trial with 4 good targets per camera)
    2. Saves reference to <output>/reference_targets/
    3. Replaces targets in trials with bad/missing detections

    Args:
        output_path: Path to study output folder
        max_depth_diff: Max depth spread for reference validation
        max_position_diff: Max position diff before replacing with reference
        dry_run: If True, only show what would be fixed
        fallback_targets: Optional cross-study fallback targets (camera -> target list).
            Used when this study has no valid within-study reference.

    Returns:
        List of (trial, camera) tuples that were fixed
    """
    output_path = Path(output_path)

    print(f"\n{'='*60}")
    print(f"  FIX TARGETS: {output_path.name}")
    print(f"{'='*60}")

    # Step 1: Find reference targets
    print(f"\n  Finding reference targets...")
    reference = find_reference_targets(output_path, max_depth_diff=max_depth_diff)

    if not reference:
        if fallback_targets:
            print(f"  WARNING: No within-study reference found — using cross-study fallback targets")
            reference = fallback_targets
        else:
            print(f"  ERROR: No valid reference targets found (need at least one trial with 4 good targets)")
            return []

    # Apply physical layout correction to ensure correct depth relationships
    # even when YOLO-detected depths are noisy
    try:
        from run_all_postprocess import _apply_physical_layout
        for cam_name in list(reference.keys()):
            reference[cam_name] = _apply_physical_layout(reference[cam_name], camera=cam_name)
            print(f"  Applied physical layout correction to {cam_name} reference targets")
    except ImportError:
        pass  # standalone usage without run_all_postprocess

    # Step 2: Save reference targets
    ref_dir = output_path / "reference_targets"
    if not dry_run:
        ref_dir.mkdir(exist_ok=True)
        for cam_name, targets in reference.items():
            ref_file = ref_dir / f"{cam_name}_targets.json"
            with open(ref_file, 'w') as f:
                json.dump(targets, f, indent=2)
        print(f"\n  Saved reference targets to: {ref_dir}")

    # Load physical layout correction if available
    try:
        from run_all_postprocess import _apply_physical_layout
        has_physical_layout = True
    except ImportError:
        has_physical_layout = False

    # Step 3: Fix bad trials (replace with reference) and correct all trials' depths
    print(f"\n  Checking all trials...")
    fixed = []

    for trial_dir in sorted(output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue

            camera_name = cam_dir.name
            trial_name = trial_dir.name

            if camera_name not in reference:
                print(f"    {trial_name}/{camera_name}: No reference for this camera - SKIP")
                continue

            target_file = cam_dir / "target_detections_cam_frame.json"

            # Check if targets need full replacement
            needs_replace = False
            reason = ""

            if not target_file.exists():
                needs_replace = True
                reason = "no target file"
            else:
                with open(target_file) as f:
                    current_targets = json.load(f)

                if len(current_targets) < 4:
                    needs_replace = True
                    reason = f"only {len(current_targets)} targets"
                elif len(current_targets) > 4:
                    needs_replace = True
                    reason = f"{len(current_targets)} targets (too many)"
                elif is_target_quality_bad(current_targets, reference[camera_name],
                                            max_position_diff=max_position_diff):
                    needs_replace = True
                    reason = "positions far from reference"

            if needs_replace:
                # Bad detection → replace entirely with reference
                if dry_run:
                    print(f"    WOULD FIX: {trial_name}/{camera_name} ({reason})")
                else:
                    if target_file.exists():
                        backup = cam_dir / "target_detections_cam_frame.json.bak"
                        shutil.copy2(str(target_file), str(backup))

                    with open(target_file, 'w') as f:
                        json.dump(reference[camera_name], f, indent=2)
                    print(f"    FIXED: {trial_name}/{camera_name} ({reason})")

                fixed.append((trial_name, camera_name))

            elif has_physical_layout and not dry_run:
                # Good detection → still apply physical layout correction for depths
                corrected = _apply_physical_layout(current_targets, camera=camera_name)
                # Check if anything actually changed
                changed = any(
                    abs(c['z'] - o['z']) > 0.001 or abs(c['x'] - o['x']) > 0.001
                    for c, o in zip(corrected, current_targets)
                )
                if changed:
                    backup = cam_dir / "target_detections_cam_frame.json.bak"
                    if not backup.exists():
                        shutil.copy2(str(target_file), str(backup))
                    with open(target_file, 'w') as f:
                        json.dump(corrected, f, indent=2)
                    print(f"    DEPTH-CORRECTED: {trial_name}/{camera_name}")
                    fixed.append((trial_name, camera_name))

    print(f"\n  {'Would fix' if dry_run else 'Fixed'}: {len(fixed)} trial/camera combos")
    return fixed


# =============================================================================
# DOG DEPTH FILTERING
# =============================================================================

def load_dog_depths(cam_dir: Path, subject_type: str = 'dog') -> dict:
    """
    Load dog/subject depth (Z) per frame from detection results JSON.

    Args:
        cam_dir: Path to camera output folder
        subject_type: 'dog' or 'baby'

    Returns:
        Dict mapping frame_key -> depth_z (meters), only for frames with valid detection
    """
    dog_file = cam_dir / f"{subject_type}_detection_results.json"
    if not dog_file.exists():
        return {}

    with open(dog_file) as f:
        dog_data = json.load(f)

    depths = {}
    for frame_key, data in dog_data.items():
        kp3d = data.get('keypoints_3d')
        if kp3d and len(kp3d) > 0:
            kp = kp3d[0]  # First keypoint (nose)
            if len(kp) >= 3 and kp[2] > 0:
                depths[frame_key] = kp[2]

    return depths


def filter_frames_by_dog_depth(frame_keys: list, dog_depths: dict,
                                max_dog_depth: float) -> set:
    """
    Return set of frame_keys where dog depth is below threshold.

    The idea: we only care about pointing when the dog is still near the
    starting position (hasn't moved to the target yet). Once the dog moves
    away (depth > threshold), pointing data is no longer relevant.

    Args:
        frame_keys: All available frame keys
        dog_depths: Dict from load_dog_depths()
        max_dog_depth: Max dog depth threshold (meters). Frames where dog is
            deeper than this are excluded.

    Returns:
        Set of frame_keys that pass the filter
    """
    valid = set()
    for fk in frame_keys:
        if fk in dog_depths:
            if dog_depths[fk] <= max_dog_depth:
                valid.add(fk)
        # If no dog detection for this frame, exclude it
        # (dog not visible = can't confirm it's still at start)

    return valid


# =============================================================================
# HUMAN POSITION VALIDATION & RE-DETECTION
# =============================================================================

# Image dimensions (RealSense cameras)
DEFAULT_IMAGE_HEIGHT = 480
DEFAULT_IMAGE_WIDTH = 640


def is_human_in_bottom_quarter(landmarks_2d: list, image_height: int = DEFAULT_IMAGE_HEIGHT) -> bool:
    """
    Check if the detected human is in the bottom 1/4 of the frame.

    When there are two people in the scene, MediaPipe sometimes locks onto
    the wrong one (e.g., someone sitting in the bottom of the frame).
    The pointing human should be in the upper 3/4.

    Args:
        landmarks_2d: List of [x, y, visibility] tuples in pixel coords
        image_height: Image height in pixels (default: 480 for RealSense)

    Returns:
        True if human is in bottom 1/4 (bad detection)
    """
    if not landmarks_2d:
        return False

    threshold_y = image_height * 0.75  # Bottom 1/4 starts at 75%

    # Check key upper-body landmarks: nose (0), left shoulder (11), right shoulder (12)
    key_indices = [0, 11, 12]
    key_ys = []

    for idx in key_indices:
        if idx < len(landmarks_2d):
            lm = landmarks_2d[idx]
            if len(lm) >= 2 and lm[1] is not None:
                key_ys.append(lm[1])

    if not key_ys:
        return False

    # If the average Y of nose + shoulders is in bottom 1/4, it's a bad detection
    avg_y = sum(key_ys) / len(key_ys)
    return avg_y > threshold_y


def find_bad_human_frames(skeleton_data: dict, image_height: int = DEFAULT_IMAGE_HEIGHT) -> list:
    """
    Find frames where human detection is in the bottom 1/4 of the frame.

    Args:
        skeleton_data: Dict from skeleton_2d.json {frame_key: {landmarks_2d: ...}}
        image_height: Image height in pixels

    Returns:
        List of frame_keys with bad human detection
    """
    bad_frames = []
    for frame_key, data in skeleton_data.items():
        landmarks_2d = data.get('landmarks_2d', [])
        if is_human_in_bottom_quarter(landmarks_2d, image_height):
            bad_frames.append(frame_key)
    return bad_frames


def redetect_bad_frames(bad_frames: list, raw_cam_path: Path, skeleton_data: dict,
                        image_height: int = DEFAULT_IMAGE_HEIGHT) -> dict:
    """
    Re-detect human skeleton on frames where detection was in bottom 1/4.

    Crops the bottom 1/4 of the image so MediaPipe can't see the wrong person,
    then re-runs detection on the top 3/4.

    Args:
        bad_frames: List of frame_keys to re-detect
        raw_cam_path: Path to raw camera data (with color/ and depth/ subfolders)
        skeleton_data: Original skeleton data dict (will be updated in place)
        image_height: Image height in pixels

    Returns:
        Updated skeleton_data dict with corrected frames
    """
    import cv2
    from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
    from batch_process_study import get_camera_intrinsics

    color_folder = raw_cam_path / "color"
    depth_folder = raw_cam_path / "depth"

    if not color_folder.exists():
        print(f"    WARNING: No color folder at {color_folder} - cannot re-detect")
        return skeleton_data

    detector = MediaPipeHumanDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        crop_top_ratio=0.6  # Keep top 60% of image to exclude handler in bottom
    )

    # Get intrinsics from first frame
    sample_frames = sorted(color_folder.glob("frame_*.png"))
    if not sample_frames:
        print(f"    WARNING: No color frames found - cannot re-detect")
        return skeleton_data

    sample = cv2.imread(str(sample_frames[0]))
    h, w = sample.shape[:2]
    fx, fy, cx, cy = get_camera_intrinsics(w, h)

    has_depth = depth_folder.exists()
    crop_h = int(h * 0.75)  # Keep top 3/4

    fixed_count = 0
    failed_count = 0

    for frame_key in bad_frames:
        frame_num = int(frame_key.split('_')[-1])
        color_path = color_folder / f"frame_{frame_num:06d}.png"

        if not color_path.exists():
            failed_count += 1
            continue

        color_img = cv2.imread(str(color_path))
        if color_img is None:
            failed_count += 1
            continue

        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Crop bottom 1/4: keep only top 3/4
        cropped_rgb = color_rgb[:crop_h, :, :]

        # Load and crop depth if available
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
                if depth_img is not None:
                    depth_img = depth_img[:crop_h, :]  # Crop depth too
            except Exception:
                depth_img = None

        # Re-detect on cropped image
        result = detector.detect_frame(
            cropped_rgb, frame_num,
            depth_image=depth_img,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        if result is not None:
            # 2D coords from cropped image are already correct for full image
            # (we cropped the bottom, so top pixels are unchanged)
            skeleton_data[frame_key] = result.to_dict()
            fixed_count += 1
        else:
            # MediaPipe found no one in the cropped frame — remove this frame
            del skeleton_data[frame_key]
            failed_count += 1

    print(f"    Re-detected: {fixed_count} fixed, {failed_count} failed/removed")
    return skeleton_data


# =============================================================================
# REPROCESS: Pointing analysis from saved skeleton data
# =============================================================================

def reprocess_pointing(output_path: Path, trial_camera_list: list = None,
                       arm_overrides: dict = None,
                       trial_filter: str = None, camera_filter: str = None,
                       dog_depth_max: float = None, subject_type: str = 'dog',
                       raw_data_path: Path = None):
    """
    Reprocess pointing analysis from saved skeleton data.

    Args:
        output_path: Path to study output folder
        trial_camera_list: Optional list of (trial, camera) to reprocess.
            If None, reprocesses all.
        arm_overrides: Optional dict from load_arm_overrides()
        trial_filter: Only process this trial
        camera_filter: Only process this camera
        dog_depth_max: If set, only include frames where dog depth < this value (meters).
            Use ~2.5m to only analyze pointing while dog is still at starting position.
        subject_type: Subject type for depth filtering ('dog' or 'baby')
        raw_data_path: Optional path to raw data (with trial_*/cam*/color/ structure).
            If provided, frames with bad human detection (bottom 1/4) will be re-detected.
    """
    from step2_skeleton_extraction.skeleton_base import SkeletonResult
    from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
    from step2_skeleton_extraction.pointing_analysis import analyze_pointing_frame
    from step2_skeleton_extraction.csv_exporter import export_pointing_analysis_to_csv
    from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace
    from step2_skeleton_extraction.kalman_filter import smooth_pointing_analyses
    from step2_skeleton_extraction.plot_distance_to_targets import (
        plot_distance_to_targets,
        plot_distance_summary,
        plot_best_representation_analysis
    )

    output_path = Path(output_path)
    detector = MediaPipeHumanDetector(crop_top_ratio=0.6)

    print(f"\n{'='*60}")
    print(f"  REPROCESS POINTING: {output_path.name}")
    if dog_depth_max is not None:
        print(f"  Dog depth filter: < {dog_depth_max}m")
    print(f"{'='*60}")

    ok_count = 0
    skip_count = 0
    error_count = 0

    for trial_dir in sorted(output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue
        if trial_filter and trial_dir.name != trial_filter:
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue
            if camera_filter and cam_dir.name != camera_filter:
                continue

            trial_name = trial_dir.name
            camera_name = cam_dir.name

            # If we have a specific list, check it
            if trial_camera_list is not None:
                if (trial_name, camera_name) not in trial_camera_list:
                    continue

            # Check arm override
            override_arm = None
            if arm_overrides:
                for key, info in arm_overrides.items():
                    if key[1] == trial_name and key[2] == camera_name:
                        override_arm = info.get('override_arm')
                        if override_arm == 'skip':
                            print(f"\n  SKIP: {trial_name}/{camera_name} (marked skip in CSV)")
                            skip_count += 1
                            continue
                        if not info.get('reprocess', True) and trial_camera_list is None:
                            skip_count += 1
                            continue
                        break

            print(f"\n  Processing: {trial_name}/{camera_name}")

            # Construct raw data cam path if raw_data_path provided
            raw_cam_path = None
            if raw_data_path:
                raw_cam_path = raw_data_path / trial_name / camera_name

            try:
                success = _reprocess_single(
                    cam_dir, detector, override_arm,
                    analyze_pointing_frame, export_pointing_analysis_to_csv,
                    plot_2d_pointing_trace, smooth_pointing_analyses,
                    plot_distance_to_targets, plot_distance_summary,
                    plot_best_representation_analysis,
                    dog_depth_max=dog_depth_max,
                    subject_type=subject_type,
                    raw_cam_path=raw_cam_path
                )
                if success:
                    # Also regenerate dog/subject CSV with updated targets
                    _regenerate_subject_csv(cam_dir, subject_type)
                    ok_count += 1
                else:
                    skip_count += 1
            except Exception as e:
                print(f"    ERROR: {e}")
                traceback.print_exc()
                error_count += 1

    print(f"\n{'='*60}")
    print(f"  DONE: OK={ok_count} | SKIP={skip_count} | ERROR={error_count}")
    print(f"{'='*60}")


def _regenerate_subject_csv(cam_dir: Path, subject_type: str = 'dog'):
    """
    Regenerate the subject (dog/baby) distance CSV with updated targets.

    When fix_targets replaces target_detections_cam_frame.json, the dog CSV
    (processed_dog_result_table.csv) still has distances computed with old targets.
    This function regenerates it using the current (fixed) targets.
    """
    try:
        from step3_subject_extraction.dog_csv_exporter import DogCSVExporter

        subject_results = cam_dir / f"{subject_type}_detection_results.json"
        targets_file = cam_dir / "target_detections_cam_frame.json"
        skeleton_file = cam_dir / "skeleton_2d.json"
        csv_out = cam_dir / f"processed_{subject_type}_result_table.csv"

        if not subject_results.exists() or not targets_file.exists():
            return  # No dog data or no targets - nothing to regenerate

        exporter = DogCSVExporter()
        exporter.export_to_csv(
            dog_results_path=subject_results,
            human_results_path=skeleton_file if skeleton_file.exists() else None,
            targets_path=targets_file,
            output_csv_path=csv_out,
        )
    except Exception as e:
        print(f"    Warning: subject CSV regeneration failed: {e}")


def _regenerate_distance_from_csv_plot(cam_dir: Path, trial_name: str):
    """
    Regenerate pointing_distance_to_targets_from_json.png from processed_gesture.csv.

    This plot is initially only generated by batch_process_study.py (initial pipeline).
    We regenerate it here so it reflects any target fixes or arm overrides.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_path = cam_dir / "processed_gesture.csv"
    if not csv_path.exists():
        return

    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        vec_types = ['eye_to_wrist', 'shoulder_to_wrist', 'elbow_to_wrist',
                     'nose_to_wrist', 'head_orientation']
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

        if not target_cols:
            return

        n_targets = max(len(v) for v in target_cols.values())
        fig, axes = plt.subplots(n_targets, 1, figsize=(14, 3 * n_targets), sharex=True)
        if n_targets == 1:
            axes = [axes]

        x_axis = df['frame'] if 'frame' in df.columns else range(len(df))

        for target_i in range(n_targets):
            ax = axes[target_i]
            for vt, color in zip(vec_types, colors):
                col = f'{vt}_dist_to_target_{target_i + 1}'
                if col in df.columns:
                    valid = df[col].dropna()
                    if len(valid) > 0:
                        ax.plot(x_axis[:len(valid)], valid.values, '-',
                                color=color, label=vt, alpha=0.7)
            ax.set_ylabel(f'Dist to target {target_i + 1} (m)')
            ax.legend(fontsize=7, loc='upper right')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        cam_name = cam_dir.name
        fig.suptitle(f'{trial_name} — Pointing Vector Distance to Targets (from CSV)',
                     fontsize=12)
        plt.tight_layout()

        out_path = cam_dir / "pointing_distance_to_targets_from_json.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"    Warning: pointing_distance_to_targets_from_json.png failed: {e}")


def _reprocess_single(cam_dir: Path, detector, override_arm: str,
                       analyze_pointing_frame, export_pointing_analysis_to_csv,
                       plot_2d_pointing_trace, smooth_pointing_analyses,
                       plot_distance_to_targets, plot_distance_summary,
                       plot_best_representation_analysis,
                       dog_depth_max: float = None,
                       subject_type: str = 'dog',
                       raw_cam_path: Path = None) -> bool:
    """Reprocess pointing for a single trial/camera."""

    skeleton_file = cam_dir / "skeleton_2d.json"
    target_file = cam_dir / "target_detections_cam_frame.json"

    if not skeleton_file.exists():
        print(f"    No skeleton_2d.json - SKIP")
        return False

    if not target_file.exists():
        print(f"    No targets - SKIP")
        return False

    # Load skeleton data
    with open(skeleton_file) as f:
        skeleton_data = json.load(f)

    # Load targets
    with open(target_file) as f:
        targets = json.load(f)

    if not targets:
        print(f"    Empty target file - SKIP")
        return False

    # Check for bad human detections (human in bottom 1/4 of frame)
    bad_frames = find_bad_human_frames(skeleton_data)
    if bad_frames:
        print(f"    Found {len(bad_frames)} frames with human in bottom 1/4")
        if raw_cam_path and raw_cam_path.exists():
            print(f"    Re-detecting from raw data: {raw_cam_path}")
            skeleton_data = redetect_bad_frames(bad_frames, raw_cam_path, skeleton_data)
            # Save the corrected skeleton data
            with open(skeleton_file, 'w') as f:
                json.dump(skeleton_data, f, indent=2)
            print(f"    Updated skeleton_2d.json ({len(skeleton_data)} frames)")
        else:
            # No raw data — just remove the bad frames
            for fk in bad_frames:
                if fk in skeleton_data:
                    del skeleton_data[fk]
            print(f"    Removed {len(bad_frames)} bad frames (no raw data for re-detection)")

    # Dog depth filtering: only keep frames where dog is close enough
    if dog_depth_max is not None:
        dog_depths = load_dog_depths(cam_dir, subject_type=subject_type)
        if not dog_depths:
            print(f"    WARNING: No {subject_type} detection data for depth filtering")
            print(f"    Loaded {len(skeleton_data)} frames, {len(targets)} targets (no depth filter)")
        else:
            valid_frames = filter_frames_by_dog_depth(
                list(skeleton_data.keys()), dog_depths, dog_depth_max
            )
            original_count = len(skeleton_data)
            filtered_data = {k: v for k, v in skeleton_data.items() if k in valid_frames}
            print(f"    Loaded {original_count} frames, kept {len(filtered_data)} "
                  f"where {subject_type} depth < {dog_depth_max}m "
                  f"({original_count - len(filtered_data)} filtered out)")

            if not filtered_data:
                print(f"    WARNING: No frames remain after depth filtering — using ALL frames instead")
            else:
                skeleton_data = filtered_data
    else:
        print(f"    Loaded {len(skeleton_data)} frames, {len(targets)} targets")

    # Shift human Z backward to match physical layout
    # Physical: human stands behind ALL targets (cups are on a table in front of them)
    # MediaPipe underestimates human depth, so we shift all 3D landmarks backward
    # Anchor: place human 5cm behind the furthest target (back row)
    # Guard: only apply if not already shifted (check human_center.json for marker)
    HUMAN_BEHIND_BACK_ROW_CM = 5.0
    hc_file = cam_dir / "human_center.json"
    already_shifted = False
    if hc_file.exists():
        with open(hc_file) as f:
            hc_check = json.load(f)
        already_shifted = 'z_shift_applied' in hc_check

    if targets and not already_shifted:
        max_target_z = max(t['z'] for t in targets)
        expected_human_z = max_target_z + HUMAN_BEHIND_BACK_ROW_CM / 100.0

        # Compute actual median human shoulder Z from skeleton
        shoulder_zs = []
        for fk, data in skeleton_data.items():
            lm3d = data.get('landmarks_3d', [])
            if lm3d and len(lm3d) >= 13:
                lsh, rsh = lm3d[11], lm3d[12]  # left/right shoulder
                if len(lsh) >= 3 and len(rsh) >= 3:
                    shoulder_zs.append((lsh[2] + rsh[2]) / 2.0)

        if shoulder_zs:
            actual_human_z = float(np.median(shoulder_zs))
            z_shift = expected_human_z - actual_human_z

            if abs(z_shift) > 0.01:  # only shift if meaningful
                print(f"    Human Z shift: {z_shift:+.3f}m "
                      f"(shoulder {actual_human_z:.3f}m → {expected_human_z:.3f}m, "
                      f"furthest target at {max_target_z:.3f}m)")

                # Apply shift to all 3D landmarks
                for fk, data in skeleton_data.items():
                    lm3d = data.get('landmarks_3d', [])
                    if lm3d:
                        for i in range(len(lm3d)):
                            if len(lm3d[i]) >= 3:
                                lm3d[i][2] += z_shift

                # Save updated skeleton data
                with open(skeleton_file, 'w') as f:
                    json.dump(skeleton_data, f, indent=2)

                # Update human_center.json (create if needed) with shift marker
                hc_data = {}
                if hc_file.exists():
                    with open(hc_file) as f:
                        hc_data = json.load(f)
                hc = hc_data.get('human_center', [0, 0, 0])
                if len(hc) >= 3:
                    hc[2] += z_shift
                hc_data['human_center'] = hc
                hc_data['z_shift_applied'] = z_shift
                with open(hc_file, 'w') as f:
                    json.dump(hc_data, f, indent=2)

    # Determine pointing arm
    if override_arm and override_arm in ['left', 'right']:
        pointing_arm = override_arm
        print(f"    Arm: {pointing_arm} (OVERRIDE)")
    else:
        # Load from existing pointing_hand.json
        pointing_file = cam_dir / "pointing_hand.json"
        if pointing_file.exists():
            with open(pointing_file) as f:
                ph_data = json.load(f)
                pointing_arm = ph_data.get('pointing_hand', 'right')
        else:
            pointing_arm = 'right'
        print(f"    Arm: {pointing_arm} (existing)")

    # Load ground plane rotation
    ground_plane_rotation = None
    transform_file = cam_dir / "ground_plane_transform.json"
    if transform_file.exists():
        with open(transform_file) as f:
            transform_data = json.load(f)
            ground_plane_rotation = np.array(transform_data['rotation_matrix'])
        print(f"    Ground plane: {transform_data.get('transform_type', 'loaded')}")

    # Reconstruct SkeletonResult objects and recompute arm vectors
    from step2_skeleton_extraction.skeleton_base import SkeletonResult

    human_results = {}
    for frame_key, data in skeleton_data.items():
        result = SkeletonResult(
            frame_number=data.get('frame', 0),
            landmarks_2d=data.get('landmarks_2d', []),
            landmarks_3d=data.get('landmarks_3d', []),
            arm_vectors=data.get('arm_vectors', {}),
            metadata=data.get('metadata', {})
        )

        # Recompute arm vectors with (possibly new) arm
        if result.landmarks_3d:
            result.arm_vectors = detector._compute_arm_vectors(
                result.landmarks_3d, pointing_arm
            )
            result.metadata['pointing_arm'] = pointing_arm
            result.metadata['pointing_hand_whole_trial'] = pointing_arm

        human_results[frame_key] = result

    # Update pointing_hand.json
    with open(cam_dir / "pointing_hand.json", 'w') as f:
        json.dump({
            "pointing_hand": pointing_arm,
            "total_frames": len(human_results),
            "frame_distribution": {pointing_arm: len(human_results)},
            "override": bool(override_arm)
        }, f, indent=2)

    # Run pointing analysis
    analyses = {}
    for frame_key, result in human_results.items():
        if result.landmarks_3d:
            analysis = analyze_pointing_frame(
                result, targets,
                pointing_arm=pointing_arm,
                ground_plane_rotation=ground_plane_rotation
            )
            if analysis:
                analyses[frame_key] = analysis

    print(f"    Pointing analysis: {len(analyses)} frames")

    if not analyses:
        print(f"    WARNING: No valid pointing analyses")
        return False

    # Apply Kalman filtering
    analyses = smooth_pointing_analyses(
        analyses, process_noise=0.01, measurement_noise=0.1
    )

    # Export CSV
    csv_path = cam_dir / "processed_gesture.csv"
    export_pointing_analysis_to_csv(human_results, analyses, csv_path)

    # Human center for plotting
    human_center = [0, 0, 0]
    hc_file = cam_dir / "human_center.json"
    if hc_file.exists():
        with open(hc_file) as f:
            hc_data = json.load(f)
            human_center = hc_data.get('human_center', human_center)

    # Generate plots
    trial_name = f"{cam_dir.parent.name}_{cam_dir.name}"

    plot_path = cam_dir / "2d_pointing_trace.png"
    plot_2d_pointing_trace(analyses, targets, human_center, plot_path,
                           trial_name=trial_name, use_fixed_axes=True)

    dist_plot = cam_dir / "distance_to_targets_timeseries.png"
    plot_distance_to_targets(analyses, targets, dist_plot, trial_name=trial_name)

    summary_plot = cam_dir / "distance_to_targets_summary.png"
    plot_distance_summary(analyses, targets, summary_plot, trial_name=trial_name)

    accuracy_plot = cam_dir / "pointing_accuracy_comparison.png"
    plot_best_representation_analysis(analyses, targets, accuracy_plot, trial_name=trial_name)

    # Regenerate pointing_distance_to_targets_from_json.png (initially only from initial pipeline)
    _regenerate_distance_from_csv_plot(cam_dir, trial_name)

    print(f"    Done - CSV + 5 plots generated")
    return True


# =============================================================================
# GENERATE ARM OVERRIDE CSV
# =============================================================================

def generate_arm_csv(output_path: Path, csv_out: Path = None) -> Path:
    """Generate arm override CSV from existing output data."""
    import pandas as pd

    output_path = Path(output_path)
    rows = []

    for trial_dir in sorted(output_path.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith('trial_'):
            continue

        for cam_dir in sorted(trial_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith('cam'):
                continue

            detected_arm = "unknown"
            pointing_file = cam_dir / "pointing_hand.json"
            if pointing_file.exists():
                with open(pointing_file) as f:
                    data = json.load(f)
                    detected_arm = data.get('pointing_hand', 'unknown')

            num_targets = 0
            target_file = cam_dir / "target_detections_cam_frame.json"
            if target_file.exists():
                with open(target_file) as f:
                    num_targets = len(json.load(f))

            rows.append({
                'trial': trial_dir.name,
                'camera': cam_dir.name,
                'detected_arm': detected_arm,
                'num_targets': num_targets,
                'override_arm': '',
                'reprocess': 'no'
            })

    df = pd.DataFrame(rows)

    if csv_out is None:
        csv_out = output_path.parent / f"{output_path.name}_postprocess.csv"

    df.to_csv(csv_out, index=False)
    print(f"\nGenerated: {csv_out}")
    print(f"  {len(rows)} entries")
    print(f"\nEdit the CSV:")
    print(f"  - Set 'override_arm' to left/right/skip")
    print(f"  - Set 'reprocess' to yes for trials to fix")
    print(f"\nThen run:")
    print(f"  python batch_postprocess.py reprocess {output_path} --arm-csv {csv_out}")

    return csv_out


def load_arm_overrides(csv_path: Path) -> dict:
    """Load arm overrides from CSV."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    overrides = {}

    for _, row in df.iterrows():
        trial = str(row.get('trial', '')).strip()
        camera = str(row.get('camera', '')).strip()

        override_arm = str(row.get('override_arm', '')).strip().lower()
        if override_arm not in ['left', 'right', 'skip']:
            override_arm = None

        reprocess = str(row.get('reprocess', '')).strip().lower()
        reprocess = reprocess in ['yes', 'true', '1', 'y']

        key = ('', trial, camera)
        overrides[key] = {
            'override_arm': override_arm,
            'reprocess': reprocess
        }

    return overrides


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Post-process pointing gesture analysis from saved output data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See status of all trials
  python batch_postprocess.py scan /path/to/study_output

  # Fix targets + reprocess in one step
  python batch_postprocess.py fix-targets /path/to/study_output --reprocess

  # Reprocess with arm overrides
  python batch_postprocess.py reprocess /path/to/study_output --arm-csv arms.csv

  # Generate arm override CSV
  python batch_postprocess.py generate-csv /path/to/study_output
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # --- scan ---
    scan_parser = subparsers.add_parser('scan', help='Show status of all trials')
    scan_parser.add_argument('output_path', type=str, help='Path to study output folder')

    # --- fix-targets ---
    fix_parser = subparsers.add_parser('fix-targets', help='Fix missing/bad targets using reference')
    fix_parser.add_argument('output_path', type=str, help='Path to study output folder')
    fix_parser.add_argument('--max-depth-diff', type=float, default=1.0,
                           help='Max depth difference for reference validation (default: 1.0m)')
    fix_parser.add_argument('--max-position-diff', type=float, default=1.0,
                           help='Max position difference before replacing (default: 1.0m)')
    fix_parser.add_argument('--dry-run', action='store_true',
                           help='Show what would be fixed without changing files')
    fix_parser.add_argument('--reprocess', action='store_true',
                           help='Also reprocess pointing for fixed trials')
    fix_parser.add_argument('--arm-csv', type=str, default=None,
                           help='Optional arm override CSV for reprocessing')
    fix_parser.add_argument('--dog-depth-max', type=float, default=None,
                           help='Only include frames where dog depth < this (meters)')
    fix_parser.add_argument('--subject', type=str, default='dog',
                           choices=['dog', 'baby'],
                           help='Subject type for depth filtering (default: dog)')
    fix_parser.add_argument('--raw-data', type=str, default=None,
                           help='Path to raw data (trial_*/cam*/color/) for re-detecting bad human frames')

    # --- reprocess ---
    reprocess_parser = subparsers.add_parser('reprocess', help='Reprocess pointing from saved data')
    reprocess_parser.add_argument('output_path', type=str, help='Path to study output folder')
    reprocess_parser.add_argument('--arm-csv', type=str, default=None,
                                  help='Path to arm override CSV')
    reprocess_parser.add_argument('--trial', type=str, default=None,
                                  help='Only process this trial')
    reprocess_parser.add_argument('--camera', type=str, default=None,
                                  help='Only process this camera')
    reprocess_parser.add_argument('--dog-depth-max', type=float, default=None,
                                  help='Only include frames where dog depth < this (meters). '
                                       'Use ~2.5 to filter pointing to when dog is still at start.')
    reprocess_parser.add_argument('--subject', type=str, default='dog',
                                  choices=['dog', 'baby'],
                                  help='Subject type for depth filtering (default: dog)')
    reprocess_parser.add_argument('--raw-data', type=str, default=None,
                                  help='Path to raw data (trial_*/cam*/color/) for re-detecting bad human frames')

    # --- generate-csv ---
    csv_parser = subparsers.add_parser('generate-csv', help='Generate arm override CSV template')
    csv_parser.add_argument('output_path', type=str, help='Path to study output folder')
    csv_parser.add_argument('--output', type=str, default=None, help='Output CSV path')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'scan':
        scan_study_output(Path(args.output_path))

    elif args.command == 'fix-targets':
        fixed = fix_targets(
            Path(args.output_path),
            max_depth_diff=args.max_depth_diff,
            max_position_diff=args.max_position_diff,
            dry_run=args.dry_run
        )

        if args.reprocess and fixed and not args.dry_run:
            arm_overrides = None
            if args.arm_csv:
                arm_overrides = load_arm_overrides(Path(args.arm_csv))

            reprocess_pointing(
                Path(args.output_path),
                trial_camera_list=fixed,
                arm_overrides=arm_overrides,
                dog_depth_max=args.dog_depth_max,
                subject_type=args.subject,
                raw_data_path=Path(args.raw_data) if args.raw_data else None
            )

    elif args.command == 'reprocess':
        arm_overrides = None
        if args.arm_csv:
            arm_overrides = load_arm_overrides(Path(args.arm_csv))

        reprocess_pointing(
            Path(args.output_path),
            arm_overrides=arm_overrides,
            trial_filter=args.trial,
            camera_filter=args.camera,
            dog_depth_max=args.dog_depth_max,
            subject_type=args.subject,
            raw_data_path=Path(args.raw_data) if args.raw_data else None
        )

    elif args.command == 'generate-csv':
        generate_arm_csv(
            Path(args.output_path),
            csv_out=Path(args.output) if args.output else None
        )


if __name__ == "__main__":
    main()

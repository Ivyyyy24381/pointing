#!/usr/bin/env python3
"""
Run full post-processing pipeline on ALL studies in the output folder.

Steps for each study:
  1. Scan status
  2. Fix targets (find reference, apply to bad trials)
  3. Reprocess pointing analysis (with optional dog depth filter + arm CSV)
  4. Regenerate all visualizations

Usage:
    # Process all studies with defaults
    python run_all_postprocess.py pointing_data/output/

    # With dog depth filter + camera filter
    python run_all_postprocess.py pointing_data/output/ --camera cam1 --dog-depth-max 3.0

    # With annotations CSV (auto-generates arm overrides from ground truth)
    python run_all_postprocess.py pointing_data/output/ --annotations-csv annotations.csv

    # With raw data root for re-detecting bad human frames
    python run_all_postprocess.py pointing_data/output/ --raw-data-root pointing_data/

    # Full pipeline: fix targets + arm overrides + depth filter + human re-detection
    python run_all_postprocess.py pointing_data/output/ --camera cam1 --dog-depth-max 3.0 \
        --annotations-csv annotations.csv --raw-data-root pointing_data/

    # Dry run (show what would happen, no changes)
    python run_all_postprocess.py pointing_data/output/ --dry-run

    # Skip reprocessing, only regenerate plots
    python run_all_postprocess.py pointing_data/output/ --plots-only
"""

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_annotations(csv_path: Path) -> dict:
    """
    Load the annotated CSV with ground truth arm info.

    Returns dict mapping participant_id -> list of {trial_number, arm, good, error} dicts.
    Trial mapping: CSV trial_number N -> output folder trial_(N-1).
    """
    annotations = {}

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            excluded = row.get('excluded', '').strip().upper()
            if excluded == 'Y':
                continue

            participant_id = row.get('participant_id', '').strip()
            if not participant_id:
                continue

            trial_num = row.get('trial_number', '').strip()
            if not trial_num or not trial_num.isdigit():
                continue

            arm = row.get('Owner Arm Used', '').strip().upper()
            if arm == 'L':
                arm = 'left'
            elif arm == 'R':
                arm = 'right'
            else:
                arm = None

            good = row.get('Output good?', '').strip().upper() == 'Y'
            error = row.get("If no - what's wrong", '').strip()

            annotations.setdefault(participant_id, []).append({
                'trial_number': int(trial_num),
                'trial_folder': f"trial_{int(trial_num) - 1}",
                'arm': arm,
                'good': good,
                'error': error,
            })

    return annotations


def build_arm_overrides_from_annotations(annotations: dict, study_name: str,
                                          camera: str = None,
                                          study_path: Path = None) -> dict:
    """
    Build arm overrides dict (compatible with batch_postprocess.load_arm_overrides format)
    from annotations for a specific study.

    Handles P1/P2 split studies: if study_name contains _P1_ or _P2_, only the
    relevant subset of CSV trials is mapped, with trial folders renumbered from 0.

    Args:
        annotations: Dict from load_annotations()
        study_name: Output study folder name (e.g., BDL396_Willow_OWN039_PVPO_16_output)
        camera: Camera to generate overrides for (e.g., cam1). If None, generates for all.
        study_path: Path to study output folder (used to find sibling P1 for P2 studies)

    Returns:
        Dict compatible with batch_postprocess arm_overrides format
    """
    import re

    # Extract participant_id from study name (first part before _)
    participant_id = study_name.split('_')[0]

    if participant_id not in annotations:
        return {}

    # Sort entries by trial_number
    entries = sorted(annotations[participant_id], key=lambda e: e['trial_number'])

    # Detect P1/P2 split
    # Strip _output suffix for matching
    base_name = study_name.replace('_output', '')
    p1_match = re.search(r'_P1$', base_name)
    p2_match = re.search(r'_P2$', base_name)

    if p1_match:
        # P1: count how many trial folders exist in this study
        p1_trial_count = 0
        if study_path and study_path.exists():
            p1_trial_count = len([d for d in study_path.iterdir()
                                  if d.is_dir() and d.name.startswith('trial_')])
        if p1_trial_count == 0:
            p1_trial_count = len(entries)  # fallback

        # Only use the first p1_trial_count entries
        entries = entries[:p1_trial_count]
        # Trial mapping: CSV trial N → trial_(N-1) as usual

    elif p2_match:
        # P2: find sibling P1 to determine offset
        p1_trial_count = 0
        if study_path and study_path.exists():
            p1_name = base_name[:-1] + '1'  # Replace P2 with P1
            p1_path = study_path.parent / (p1_name + '_output')
            if p1_path.exists():
                p1_trial_count = len([d for d in p1_path.iterdir()
                                      if d.is_dir() and d.name.startswith('trial_')])

        if p1_trial_count == 0:
            # Fallback: estimate from total entries vs P2 trial count
            p2_trial_count = 0
            if study_path and study_path.exists():
                p2_trial_count = len([d for d in study_path.iterdir()
                                      if d.is_dir() and d.name.startswith('trial_')])
            if p2_trial_count > 0:
                p1_trial_count = len(entries) - p2_trial_count

        if p1_trial_count > 0:
            # Only use entries after P1's trials, remap to trial_0..
            entries = entries[p1_trial_count:]
            # Remap trial folders: trial_0, trial_1, ...
            for i, entry in enumerate(entries):
                entry = dict(entry)  # copy to avoid mutating original
                entry['trial_folder'] = f"trial_{i}"
                entries[i] = entry

    overrides = {}
    for entry in entries:
        trial_folder = entry['trial_folder']
        arm = entry['arm']

        if arm is None:
            continue

        # Generate for specified camera or all common cameras
        cameras = [camera] if camera else ['cam1', 'cam2', 'cam3']
        for cam in cameras:
            key = ('', trial_folder, cam)
            overrides[key] = {
                'override_arm': arm,
                'reprocess': True,
            }

    return overrides


def find_raw_data_path(study_path: Path, raw_data_root: Path) -> Path:
    """
    Find the raw data path for a study by mapping output folder name to raw folder.

    Output folders have _output suffix: BDL396_Willow_OWN039_PVPO_16_output
    Raw data folders don't: BDL396_Willow_OWN039_PVPO_16

    Args:
        study_path: Path to the output study folder
        raw_data_root: Root path containing raw data folders

    Returns:
        Path to raw data folder, or None if not found
    """
    study_name = study_path.name

    # Strip _output suffix
    if study_name.endswith('_output'):
        raw_name = study_name[:-7]  # Remove '_output'
    else:
        raw_name = study_name

    raw_path = raw_data_root / raw_name
    if raw_path.exists():
        return raw_path

    # Fallback: try to find by participant_id prefix
    participant_id = study_name.split('_')[0]
    for d in raw_data_root.iterdir():
        if d.is_dir() and d.name.startswith(participant_id) and not d.name.endswith('_output'):
            return d

    return None


def _apply_physical_layout(targets: list, camera: str = 'cam1') -> list:
    """
    Adjust target positions to respect the known physical layout.

    Physical layout for cam1 (measured ground truth):
        Order (left to right in camera view): cup4, cup3, (human), cup2, cup1

        X axis (horizontal):
            cup1 ↔ cup4: 210 cm total span
            cup2 ↔ cup3:  78 cm (inner pair, roughly centered in span)

        Z axis (depth):
            cup1 & cup4: closer to camera (front row)
            cup2 & cup3: 32 cm further from camera (back row)
            human: ~24 cm behind cup1

    This keeps the overall center/span from detection but enforces:
    - Inner pair (cup2/cup3) horizontal spacing = 78/210 of total span
    - Depth difference between front pair (1,4) and back pair (2,3) = 32cm
    """
    if camera != 'cam1' or len(targets) != 4:
        return targets

    # Physical constraints
    TOTAL_SPAN_CM = 210.0     # cup1 ↔ cup4 horizontal
    INNER_SPAN_CM = 78.0      # cup2 ↔ cup3 horizontal
    DEPTH_DIFF_CM = 32.0      # cup2/cup3 are 32cm further than cup1/cup4

    by_label = {t['label']: t for t in targets}
    x1 = by_label['target_1']['x']  # rightmost (cup1)
    x4 = by_label['target_4']['x']  # leftmost (cup4)
    total_span = x1 - x4  # positive, in camera coords (meters)

    # Scale factor: camera meters per physical cm
    scale = total_span / TOTAL_SPAN_CM

    # Center of array
    x_center = (x1 + x4) / 2.0

    # Horizontal positions: cup2 and cup3 are the inner pair, roughly centered
    # cup2 is right-of-center, cup3 is left-of-center
    inner_half = (INNER_SPAN_CM / 2.0) * scale
    x_cup2 = x_center + inner_half
    x_cup3 = x_center - inner_half

    # Depth: front row (cup1, cup4) and back row (cup2, cup3)
    z_front = (by_label['target_1']['z'] + by_label['target_4']['z']) / 2.0
    z_back = z_front + DEPTH_DIFF_CM / 100.0  # 32cm further from camera

    adjusted = []
    for t in targets:
        t_new = dict(t)
        label = t['label']
        if label == 'target_1':
            t_new['x'] = x1              # keep rightmost anchor
            t_new['z'] = z_front
            t_new['avg_depth_m'] = z_front
        elif label == 'target_4':
            t_new['x'] = x4              # keep leftmost anchor
            t_new['z'] = z_front
            t_new['avg_depth_m'] = z_front
        elif label == 'target_2':
            t_new['x'] = x_cup2          # inner right
            t_new['z'] = z_back
            t_new['avg_depth_m'] = z_back
        elif label == 'target_3':
            t_new['x'] = x_cup3          # inner left
            t_new['z'] = z_back
            t_new['avg_depth_m'] = z_back
        adjusted.append(t_new)

    return adjusted


def build_cam1_physical_fallback() -> list:
    """
    Build hardcoded cam1 fallback targets from physical measurements alone.

    Used as the ultimate fallback when no YOLO detections exist at all.
    Positions are based on typical detected depths and the known physical layout.

    Physical constraints:
        X: cup1 ↔ cup4 = 210cm, cup2 ↔ cup3 = 78cm (centered)
        Z: cup2/cup3 are 32cm further from camera than cup1/cup4
    """
    # Typical observed values from cam1 across studies
    Z_FRONT = 3.85              # cup1, cup4 (closer to camera)
    Z_BACK = Z_FRONT + 0.32    # cup2, cup3 (32cm further)
    CENTER_X = 0.20             # slight offset from camera center
    TOTAL_SPAN = 1.34           # typical detected span in camera coords (meters)
    INNER_SPAN = TOTAL_SPAN * 78.0 / 210.0  # cup2-cup3 gap scaled to camera coords
    Y_DEFAULT = -0.20

    # Camera intrinsics (typical RealSense D435 at 640x480)
    FX, FY, CX, CY = 615.0, 615.0, 320.0, 240.0

    # Horizontal positions
    positions = {
        'target_1': CENTER_X + TOTAL_SPAN / 2.0,       # rightmost
        'target_2': CENTER_X + INNER_SPAN / 2.0,       # inner right
        'target_3': CENTER_X - INNER_SPAN / 2.0,       # inner left
        'target_4': CENTER_X - TOTAL_SPAN / 2.0,       # leftmost
    }
    depths = {
        'target_1': Z_FRONT, 'target_4': Z_FRONT,      # front row
        'target_2': Z_BACK,  'target_3': Z_BACK,       # back row (32cm further)
    }

    targets = []
    for label in ['target_1', 'target_2', 'target_3', 'target_4']:
        x = positions[label]
        z = depths[label]
        px = int(x * FX / z + CX)
        py = int(Y_DEFAULT * FY / z + CY)
        targets.append({
            'bbox': [px - 8, py - 8, px + 8, py + 8],
            'center_px': [px, py],
            'avg_depth_m': z,
            'x': round(x, 4),
            'y': round(Y_DEFAULT, 4),
            'z': z,
            'label': label,
        })

    return targets


def build_global_reference_targets(studies: list, camera_filter: str = None,
                                    max_depth_diff: float = 2.0) -> dict:
    """
    Build global reference targets from ALL studies for cross-study fallback.

    Collects reference targets from each study that has valid 4-target detections,
    then takes the global median. Positions are then adjusted to match the known
    physical layout (horizontal spacing ratios).

    If no studies have valid detections, falls back to hardcoded cam1 positions
    based on physical measurements.

    Args:
        studies: List of study Path objects
        camera_filter: Only build reference for this camera (e.g. 'cam1')
        max_depth_diff: Max target depth spread for validation

    Returns:
        Dict mapping camera_name -> list of 4 target dicts (global median positions)
    """
    import numpy as np
    from batch_postprocess import find_reference_targets

    all_references = {}  # camera -> list of (study_name, targets_list)

    for study_path in studies:
        ref = find_reference_targets(study_path, max_depth_diff=max_depth_diff)
        for cam_name, targets in ref.items():
            if camera_filter and cam_name != camera_filter:
                continue
            all_references.setdefault(cam_name, []).append((study_path.name, targets))

    global_ref = {}
    for cam_name, ref_list in sorted(all_references.items()):
        n = len(ref_list)
        if n == 0:
            continue

        # Stack all reference positions
        positions = np.zeros((n, 4, 3))
        pixel_centers = np.zeros((n, 4, 2))
        avg_depths = np.zeros((n, 4))

        for i, (study_name, targets) in enumerate(ref_list):
            for j, t in enumerate(targets):
                positions[i, j] = [t.get('x', 0), t.get('y', 0), t.get('z', 0)]
                px = t.get('center_px', [0, 0])
                pixel_centers[i, j] = px[:2] if len(px) >= 2 else [0, 0]
                avg_depths[i, j] = t.get('avg_depth_m', t.get('z', 0))

        median_pos = np.median(positions, axis=0)
        median_px = np.median(pixel_centers, axis=0)
        median_depth = np.median(avg_depths, axis=0)

        ref_targets = []
        for j in range(4):
            template = ref_list[0][1][j]
            ref_targets.append({
                'bbox': template.get('bbox', [0, 0, 0, 0]),
                'center_px': [int(median_px[j, 0]), int(median_px[j, 1])],
                'avg_depth_m': float(median_depth[j]),
                'x': float(median_pos[j, 0]),
                'y': float(median_pos[j, 1]),
                'z': float(median_pos[j, 2]),
                'label': f'target_{j+1}'
            })

        # Apply physical layout correction (adjusts horizontal spacing ratios)
        ref_targets = _apply_physical_layout(ref_targets, camera=cam_name)
        global_ref[cam_name] = ref_targets

    # If cam1 was requested but no studies had valid cam1 detections,
    # fall back to hardcoded physical layout
    cam1_key = 'cam1'
    if (camera_filter in (None, cam1_key)) and cam1_key not in global_ref:
        print(f"  WARNING: No cross-study detections for {cam1_key} — using hardcoded physical fallback")
        global_ref[cam1_key] = build_cam1_physical_fallback()

    return global_ref


def verify_arm_alignment(studies: list, annotations: dict, camera: str = 'cam1') -> list:
    """
    Compare detected pointing arm vs annotated arm for all studies.

    Returns a list of mismatch dicts for reporting.

    Args:
        studies: List of study Path objects
        annotations: Dict from load_annotations()
        camera: Camera to check

    Returns:
        List of {study, trial, detected_arm, annotated_arm} dicts for mismatches
    """
    import json

    mismatches = []
    total_checked = 0
    total_matched = 0

    for study_path in studies:
        # Use build_arm_overrides_from_annotations to get correct P1/P2 mapping
        overrides = build_arm_overrides_from_annotations(
            annotations, study_path.name, camera=camera, study_path=study_path
        )
        if not overrides:
            continue

        for key, info in overrides.items():
            trial_folder = key[1]
            cam_name = key[2]
            annotated_arm = info.get('override_arm')
            if annotated_arm is None:
                continue

            cam_dir = study_path / trial_folder / cam_name
            pointing_file = cam_dir / "pointing_hand.json"
            if not pointing_file.exists():
                continue

            try:
                with open(pointing_file) as f:
                    ph = json.load(f)
                detected_arm = ph.get('pointing_hand', 'unknown')
            except Exception:
                continue

            total_checked += 1
            if detected_arm == annotated_arm:
                total_matched += 1
            else:
                mismatches.append({
                    'study': study_path.name,
                    'trial': trial_folder,
                    'trial_number': 0,  # not easily available after remap
                    'detected_arm': detected_arm,
                    'annotated_arm': annotated_arm,
                    'was_override': ph.get('override', False),
                })

    print(f"\n{'='*60}")
    print(f"  ARM ALIGNMENT VERIFICATION")
    print(f"{'='*60}")
    print(f"  Checked: {total_checked} trial/camera combos")
    print(f"  Matched: {total_matched}")
    print(f"  Mismatches: {len(mismatches)}")

    if mismatches:
        print(f"\n  MISMATCHES:")
        for m in mismatches:
            override_str = " (was override)" if m['was_override'] else ""
            print(f"    {m['study']} / {m['trial']} (trial #{m['trial_number']}): "
                  f"detected={m['detected_arm']}, annotated={m['annotated_arm']}{override_str}")
    else:
        print(f"  All arms match annotations!")

    return mismatches


def find_studies(base_path: Path) -> list:
    """
    Find all study output folders under base_path.

    A study folder is one that contains trial_* subdirectories where
    each trial has cam* subdirectories with output data.

    Handles both:
    - Direct study path: base_path/trial_0/cam1/...
    - Parent of studies: base_path/BDL396_study1/trial_0/cam1/...
    """
    base_path = Path(base_path)

    # First check if any subdirectories are study folders (contain trial_*/cam*/)
    nested_studies = []
    for d in sorted(base_path.iterdir()):
        if not d.is_dir() or d.name.startswith('.'):
            continue
        # A study folder has trial_* dirs that have cam* dirs
        trial_dirs = [t for t in d.iterdir()
                      if t.is_dir() and t.name.startswith('trial_')]
        if trial_dirs:
            # Verify at least one trial has cam dirs
            has_cams = any(
                any(c.is_dir() and c.name.startswith('cam') for c in t.iterdir())
                for t in trial_dirs
            )
            if has_cams:
                nested_studies.append(d)

    if nested_studies:
        return nested_studies

    # If no nested studies, check if base_path itself is a study
    trial_dirs = [d for d in base_path.iterdir()
                  if d.is_dir() and d.name.startswith('trial_')]
    if trial_dirs:
        return [base_path]

    return []


def run_pipeline(study_path: Path,
                 dog_depth_max: float = None,
                 arm_csv: str = None,
                 arm_overrides: dict = None,
                 max_depth_diff: float = 2.0,
                 max_position_diff: float = 1.0,
                 dry_run: bool = False,
                 plots_only: bool = False,
                 z_min: float = 2.0,
                 z_max: float = 5.0,
                 subject_type: str = 'dog',
                 camera_filter: str = None,
                 raw_data_path: Path = None,
                 fallback_targets: dict = None):
    """Run the full pipeline on a single study."""

    from batch_postprocess import (
        scan_study_output,
        fix_targets,
        reprocess_pointing,
        load_arm_overrides
    )
    from batch_regenerate_plots import (
        find_camera_outputs,
        process_camera_output
    )

    print(f"\n{'#'*70}")
    print(f"  STUDY: {study_path.name}")
    print(f"  Path:  {study_path}")
    if raw_data_path:
        print(f"  Raw:   {raw_data_path}")
    print(f"{'#'*70}")

    # ── Step 1: Scan ──
    print(f"\n  ── STEP 1: SCAN ──")
    scan_study_output(study_path)

    if plots_only:
        # Skip fix + reprocess, jump to plots
        print(f"\n  ── SKIP STEPS 2-3 (--plots-only) ──")
    else:
        # ── Step 2: Fix targets ──
        print(f"\n  ── STEP 2: FIX TARGETS ──")
        fixed = fix_targets(
            study_path,
            max_depth_diff=max_depth_diff,
            max_position_diff=max_position_diff,
            dry_run=dry_run,
            fallback_targets=fallback_targets
        )

        # ── Step 3: Reprocess pointing ──
        print(f"\n  ── STEP 3: REPROCESS POINTING ──")
        if dry_run:
            print(f"  (dry run - skipping reprocess)")
        else:
            # Use pre-built arm_overrides (from annotations) or load from CSV
            overrides = arm_overrides
            if overrides is None and arm_csv:
                overrides = load_arm_overrides(Path(arm_csv))

            if overrides:
                n_overrides = sum(1 for v in overrides.values()
                                  if v.get('override_arm') in ('left', 'right'))
                print(f"  Arm overrides: {n_overrides} trial/camera combos")

            reprocess_pointing(
                study_path,
                arm_overrides=overrides,
                dog_depth_max=dog_depth_max,
                subject_type=subject_type,
                camera_filter=camera_filter,
                raw_data_path=raw_data_path
            )

    # ── Step 4: Regenerate visualizations ──
    print(f"\n  ── STEP 4: REGENERATE VISUALIZATIONS ──")
    if dry_run:
        print(f"  (dry run - skipping visualization)")
    else:
        x_range = (-1.5, 1.5)
        z_range = (z_min, z_max)

        camera_outputs = find_camera_outputs(study_path)
        if camera_filter:
            camera_outputs = [c for c in camera_outputs if c.name == camera_filter]
        print(f"  Found {len(camera_outputs)} camera output folder(s)")

        ok = 0
        err = 0
        for cam_output in camera_outputs:
            try:
                results = process_camera_output(
                    cam_output,
                    x_range=x_range,
                    z_range=z_range,
                    plot_types=['all'],
                    subject_type=subject_type
                )
                ok += sum(1 for s in results.values() if s == 'OK')
                err += sum(1 for s in results.values() if s.startswith('ERROR'))
            except Exception as e:
                print(f"    ERROR on {cam_output}: {e}")
                err += 1

        print(f"\n  Visualizations: OK={ok} | ERROR={err}")

    print(f"\n  STUDY DONE: {study_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run full post-processing + visualization on all studies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all studies (cam1 only, with dog depth filter)
  python run_all_postprocess.py pointing_data/output/ --camera cam1 --dog-depth-max 3.0

  # With annotations CSV for ground truth arm overrides
  python run_all_postprocess.py pointing_data/output/ --camera cam1 --dog-depth-max 3.0 \\
      --annotations-csv "pointing_data/PVPO_Production_Data - Output_Tracker.csv"

  # With raw data root for re-detecting bad human frames
  python run_all_postprocess.py pointing_data/output/ --camera cam1 --dog-depth-max 3.0 \\
      --annotations-csv annotations.csv --raw-data-root pointing_data/

  # Preview changes
  python run_all_postprocess.py pointing_data/output/ --dry-run

  # Only regenerate plots (skip reprocessing)
  python run_all_postprocess.py pointing_data/output/ --plots-only

  # Custom Z range for plots
  python run_all_postprocess.py pointing_data/output/ --z-min 2.0 --z-max 5.0
"""
    )

    parser.add_argument('path', type=str,
                        help='Path to output folder (single study or parent of multiple studies)')
    parser.add_argument('--dog-depth-max', type=float, default=None,
                        help='Only include frames where dog depth < this (meters)')
    parser.add_argument('--arm-csv', type=str, default=None,
                        help='Path to per-study arm override CSV')
    parser.add_argument('--annotations-csv', type=str, default=None,
                        help='Path to annotated CSV with ground truth arm info. '
                             'Auto-generates arm overrides per study from "Owner Arm Used" column.')
    parser.add_argument('--max-depth-diff', type=float, default=2.0,
                        help='Max target depth spread for reference (default: 2.0m)')
    parser.add_argument('--max-position-diff', type=float, default=1.0,
                        help='Max position diff before replacing targets (default: 1.0m)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--plots-only', action='store_true',
                        help='Skip reprocessing, only regenerate visualizations')
    parser.add_argument('--z-min', type=float, default=2.0,
                        help='Min Z axis for plots (default: 2.0)')
    parser.add_argument('--z-max', type=float, default=5.0,
                        help='Max Z axis for plots (default: 5.0)')
    parser.add_argument('--camera', type=str, default=None,
                        help='Only reprocess + visualize this camera (e.g. cam1)')
    parser.add_argument('--raw-data', type=str, default=None,
                        help='Path to raw data for a single study (trial_*/cam*/color/ structure).')
    parser.add_argument('--raw-data-root', type=str, default=None,
                        help='Root path containing raw data folders for ALL studies. '
                             'Auto-maps output folder names (strips _output suffix).')
    parser.add_argument('--subject', type=str, default='dog',
                        choices=['dog', 'baby'],
                        help='Subject type (default: dog)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only run arm alignment verification (no reprocessing)')

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"ERROR: Path does not exist: {base_path}")
        sys.exit(1)

    studies = find_studies(base_path)
    if not studies:
        print(f"ERROR: No study folders found in {base_path}")
        sys.exit(1)

    # Load annotations if provided
    annotations = None
    if args.annotations_csv:
        annotations_path = Path(args.annotations_csv)
        if not annotations_path.exists():
            print(f"ERROR: Annotations CSV does not exist: {annotations_path}")
            sys.exit(1)
        annotations = load_annotations(annotations_path)
        print(f"  Loaded annotations for {len(annotations)} subjects")

    raw_data_root = Path(args.raw_data_root) if args.raw_data_root else None

    print(f"\n{'='*70}")
    print(f"  BATCH POST-PROCESSING")
    print(f"  Studies found: {len(studies)}")
    for s in studies:
        print(f"    - {s.name}")
    if args.camera:
        print(f"  Camera filter: {args.camera}")
    if raw_data_root:
        print(f"  Raw data root: {raw_data_root}")
    elif args.raw_data:
        print(f"  Raw data: {args.raw_data}")
    if args.dog_depth_max:
        print(f"  Dog depth filter: < {args.dog_depth_max}m")
    if args.annotations_csv:
        print(f"  Annotations CSV: {args.annotations_csv}")
    if args.arm_csv:
        print(f"  Arm CSV: {args.arm_csv}")
    if args.dry_run:
        print(f"  MODE: DRY RUN")
    if args.plots_only:
        print(f"  MODE: PLOTS ONLY")
    if args.verify_only:
        print(f"  MODE: VERIFY ONLY")
    print(f"  Plot Z range: {args.z_min} - {args.z_max}")
    print(f"{'='*70}")

    # Fast path: verify-only mode
    if args.verify_only:
        if annotations:
            camera_for_verify = args.camera or 'cam1'
            verify_arm_alignment(studies, annotations, camera=camera_for_verify)
        else:
            print("ERROR: --verify-only requires --annotations-csv")
            sys.exit(1)
        return

    start = time.time()

    # Step 0: Build global reference targets (cross-study fallback)
    print(f"\n{'='*70}")
    print(f"  STEP 0: BUILDING GLOBAL REFERENCE TARGETS")
    print(f"{'='*70}")
    global_fallback = build_global_reference_targets(
        studies, camera_filter=args.camera, max_depth_diff=args.max_depth_diff
    )
    for cam, targets in global_fallback.items():
        print(f"  {cam}: global reference from {len(studies)} studies")
        for t in targets:
            print(f"    {t['label']}: x={t['x']:+.3f} y={t['y']:+.3f} z={t['z']:.3f}m")

    # Process each study
    for study_path in studies:
        try:
            # Build per-study arm overrides from annotations
            study_arm_overrides = None
            if annotations:
                study_arm_overrides = build_arm_overrides_from_annotations(
                    annotations, study_path.name, camera=args.camera,
                    study_path=study_path
                )
                if study_arm_overrides:
                    print(f"\n  Annotations: {len(study_arm_overrides)} arm overrides for {study_path.name}")

            # Find raw data path for this study
            study_raw_data = None
            if raw_data_root:
                study_raw_data = find_raw_data_path(study_path, raw_data_root)
                if study_raw_data:
                    print(f"  Raw data: {study_raw_data}")
                else:
                    print(f"  Raw data: NOT FOUND for {study_path.name}")
            elif args.raw_data:
                study_raw_data = Path(args.raw_data)

            run_pipeline(
                study_path,
                dog_depth_max=args.dog_depth_max,
                arm_csv=args.arm_csv,
                arm_overrides=study_arm_overrides,
                max_depth_diff=args.max_depth_diff,
                max_position_diff=args.max_position_diff,
                dry_run=args.dry_run,
                plots_only=args.plots_only,
                z_min=args.z_min,
                z_max=args.z_max,
                subject_type=args.subject,
                camera_filter=args.camera,
                raw_data_path=study_raw_data,
                fallback_targets=global_fallback
            )
        except Exception as e:
            print(f"\n  STUDY ERROR: {study_path.name}: {e}")
            import traceback
            traceback.print_exc()

    # Final step: Verify arm alignment against annotations
    if annotations and not args.dry_run:
        camera_for_verify = args.camera or 'cam1'
        mismatches = verify_arm_alignment(studies, annotations, camera=camera_for_verify)
        if mismatches:
            print(f"\n  WARNING: {len(mismatches)} arm mismatches found!")
            print(f"  These trials have annotations overriding detected arm.")
            print(f"  Check if arm detection was correct or if annotation was wrong.")

    elapsed = time.time() - start
    print(f"\n{'='*70}")
    print(f"  ALL DONE ({len(studies)} studies in {elapsed:.1f}s)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

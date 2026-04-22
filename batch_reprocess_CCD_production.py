#!/usr/bin/env python3
"""
Batch reprocess CCD Point Production data.

Detects the adult experimenter's pointing gesture using MediaPipe skeleton
extraction + arm vector analysis. The human pointer is an adult standing
in the front camera view, in the upper half of the image, behind all targets.

Full pipeline per trial:
  1. Extract from zip (if needed)
  2. YOLO target detection (with fixed geometry correction)
  3. MediaPipe skeleton extraction (for the adult pointer)
  4. Pointing analysis (arm vectors → ground intersection → target distances)
  5. CSV + plot export
  6. Copy results to output directory
  7. Cleanup extracted data

Usage:
    # Process all subjects
    python batch_reprocess_CCD_production.py

    # Process single subject
    python batch_reprocess_CCD_production.py --subject CCD0346

    # Skip target detection (reuse existing)
    python batch_reprocess_CCD_production.py --skip-targets
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

# Ensure matplotlib uses non-interactive backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project paths ───
POINTING_DIR = Path("/home/tigerli/Documents/GitHub/pointing")
SSD_SOURCE = Path("/media/tigerli/Extreme SSD/pointing_data/point_production_CCD")
OUTPUT_DIR = Path("/home/tigerli/Documents/pointing_data/point_production_CCD_output")
WORK_DIR = Path("/home/tigerli/Documents/pointing_data/_ccd_prod_work")

# Add project paths for imports
sys.path.insert(0, str(POINTING_DIR))
sys.path.insert(0, str(POINTING_DIR / "step3_subject_extraction" / "dog_script"))
sys.path.insert(0, str(POINTING_DIR / "step0_data_loading"))

YOLO_MODEL = POINTING_DIR / "step0_data_loading" / "best.pt"

# ─── Fixed fallback target positions for CCD production ───
# Same targets as comprehension - 4 cups on a curved arc.
# These are from the experimenter's (front camera) perspective.
CCD_TARGETS_FIXED = [
    {"label": "target_1", "x":  0.4430, "y": 0.6570, "z": 1.6520},
    {"label": "target_2", "x":  0.1560, "y": 0.5440, "z": 2.3530},
    {"label": "target_3", "x": -0.4620, "y": 0.4370, "z": 2.8180},
    {"label": "target_4", "x": -1.1500, "y": 0.3520, "z": 3.0760},
]

# ─── Relative offsets between targets (from CCD_TARGETS_FIXED) ───
_CCD_CENTROID = {
    "x": np.mean([t["x"] for t in CCD_TARGETS_FIXED]),
    "y": np.mean([t["y"] for t in CCD_TARGETS_FIXED]),
    "z": np.mean([t["z"] for t in CCD_TARGETS_FIXED]),
}
_CCD_OFFSETS = [
    {
        "label": t["label"],
        "dx": t["x"] - _CCD_CENTROID["x"],
        "dy": t["y"] - _CCD_CENTROID["y"],
        "dz": t["z"] - _CCD_CENTROID["z"],
    }
    for t in CCD_TARGETS_FIXED
]


def _correct_targets_with_fixed_geometry(raw_targets: list) -> list:
    """
    Correct YOLO-detected target positions using the known fixed relative
    geometry of the 4 CCD targets.
    """
    if not raw_targets:
        return raw_targets

    anchors = [t for t in raw_targets
               if 0.5 < t["z"] < 5.0 and not np.isnan(t["z"])]

    if not anchors:
        print("    Target correction: no anchors with valid depth, using fixed targets")
        return CCD_TARGETS_FIXED

    shifts = []
    for anchor in anchors:
        label_idx = int(anchor["label"].split("_")[1]) - 1
        if 0 <= label_idx < 4:
            fixed = CCD_TARGETS_FIXED[label_idx]
            shifts.append({
                "dx": anchor["x"] - fixed["x"],
                "dy": anchor["y"] - fixed["y"],
                "dz": anchor["z"] - fixed["z"],
            })

    if not shifts:
        return raw_targets

    avg_shift = {
        "dx": np.mean([s["dx"] for s in shifts]),
        "dy": np.mean([s["dy"] for s in shifts]),
        "dz": np.mean([s["dz"] for s in shifts]),
    }

    corrected = []
    for t in CCD_TARGETS_FIXED:
        corrected.append({
            "label": t["label"],
            "x": t["x"] + avg_shift["dx"],
            "y": t["y"] + avg_shift["dy"],
            "z": t["z"] + avg_shift["dz"],
        })

    print(f"    Target correction: {len(anchors)} anchor(s), "
          f"shift=({avg_shift['dx']:+.3f}, {avg_shift['dy']:+.3f}, {avg_shift['dz']:+.3f})")

    return corrected


# ─────────────────────────────────────────────────────────────
#  Trial layout detection
# ─────────────────────────────────────────────────────────────
def _resolve_trial_layout(trial_dir: Path) -> dict:
    """
    Detect trial data layout and return normalized paths.

    Supports two layouts:
      Old: trial_dir/Color/ + trial_dir/Depth/ (.raw files)
      New: trial_dir/cam1/color/ + trial_dir/cam1/depth/ (.npy files)
    """
    # Old layout
    color_dir = trial_dir / "Color"
    depth_dir = trial_dir / "Depth"
    if color_dir.is_dir() and depth_dir.is_dir():
        raw_files = [f for f in depth_dir.iterdir()
                     if f.suffix == ".raw" and not f.name.startswith("._")]
        if raw_files:
            return {"color_dir": color_dir, "depth_dir": depth_dir,
                    "depth_format": "raw", "color_prefix": "Color_",
                    "depth_prefix": "Depth_"}

    # New layout
    cam1_dir = trial_dir / "cam1"
    if cam1_dir.is_dir():
        color_dir = cam1_dir / "color"
        depth_dir = cam1_dir / "depth"
        if color_dir.is_dir() and depth_dir.is_dir():
            npy_files = [f for f in depth_dir.iterdir()
                         if f.suffix == ".npy" and not f.name.startswith("._")]
            if npy_files:
                return {"color_dir": color_dir, "depth_dir": depth_dir,
                        "depth_format": "npy", "color_prefix": "frame_",
                        "depth_prefix": "frame_"}

    return None


def find_trial_dirs(data_dir: Path) -> list:
    """Find trial directories with color+depth frames."""
    trials = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        is_trial = (d.name.isdigit() or
                    (d.name.startswith("trial_") and d.name[6:].isdigit()))
        if is_trial and _resolve_trial_layout(d) is not None:
            trials.append(d)
    return trials


def find_data_dir(subject_src: Path) -> tuple:
    """Find the data directory for a subject. Returns (zip_path, data_dir)."""
    zip_path = None
    data_dir = None

    zips = [z for z in subject_src.glob("*.zip") if not z.name.startswith("._")]

    # Check for extracted data (inner folder with numbered trial dirs)
    for d in sorted(subject_src.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            if any(sd.name.isdigit() for sd in d.iterdir() if sd.is_dir()):
                data_dir = d
                break

    if data_dir is None and zips:
        zip_path = zips[0]
    elif data_dir is None:
        has_trials = any(
            d.is_dir() and (d.name.isdigit() or
                            (d.name.startswith("trial_") and d.name[6:].isdigit()))
            for d in subject_src.iterdir()
        )
        if has_trials:
            data_dir = subject_src

    return zip_path, data_dir


# ─────────────────────────────────────────────────────────────
#  Intrinsics
# ─────────────────────────────────────────────────────────────
def load_intrinsics(data_dir: Path):
    """Load camera intrinsics, scaling to match actual image resolution."""
    nominal = None
    meta_path = data_dir / "rosbag_metadata.yaml"
    if not meta_path.exists():
        meta_path = data_dir.parent / "rosbag_metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        intr = meta["intrinsics"]
        nominal = {
            "fx": intr["fx"], "fy": intr["fy"],
            "cx": intr["ppx"], "cy": intr["ppy"],
            "width": intr["width"], "height": intr["height"],
        }

    # Detect actual image resolution
    trials = find_trial_dirs(data_dir)
    actual_w, actual_h = None, None
    if trials:
        layout = _resolve_trial_layout(trials[0])
        if layout:
            color_files = sorted([f for f in layout["color_dir"].iterdir()
                                  if f.suffix == ".png" and not f.name.startswith("._")])
            if color_files:
                img = cv2.imread(str(color_files[0]))
                if img is not None:
                    actual_h, actual_w = img.shape[:2]

    if nominal and actual_w and actual_h:
        nom_w, nom_h = nominal["width"], nominal["height"]
        if nom_w != actual_w or nom_h != actual_h:
            sx = actual_w / nom_w
            sy = actual_h / nom_h
            print(f"  Scaling intrinsics: {nom_w}x{nom_h} → {actual_w}x{actual_h}")
            return {
                "fx": nominal["fx"] * sx, "fy": nominal["fy"] * sy,
                "cx": nominal["cx"] * sx, "cy": nominal["cy"] * sy,
                "width": actual_w, "height": actual_h,
            }
        return nominal
    elif nominal:
        return nominal

    # Fallback: default CCD front camera intrinsics (640x480 common)
    if actual_w == 640 and actual_h == 480:
        return {"fx": 318.2, "fy": 424.3, "cx": 319.4, "cy": 238.6,
                "width": 640, "height": 480}

    return {"fx": 636.42, "fy": 636.42, "cx": 638.78, "cy": 357.97,
            "width": 1280, "height": 720}


# ─────────────────────────────────────────────────────────────
#  YOLO Target detection
# ─────────────────────────────────────────────────────────────
def detect_targets_yolo(trial_dir: Path, intrinsics: dict) -> list:
    """Run YOLO target detection on a trial directory."""
    from target_detector import TargetDetector

    layout = _resolve_trial_layout(trial_dir)
    if layout is None:
        return []
    color_dir = layout["color_dir"]
    depth_dir = layout["depth_dir"]
    depth_format = layout["depth_format"]

    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    if not color_files:
        return []

    # Try multiple frames for detection
    for frame_idx in [len(color_files) // 2, 0, len(color_files) - 1]:
        color_path = color_files[frame_idx]
        color_img = cv2.imread(str(color_path))
        if color_img is None:
            continue

        h, w = color_img.shape[:2]
        depth_img = _load_depth_for_color(color_path, depth_dir, depth_format, w, h)

        detector = TargetDetector(model_path=str(YOLO_MODEL), confidence_threshold=0.3)
        fx, fy = intrinsics["fx"], intrinsics["fy"]
        cx, cy = intrinsics["cx"], intrinsics["cy"]

        detections = detector.detect(color_img, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)
        if not detections:
            detector.confidence_threshold = 0.15
            detections = detector.detect(color_img, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)

        if detections:
            raw_targets = []
            for i, det in enumerate(sorted(detections, key=lambda d: d.center[0]), 1):
                if det.center_3d is not None:
                    raw_targets.append({
                        "label": f"target_{i}",
                        "x": float(det.center_3d[0]),
                        "y": float(det.center_3d[1]),
                        "z": float(det.center_3d[2]),
                    })
                elif det.depth is not None:
                    cx_px, cy_px = det.center
                    x = (cx_px - cx) * det.depth / fx
                    y = (cy_px - cy) * det.depth / fy
                    raw_targets.append({
                        "label": f"target_{i}",
                        "x": float(x), "y": float(y), "z": float(det.depth),
                    })

            if raw_targets:
                print(f"    YOLO: Detected {len(raw_targets)} raw targets (frame {color_path.name})")
                for t in raw_targets:
                    print(f"      {t['label']}: x={t['x']:.3f}, y={t['y']:.3f}, z={t['z']:.3f}")
                targets = _correct_targets_with_fixed_geometry(raw_targets)
                if targets is not raw_targets:
                    print(f"    YOLO: Corrected targets:")
                    for t in targets:
                        print(f"      {t['label']}: x={t['x']:.3f}, y={t['y']:.3f}, z={t['z']:.3f}")
                return targets

    print(f"    YOLO: No targets detected in any frame")
    return []


def _load_depth_for_color(color_path: Path, depth_dir: Path,
                          depth_format: str, w: int, h: int):
    """Load depth frame matching a color frame."""
    if depth_format == "npy":
        depth_name = color_path.stem + ".npy"
        if color_path.stem.startswith("Color_"):
            frame_num = color_path.stem.replace("Color_", "")
            depth_name = f"frame_{frame_num}.npy"
        depth_path = depth_dir / depth_name
        if depth_path.exists():
            raw = np.load(str(depth_path))
            depth_img = raw.astype(np.float32) / 1000.0
            if depth_img.shape[:2] != (h, w):
                depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
            return depth_img
    else:
        depth_name = color_path.stem.replace("Color", "Depth") + ".raw"
        depth_path = depth_dir / depth_name
        if depth_path.exists():
            raw_size = depth_path.stat().st_size
            raw_pixels = raw_size // 2
            if raw_pixels == w * h:
                dw, dh = w, h
            elif raw_pixels == 1280 * 720:
                dw, dh = 1280, 720
            elif raw_pixels == 640 * 480:
                dw, dh = 640, 480
            else:
                return None

            with open(depth_path, "rb") as f:
                raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((dh, dw))
            if (dw, dh) != (w, h):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            return raw.astype(np.float32) / 1000.0
    return None


def save_target_coordinates_json(trial_dir: Path, targets: list):
    """Save targets as target_coordinates.json."""
    target_data = {
        "targets": [
            {
                "id": i + 1,
                "label": t["label"],
                "world_coords": [t["x"], t["y"], t["z"]],
                "depth_m": t["z"],
            }
            for i, t in enumerate(targets)
        ],
        "labeling_metadata": {
            "source": "YOLO_detection" if len(targets) > 0 else "fixed_CCD_TARGETS",
        },
    }
    out_path = trial_dir / "target_coordinates.json"
    with open(out_path, "w") as f:
        json.dump(target_data, f, indent=2)
    return out_path


# ─────────────────────────────────────────────────────────────
#  Zip extraction
# ─────────────────────────────────────────────────────────────
def extract_subject_from_zip(zip_path: Path, dest: Path):
    """Extract zip to destination, finding the inner data directory."""
    print(f"  Extracting {zip_path.name}...")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        total = sum(info.file_size for info in zf.infolist())
        zf.extractall(dest)

    print(f"  Extracted {total / 1e9:.1f} GB")
    return dest


# ─────────────────────────────────────────────────────────────
#  MediaPipe skeleton extraction + pointing analysis
# ─────────────────────────────────────────────────────────────
def process_pointing_for_trial(trial_dir: Path, intrinsics: dict,
                               targets: list, output_trial_dir: Path):
    """
    Run MediaPipe skeleton extraction and pointing analysis for one trial.

    The human pointer is an ADULT standing in the upper portion of the frame.
    We use crop_top_ratio to focus on the upper part of the image where the
    pointer is standing (excludes babies/handlers in the lower portion).

    Returns True if successful, False otherwise.
    """
    from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
    from step2_skeleton_extraction.batch_processor import determine_pointing_hand_whole_trial
    from step2_skeleton_extraction.pointing_analysis import analyze_pointing_frame
    from step2_skeleton_extraction.csv_exporter import export_pointing_analysis_to_csv
    from step2_skeleton_extraction.kalman_filter import (
        LandmarkKalmanFilter, smooth_pointing_analyses
    )

    layout = _resolve_trial_layout(trial_dir)
    if layout is None:
        print(f"    Cannot resolve trial layout")
        return False

    color_dir = layout["color_dir"]
    depth_dir = layout["depth_dir"]
    depth_format = layout["depth_format"]

    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    if not color_files:
        print(f"    No color files found")
        return False

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    w, h = intrinsics["width"], intrinsics["height"]

    # Initialize MediaPipe detector
    # Disable segmentation — it crashes with SegmentationSmoothingCalculator.
    # We only need skeleton landmarks for pointing analysis, not segmentation masks.
    detector = MediaPipeHumanDetector(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1,
        enable_segmentation=False,
    )

    # Initialize Kalman filter for landmark smoothing
    landmark_filter = LandmarkKalmanFilter(
        num_landmarks=33,
        process_noise=0.005,
        measurement_noise=0.05
    )

    human_results = {}

    for i, color_path in enumerate(color_files, 1):
        # Extract frame number from filename
        stem = color_path.stem
        if stem.startswith("Color_"):
            frame_num = int(stem.replace("Color_", ""))
        elif stem.startswith("frame_"):
            frame_num = int(stem.replace("frame_", ""))
        else:
            frame_num = i

        color_img = cv2.imread(str(color_path))
        if color_img is None:
            continue
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Load depth
        depth_img = _load_depth_for_color(color_path, depth_dir, depth_format,
                                          color_img.shape[1], color_img.shape[0])

        frame_key = f"frame_{frame_num:06d}"
        result = detector.detect_frame(
            color_rgb, frame_num,
            depth_image=depth_img,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        if result:
            # Apply Kalman filtering
            if result.landmarks_3d:
                filtered = landmark_filter.update(result.landmarks_3d)
                result.landmarks_3d_raw = result.landmarks_3d
                result.landmarks_3d = filtered
            human_results[frame_key] = result

        if i % 100 == 0 or i == len(color_files):
            print(f"      {i}/{len(color_files)} frames...", end='\r')

    print(f"    Skeleton: {len(human_results)}/{len(color_files)} frames detected")

    if not human_results:
        print(f"    WARNING: No skeletons detected")
        return False

    # Validate: check that the detected person is likely the adult pointer
    # The pointer should be in the upper portion of the image and at a reasonable depth
    _validate_pointer_detection(human_results, targets, intrinsics)

    # Determine pointing hand (whole-trial voting)
    results_list = list(human_results.values())
    pointing_hand = determine_pointing_hand_whole_trial(results_list)
    print(f"    Pointing hand: {pointing_hand}")

    # Update all results with whole-trial pointing hand
    for result in human_results.values():
        result.metadata['pointing_hand_whole_trial'] = pointing_hand
        if pointing_hand in ['left', 'right'] and result.landmarks_3d:
            result.arm_vectors = detector._compute_arm_vectors(
                result.landmarks_3d, pointing_hand
            )
            result.metadata['pointing_arm'] = pointing_hand

    # Run pointing analysis
    analyses = {}
    for frame_key, result in human_results.items():
        if result.landmarks_3d:
            analysis = analyze_pointing_frame(
                result, targets,
                pointing_arm=result.metadata.get('pointing_arm', 'right'),
                ground_plane_rotation=None  # No ground rotation for CCD (curved arc)
            )
            if analysis:
                analyses[frame_key] = analysis

    if not analyses:
        print(f"    WARNING: No valid pointing analyses")
        return False

    # Apply Kalman smoothing to pointing trajectories
    analyses = smooth_pointing_analyses(
        analyses, process_noise=0.01, measurement_noise=0.1
    )

    # Export CSV
    output_trial_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_trial_dir / "processed_gesture.csv"
    export_pointing_analysis_to_csv(
        human_results, analyses, csv_path, global_start_frame=0
    )

    # Save target coordinates
    save_target_coordinates_json(output_trial_dir, targets)

    # Save skeleton JSON
    output_data = {fk: r.to_dict() for fk, r in human_results.items()}
    with open(output_trial_dir / "skeleton_2d.json", 'w') as f:
        json.dump(output_data, f, indent=2)

    # Generate pointing trace plot
    try:
        from step2_skeleton_extraction.plot_pointing_trace import plot_2d_pointing_trace
        # Estimate human position from average hip center
        hip_positions = []
        for result in human_results.values():
            if result.landmarks_3d and len(result.landmarks_3d) > 24:
                lh = np.array(result.landmarks_3d[23])
                rh = np.array(result.landmarks_3d[24])
                hc = (lh + rh) / 2.0
                if not np.all(hc == 0):
                    hip_positions.append(hc)
        human_pos = list(np.mean(hip_positions, axis=0)) if hip_positions else [0, 0, 2]
        plot_2d_pointing_trace(analyses, targets, human_pos,
                              output_trial_dir / "2d_pointing_trace.png")
    except Exception as e:
        print(f"    Plot warning: {e}")

    # Generate distance-to-targets plot
    try:
        from step2_skeleton_extraction.plot_distance_to_targets import plot_distance_to_targets
        plot_distance_to_targets(analyses, targets, output_trial_dir / "distance_to_targets.png")
    except Exception as e:
        print(f"    Distance plot warning: {e}")

    print(f"    Saved {len(analyses)} frames → {csv_path.name}")
    return True


def _validate_pointer_detection(human_results: dict, targets: list, intrinsics: dict):
    """
    Validate that the detected skeleton is likely the adult pointer.

    Checks:
    1. Average nose Y position should be in upper half of image (< 50% of height)
    2. If 3D data available, hip center should be behind targets (further z)
       and between the middle two targets (laterally)

    Prints warnings if validation fails.
    """
    if not human_results:
        return

    h = intrinsics["height"]
    nose_ys = []
    hip_positions = []

    for result in human_results.values():
        if result.landmarks_2d and len(result.landmarks_2d) > 0:
            nose_y = result.landmarks_2d[0][1]  # nose Y in pixels
            nose_ys.append(nose_y)

        if result.landmarks_3d and len(result.landmarks_3d) > 24:
            left_hip = np.array(result.landmarks_3d[23])
            right_hip = np.array(result.landmarks_3d[24])
            hip_center = (left_hip + right_hip) / 2.0
            if not np.all(hip_center == 0):
                hip_positions.append(hip_center)

    if nose_ys:
        avg_nose_y = np.mean(nose_ys)
        nose_ratio = avg_nose_y / h
        if nose_ratio > 0.6:
            print(f"    WARNING: Detected person's nose is at {nose_ratio:.0%} of image height "
                  f"(expected < 60% for adult pointer)")
        else:
            print(f"    Pointer validation: nose at {nose_ratio:.0%} of image height (OK)")

    if hip_positions and targets:
        avg_hip = np.mean(hip_positions, axis=0)
        target_depths = [t["z"] for t in targets]
        max_target_depth = max(target_depths)

        # For production: human should be BEHIND targets (further from camera)
        # In camera coords, larger z = further away
        # But with MediaPipe world landmarks, the relationship might differ
        # Print info for debugging
        print(f"    Pointer hip 3D: x={avg_hip[0]:.3f}, y={avg_hip[1]:.3f}, z={avg_hip[2]:.3f}")


# ─────────────────────────────────────────────────────────────
#  Per-subject processing
# ─────────────────────────────────────────────────────────────
def process_subject(subject_name: str, subject_src: Path,
                    skip_targets: bool = False):
    """Process one CCD production subject through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {subject_name}")
    print(f"{'='*60}")

    zip_path, data_dir = find_data_dir(subject_src)

    if zip_path is None and data_dir is None:
        print(f"  ERROR: No data found for {subject_name}")
        return 0, 0

    # Extract zip if needed
    work_dir = None
    if zip_path is not None and data_dir is None:
        work_dir = WORK_DIR / subject_name
        if work_dir.exists():
            shutil.rmtree(work_dir)
        extract_subject_from_zip(zip_path, work_dir)
        data_dir = work_dir
        # Handle nested zip structure: zip may contain a single subdir with trials inside
        subdirs = [d for d in work_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) == 1 and any(sd.is_dir() and sd.name.isdigit() for sd in subdirs[0].iterdir()):
            data_dir = subdirs[0]
            print(f"  Using nested data dir: {data_dir.name}")

    # Load intrinsics
    intrinsics = load_intrinsics(data_dir)
    print(f"  Intrinsics: {intrinsics['width']}x{intrinsics['height']}, "
          f"fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Find trial directories
    trials = find_trial_dirs(data_dir)
    if not trials:
        print(f"  ERROR: No trial directories found in {data_dir}")
        if work_dir:
            shutil.rmtree(work_dir)
        return 0, 0

    print(f"  Found {len(trials)} trials")

    # Detect targets once per subject
    subject_targets = None
    if not skip_targets:
        print(f"\n  --- Target Detection ---")
        for trial_dir in trials:
            subject_targets = detect_targets_yolo(trial_dir, intrinsics)
            if subject_targets and len(subject_targets) >= 2:
                print(f"  Using YOLO-detected targets from trial {trial_dir.name}")
                break
            subject_targets = None

    if subject_targets is None or len(subject_targets) == 0:
        print(f"  Using fixed CCD_TARGETS")
        subject_targets = CCD_TARGETS_FIXED

    # Output directory
    output_subject_dir = OUTPUT_DIR / subject_name

    ok = 0
    fail = 0

    for trial_dir in trials:
        trial_name = trial_dir.name
        layout = _resolve_trial_layout(trial_dir)

        # Normalize output dir name
        if trial_name.startswith("trial_"):
            output_trial_name = trial_name
        else:
            output_trial_name = f"trial_{trial_name}"

        output_trial_dir = output_subject_dir / output_trial_name

        print(f"\n  --- Trial {trial_name} ---"
              + (f" [{layout['depth_format']}]" if layout else ""))

        try:
            success = process_pointing_for_trial(
                trial_dir, intrinsics, subject_targets, output_trial_dir)

            if success and (output_trial_dir / "processed_gesture.csv").exists():
                ok += 1
            else:
                print(f"    WARNING: No CSV generated")
                fail += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            fail += 1

    # Generate combined CSVs for this subject
    if ok > 0:
        generate_combined_csv(output_subject_dir, subject_name)

    # Cleanup
    if work_dir is not None and work_dir.exists():
        print(f"\n  Cleaning up {work_dir}...")
        shutil.rmtree(work_dir)

    print(f"\n  {subject_name}: {ok} OK, {fail} failed / {len(trials)} trials")
    return ok, fail


def generate_combined_csv(subject_dir: Path, subject_name: str):
    """Generate combined CSV for a subject from all trial CSVs."""
    import pandas as pd

    all_dfs = []
    for trial_dir in sorted(subject_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        csv_path = trial_dir / "processed_gesture.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                df.insert(0, "subject", subject_name)
                df.insert(1, "trial", trial_dir.name)
                all_dfs.append(df)
            except Exception as e:
                print(f"    Warning: Error reading {csv_path}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(subject_dir / "combined_gesture.csv", index=False)
        print(f"  Combined gesture CSV: {len(combined)} rows, {len(all_dfs)} trials")


def generate_global_csv(output_dir: Path):
    """Generate global CSV from all subjects."""
    import pandas as pd

    all_dfs = []
    for subject_dir in sorted(output_dir.iterdir()):
        if not subject_dir.is_dir():
            continue
        combined = subject_dir / "combined_gesture.csv"
        if combined.exists():
            try:
                df = pd.read_csv(combined)
                all_dfs.append(df)
            except Exception:
                pass

    if all_dfs:
        global_df = pd.concat(all_dfs, ignore_index=True)
        global_path = output_dir / "all_subjects_gesture.csv"
        global_df.to_csv(global_path, index=False)
        print(f"Global gesture CSV: {len(global_df)} rows from {len(all_dfs)} subjects")
        return global_path
    else:
        print("No data found for global CSV")
        return None


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Batch reprocess CCD Point Production data")
    parser.add_argument("--subject", help="Process only this subject (prefix match)")
    parser.add_argument("--skip-targets", action="store_true",
                        help="Skip YOLO target detection, use fixed targets")
    parser.add_argument("--source", type=str, default=str(SSD_SOURCE),
                        help=f"Source directory (default: {SSD_SOURCE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    source_dir = Path(args.source)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Find all subjects
    subjects = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if args.subject:
        subjects = [d for d in subjects if d.name.startswith(args.subject)]

    print(f"Source: {source_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Subjects: {len(subjects)}")
    print(f"YOLO model: {YOLO_MODEL}")
    print()

    total_ok = 0
    total_fail = 0

    for i, subject_dir in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject_dir.name}")
        ok, fail = process_subject(
            subject_dir.name,
            subject_dir,
            skip_targets=args.skip_targets,
        )
        total_ok += ok
        total_fail += fail

    # Generate global CSVs
    if total_ok > 0:
        print(f"\n{'='*60}")
        print("Generating global CSV...")
        generate_global_csv(OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"DONE: {total_ok} trials OK, {total_fail} failed")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

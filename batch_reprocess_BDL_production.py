#!/usr/bin/env python3
"""
Batch reprocess BDL Point Production data.

Same experiment as CCD production: detects the adult experimenter's pointing
gesture using MediaPipe skeleton extraction + arm vector analysis.

BDL data format differs from CCD:
  - Input: flat folder of zip files (each zip IS one subject)
  - Zip structure: double-nested NAME/NAME/1/, NAME/NAME/2/, ...
  - Color: _Color_NNNN.png in Color/ directory (ignore fNNN.png duplicates)
  - Depth: _Depth_Color_NNNN.raw (uint16, 1280x720) in Depth_Color/ directory
  - No rosbag_metadata.yaml — uses fallback 1280x720 intrinsics

Usage:
    # Process all subjects
    python batch_reprocess_BDL_production.py

    # Process single subject
    python batch_reprocess_BDL_production.py --subject BDL049

    # Skip target detection (reuse existing)
    python batch_reprocess_BDL_production.py --skip-targets
"""

import argparse
import json
import os
import shutil
import sys
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np

# Ensure matplotlib uses non-interactive backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Project paths ───
POINTING_DIR = Path("/home/tigerli/Documents/GitHub/pointing")
SSD_SOURCE = Path("/media/tigerli/Extreme SSD/pointing_data/pointing_production_BDL")
OUTPUT_DIR = Path("/home/tigerli/Documents/pointing_data/point_production_BDL_output")
WORK_DIR = Path("/home/tigerli/Documents/pointing_data/_bdl_prod_work")

# Add project paths for imports
sys.path.insert(0, str(POINTING_DIR))
sys.path.insert(0, str(POINTING_DIR / "step3_subject_extraction" / "dog_script"))
sys.path.insert(0, str(POINTING_DIR / "step0_data_loading"))

YOLO_MODEL = POINTING_DIR / "step0_data_loading" / "best.pt"

# ─── Fixed fallback target positions (same as CCD) ───
CCD_TARGETS_FIXED = [
    {"label": "target_1", "x":  0.4430, "y": 0.6570, "z": 1.6520},
    {"label": "target_2", "x":  0.1560, "y": 0.5440, "z": 2.3530},
    {"label": "target_3", "x": -0.4620, "y": 0.4370, "z": 2.8180},
    {"label": "target_4", "x": -1.1500, "y": 0.3520, "z": 3.0760},
]

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
    """Correct YOLO-detected target positions using known fixed relative geometry."""
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
#  Trial layout detection (BDL-specific)
# ─────────────────────────────────────────────────────────────
def _resolve_trial_layout(trial_dir: Path) -> dict:
    """
    Detect BDL trial data layout.

    BDL layout:
      trial_dir/Color/ with _Color_NNNN.png files
      trial_dir/Depth_Color/ with _Depth_Color_NNNN.raw files
    """
    color_dir = trial_dir / "Color"
    depth_color_dir = trial_dir / "Depth_Color"

    if color_dir.is_dir() and depth_color_dir.is_dir():
        # Check for _Color_ prefixed files (original frames with matching depth)
        color_files = [f for f in color_dir.iterdir()
                       if f.suffix == ".png" and f.name.startswith("_Color_")
                       and not f.name.startswith("._")]
        raw_files = [f for f in depth_color_dir.iterdir()
                     if f.suffix == ".raw" and not f.name.startswith("._")]

        if color_files and raw_files:
            return {
                "color_dir": color_dir,
                "depth_dir": depth_color_dir,
                "depth_format": "raw_bdl",
                "color_prefix": "_Color_",
                "depth_prefix": "_Depth_Color_",
            }

        # Fallback: fNNN.png files only (no _Color_ prefix)
        f_files = [f for f in color_dir.iterdir()
                   if f.suffix == ".png" and f.name.startswith("f")
                   and not f.name.startswith("._")]
        if f_files and raw_files:
            return {
                "color_dir": color_dir,
                "depth_dir": depth_color_dir,
                "depth_format": "raw_bdl_fprefix",
                "color_prefix": "f",
                "depth_prefix": "_Depth_Color_",
            }

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


# ─────────────────────────────────────────────────────────────
#  Intrinsics
# ─────────────────────────────────────────────────────────────
def load_intrinsics(data_dir: Path):
    """Load camera intrinsics for BDL data (1280x720 fallback)."""
    import yaml

    # Try rosbag_metadata.yaml (unlikely for BDL but check)
    nominal = None
    for meta_path in [data_dir / "rosbag_metadata.yaml",
                      data_dir.parent / "rosbag_metadata.yaml"]:
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            intr = meta["intrinsics"]
            nominal = {
                "fx": intr["fx"], "fy": intr["fy"],
                "cx": intr["ppx"], "cy": intr["ppy"],
                "width": intr["width"], "height": intr["height"],
            }
            break

    # Detect actual image resolution
    trials = find_trial_dirs(data_dir)
    actual_w, actual_h = None, None
    if trials:
        layout = _resolve_trial_layout(trials[0])
        if layout:
            color_files = sorted([f for f in layout["color_dir"].iterdir()
                                  if f.suffix == ".png"
                                  and f.name.startswith(layout["color_prefix"])
                                  and not f.name.startswith("._")])
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

    # Fallback: default RealSense D435 intrinsics for 1280x720
    return {"fx": 636.42, "fy": 636.42, "cx": 638.78, "cy": 357.97,
            "width": actual_w or 1280, "height": actual_h or 720}


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
    color_prefix = layout["color_prefix"]

    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png"
                          and f.name.startswith(color_prefix)
                          and not f.name.startswith("._")])
    if not color_files:
        return []

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
    """Load depth frame matching a color frame (BDL format)."""
    if depth_format in ("raw_bdl", "raw_bdl_fprefix"):
        # Map color filename to depth filename
        stem = color_path.stem
        if stem.startswith("_Color_"):
            # _Color_NNNN → _Depth_Color_NNNN
            # Handle suffixes like _Color_2771(1) → frame_num = 2771
            frame_num = stem.replace("_Color_", "")
            if "(" in frame_num:
                frame_num = frame_num[:frame_num.index("(")]
            depth_name = f"_Depth_Color_{frame_num}.raw"
        elif stem.startswith("f"):
            # fNNN files don't have matching depth by name
            # Try to find depth by building a mapping from the sorted lists
            # This is handled in the main processing loop
            return None
        else:
            return None

        depth_path = depth_dir / depth_name
        if depth_path.exists():
            raw_size = depth_path.stat().st_size
            raw_pixels = raw_size // 2
            if raw_pixels == 1280 * 720:
                dw, dh = 1280, 720
            elif raw_pixels == w * h:
                dw, dh = w, h
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
#  Zip extraction (BDL: flat folder of zips)
# ─────────────────────────────────────────────────────────────
def extract_subject_from_zip(zip_path: Path, dest: Path):
    """Extract zip to destination."""
    print(f"  Extracting {zip_path.name}...")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        # Filter out macOS resource fork files
        members = [m for m in zf.infolist()
                   if not m.filename.startswith("__MACOSX")
                   and "/._" not in m.filename
                   and not m.filename.startswith("._")]
        total = sum(m.file_size for m in members)
        for m in members:
            zf.extract(m, dest)

    print(f"  Extracted {total / 1e9:.1f} GB")
    return dest


def find_data_dir_in_extracted(work_dir: Path) -> Path:
    """Find the actual data directory after extraction (handle double nesting)."""
    # BDL: NAME/NAME/1/, NAME/NAME/2/, ...
    # After extracting to work_dir: work_dir/NAME/NAME/1/...
    data_dir = work_dir

    # Look for numbered trial directories
    for depth in range(4):
        has_trials = any(d.is_dir() and d.name.isdigit()
                        for d in data_dir.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        if has_trials:
            return data_dir

        # Go one level deeper
        subdirs = [d for d in data_dir.iterdir()
                   if d.is_dir() and not d.name.startswith(".")]
        if len(subdirs) == 1:
            data_dir = subdirs[0]
        elif subdirs:
            # Multiple subdirs - check each for trial dirs
            for sd in subdirs:
                if any(d.is_dir() and d.name.isdigit()
                       for d in sd.iterdir() if d.is_dir()):
                    return sd
            break
        else:
            break

    return data_dir


# ─────────────────────────────────────────────────────────────
#  MediaPipe skeleton extraction + pointing analysis
# ─────────────────────────────────────────────────────────────
def process_pointing_for_trial(trial_dir: Path, intrinsics: dict,
                               targets: list, output_trial_dir: Path,
                               skip_depth: bool = False):
    """
    Run MediaPipe skeleton extraction and pointing analysis for one trial.
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
    color_prefix = layout["color_prefix"]

    # Use only _Color_ prefixed files (have matching depth)
    # Filter out duplicate files with (N) suffix (e.g., _Color_2771(1).png)
    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png"
                          and f.name.startswith(color_prefix)
                          and "(" not in f.name
                          and not f.name.startswith("._")])
    if not color_files:
        print(f"    No color files found")
        return False

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    w, h = intrinsics["width"], intrinsics["height"]

    # Initialize MediaPipe detector
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
        # Handle suffixes like _Color_2771(1).png → 2771
        stem = color_path.stem
        if stem.startswith("_Color_"):
            num_str = stem.replace("_Color_", "")
            # Strip parenthesized suffix like (1)
            if "(" in num_str:
                num_str = num_str[:num_str.index("(")]
            frame_num = int(num_str)
        elif stem.startswith("f"):
            num_str = stem[1:]
            if "(" in num_str:
                num_str = num_str[:num_str.index("(")]
            frame_num = int(num_str)
        else:
            frame_num = i

        color_img = cv2.imread(str(color_path))
        if color_img is None:
            continue
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Load depth
        if skip_depth:
            # Use synthetic constant depth at expected human position
            # MediaPipe needs depth to anchor 3D landmark positions
            # Expected human position: behind targets at z = midpoint(t2,t3) + 0.3
            t2_z = targets[1]["z"] if len(targets) > 1 else targets[0]["z"]
            t3_z = targets[2]["z"] if len(targets) > 2 else targets[-1]["z"]
            expected_depth = (t2_z + t3_z) / 2 + 0.3
            depth_img = np.full((color_img.shape[0], color_img.shape[1]),
                                expected_depth, dtype=np.float32)
        else:
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

    # If depth is unreliable (skip_depth), translate skeleton to expected position
    # This preserves arm direction while fixing absolute position for
    # ground plane intersection and target distance calculations
    if skip_depth:
        hip_positions = []
        for result in human_results.values():
            if result.landmarks_3d and len(result.landmarks_3d) > 24:
                lh = np.array(result.landmarks_3d[23])
                rh = np.array(result.landmarks_3d[24])
                hc = (lh + rh) / 2.0
                if not np.all(hc == 0) and not np.any(np.isnan(hc)):
                    hip_positions.append(hc)

        if hip_positions:
            avg_hip = np.mean(hip_positions, axis=0)

            # Expected human position: between targets 2 & 3, 0.3m behind
            t2 = targets[1] if len(targets) > 1 else targets[0]
            t3 = targets[2] if len(targets) > 2 else targets[-1]
            expected_x = (t2["x"] + t3["x"]) / 2
            expected_y = (t2["y"] + t3["y"]) / 2
            expected_z = (t2["z"] + t3["z"]) / 2 + 0.3

            offset = np.array([expected_x - avg_hip[0],
                               expected_y - avg_hip[1],
                               expected_z - avg_hip[2]])
            print(f"    Skeleton translation: hip {avg_hip[0]:.3f},{avg_hip[1]:.3f},{avg_hip[2]:.3f}"
                  f" → {expected_x:.3f},{expected_y:.3f},{expected_z:.3f}"
                  f" (offset {offset[0]:+.3f},{offset[1]:+.3f},{offset[2]:+.3f})")

            # Apply translation to all landmarks in all frames
            for result in human_results.values():
                if result.landmarks_3d:
                    result.landmarks_3d = [
                        tuple((np.array(lm) + offset).tolist())
                        for lm in result.landmarks_3d
                    ]

    # Validate pointer detection
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
                ground_plane_rotation=None
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
    """Validate that the detected skeleton is likely the adult pointer."""
    if not human_results:
        return

    h = intrinsics["height"]
    nose_ys = []
    hip_positions = []

    for result in human_results.values():
        if result.landmarks_2d and len(result.landmarks_2d) > 0:
            nose_y = result.landmarks_2d[0][1]
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
        print(f"    Pointer hip 3D: x={avg_hip[0]:.3f}, y={avg_hip[1]:.3f}, z={avg_hip[2]:.3f}")


# ─────────────────────────────────────────────────────────────
#  Per-subject processing
# ─────────────────────────────────────────────────────────────
def process_subject(subject_name: str, zip_path: Path,
                    skip_targets: bool = False):
    """Process one BDL production subject through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"Processing: {subject_name}")
    print(f"{'='*60}")

    # Extract zip
    work_dir = WORK_DIR / subject_name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    extract_subject_from_zip(zip_path, work_dir)

    # Find the actual data directory (handle double nesting)
    data_dir = find_data_dir_in_extracted(work_dir)
    print(f"  Data dir: {data_dir.relative_to(work_dir)}")

    # Load intrinsics
    intrinsics = load_intrinsics(data_dir)
    print(f"  Intrinsics: {intrinsics['width']}x{intrinsics['height']}, "
          f"fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Find trial directories
    trials = find_trial_dirs(data_dir)
    if not trials:
        print(f"  ERROR: No trial directories found in {data_dir}")
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
    if work_dir.exists():
        print(f"\n  Cleaning up {work_dir}...")
        shutil.rmtree(work_dir)

    print(f"\n  {subject_name}: {ok} OK, {fail} failed / {len(trials)} trials")
    return ok, fail


def process_subject_from_dir(subject_name: str, data_dir_path: Path,
                             skip_targets: bool = False,
                             skip_depth: bool = False):
    """Process one BDL production subject from an already-extracted directory."""
    print(f"\n{'='*60}")
    print(f"Processing (from dir): {subject_name}")
    print(f"{'='*60}")

    # Find data dir (handle double nesting)
    data_dir = data_dir_path
    inner = data_dir / data_dir.name
    if inner.is_dir():
        data_dir = inner

    # Look for trial directories
    has_trials = any(d.is_dir() and d.name.isdigit()
                    for d in data_dir.iterdir() if d.is_dir())
    if not has_trials:
        data_dir = find_data_dir_in_extracted(data_dir_path)

    print(f"  Data dir: {data_dir}")

    # Load intrinsics
    intrinsics = load_intrinsics(data_dir)
    print(f"  Intrinsics: {intrinsics['width']}x{intrinsics['height']}, "
          f"fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")

    # Find trial directories
    trials = find_trial_dirs(data_dir)
    if not trials:
        print(f"  ERROR: No trial directories found in {data_dir}")
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
                trial_dir, intrinsics, subject_targets, output_trial_dir,
                skip_depth=skip_depth)

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
        if not subject_dir.is_dir() or subject_dir.name.startswith("_"):
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
        description="Batch reprocess BDL Point Production data")
    parser.add_argument("--subject", help="Process only this subject (prefix match)")
    parser.add_argument("--skip-targets", action="store_true",
                        help="Skip YOLO target detection, use fixed targets")
    parser.add_argument("--source", type=str, default=str(SSD_SOURCE),
                        help=f"Source directory (default: {SSD_SOURCE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--data-dir", type=str,
                        help="Process a single already-extracted directory")
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip depth loading (for video-only subjects with lossy depth)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If --data-dir is specified, process just that directory
    if args.data_dir:
        data_dir_path = Path(args.data_dir)
        subject_name = args.subject or data_dir_path.name
        print(f"Source (dir): {data_dir_path}")
        print(f"Output: {OUTPUT_DIR}")
        if args.skip_depth:
            print(f"Depth: SKIPPED (using MediaPipe monocular estimation)")
        ok, fail = process_subject_from_dir(
            subject_name, data_dir_path,
            skip_targets=args.skip_targets,
            skip_depth=args.skip_depth)
        if ok > 0:
            print(f"\n{'='*60}")
            print("Generating global CSV...")
            generate_global_csv(OUTPUT_DIR)
        print(f"\n{'='*60}")
        print(f"DONE: {ok} trials OK, {fail} failed")
        print(f"{'='*60}")
        return

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # BDL: flat folder of zips (each zip IS a subject) + extracted dirs
    zips = sorted([z for z in source_dir.glob("*.zip")
                   if not z.name.startswith("._")])
    # Also find already-extracted directories (not zips)
    dirs = sorted([d for d in source_dir.iterdir()
                   if d.is_dir() and not d.name.startswith(".")
                   and not d.name.startswith("_")])
    # Exclude dirs that have a matching zip (they'll be processed via zip)
    zip_stems = {z.stem for z in zips}
    extra_dirs = [d for d in dirs if d.name not in zip_stems]

    if args.subject:
        zips = [z for z in zips if z.stem.startswith(args.subject)]
        extra_dirs = [d for d in extra_dirs if d.name.startswith(args.subject)]

    total_items = len(zips) + len(extra_dirs)

    print(f"Source: {source_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Subjects: {total_items} ({len(zips)} zips, {len(extra_dirs)} dirs)")
    print(f"YOLO model: {YOLO_MODEL}")
    print()

    total_ok = 0
    total_fail = 0
    idx = 0

    for i, zip_path in enumerate(zips, 1):
        idx += 1
        subject_name = zip_path.stem
        print(f"\n[{idx}/{total_items}] {subject_name}")
        ok, fail = process_subject(
            subject_name,
            zip_path,
            skip_targets=args.skip_targets,
        )
        total_ok += ok
        total_fail += fail

    for d in extra_dirs:
        idx += 1
        subject_name = d.name
        print(f"\n[{idx}/{total_items}] {subject_name} (from dir)")
        ok, fail = process_subject_from_dir(
            subject_name,
            d,
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

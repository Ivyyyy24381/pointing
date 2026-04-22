#!/usr/bin/env python3
"""
Batch reprocess ALL CCD Pointing Comprehension subjects.

Full pipeline per trial:
  1. Extract from zip (if needed)
  2. YOLO target detection (automatic) → target_coordinates.json
  3. SAM3 segmentation (automatic, text="baby") → masked_video.mp4
  4. MediaPipe skeleton extraction → skeleton JSON
  5. pose_visualize → CSV + plots
  6. Copy results to output directory
  7. Run optimal path deviation
  8. Cleanup extracted data

Usage:
    # Process all subjects (SAM3 auto + YOLO targets)
    python batch_reprocess_CCD.py

    # Process single subject
    python batch_reprocess_CCD.py --subject CCD0384

    # Skip SAM, reuse existing masked_video.mp4
    python batch_reprocess_CCD.py --skip-sam

    # Skip SAM + MediaPipe, reuse existing skeleton JSON (just redo pose_visualize)
    python batch_reprocess_CCD.py --skip-sam --skip-mediapipe

    # Use fixed targets instead of YOLO detection
    python batch_reprocess_CCD.py --fixed-targets
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

# ─── Project paths ───
POINTING_DIR = Path("/home/tigerli/Documents/GitHub/pointing")
SSD_SOURCE = Path("/media/tigerli/Extreme SSD/pointing_data/point_comprehension_CCD")
OUTPUT_DIR = Path("/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output")
WORK_DIR = Path("/home/tigerli/Documents/pointing_data/_ccd_work")

# Add project paths for imports
sys.path.insert(0, str(POINTING_DIR))
sys.path.insert(0, str(POINTING_DIR / "step3_subject_extraction" / "dog_script"))
sys.path.insert(0, str(POINTING_DIR / "step0_data_loading"))

YOLO_MODEL = POINTING_DIR / "step0_data_loading" / "best.pt"

# ─── Fixed fallback target positions for CCD comprehension ───
# Recovered from old pipeline CSV spherical coordinates
CCD_TARGETS_FIXED = [
    {"label": "target_1", "x":  0.4430, "y": 0.6570, "z": 1.6520},
    {"label": "target_2", "x":  0.1560, "y": 0.5440, "z": 2.3530},
    {"label": "target_3", "x": -0.4620, "y": 0.4370, "z": 2.8180},
    {"label": "target_4", "x": -1.1500, "y": 0.3520, "z": 3.0760},
]


# ─────────────────────────────────────────────────────────────
#  Intrinsics
# ─────────────────────────────────────────────────────────────
def load_intrinsics(data_dir: Path):
    """Load camera intrinsics from rosbag_metadata.yaml.

    Falls back to default intrinsics based on detected image resolution.
    """
    meta_path = data_dir / "rosbag_metadata.yaml"
    if not meta_path.exists():
        # Try parent
        meta_path = data_dir.parent / "rosbag_metadata.yaml"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        intr = meta["intrinsics"]
        return {
            "fx": intr["fx"], "fy": intr["fy"],
            "cx": intr["ppx"], "cy": intr["ppy"],
            "width": intr["width"], "height": intr["height"],
        }

    # Auto-detect resolution from first trial's color frames
    trials = find_trial_dirs(data_dir)
    if trials:
        layout = _resolve_trial_layout(trials[0])
        if layout:
            color_files = sorted([f for f in layout["color_dir"].iterdir()
                                  if f.suffix == ".png" and not f.name.startswith("._")])
            if color_files:
                img = cv2.imread(str(color_files[0]))
                if img is not None:
                    h, w = img.shape[:2]
                    if w == 640 and h == 480:
                        # RealSense D435 default 640x480 intrinsics
                        print(f"  Using default D435 640x480 intrinsics")
                        return {"fx": 382.0, "fy": 382.0,
                                "cx": 320.0, "cy": 240.0,
                                "width": 640, "height": 480}

    # Default CCD side camera intrinsics (1280x720)
    return {"fx": 648.52, "fy": 648.52, "cx": 648.61, "cy": 366.44,
            "width": 1280, "height": 720}


# ─── Relative offsets between targets (from CCD_TARGETS_FIXED) ───
# Precompute: each target's offset from the centroid of all 4 targets.
# The relative geometry is FIXED across all subjects.
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

    Strategy:
      - The 4 targets always form the same curved arc pattern.
      - YOLO gives us 2D detections (reliable) + depth (potentially noisy).
      - Use detections with reasonable depth as anchors to compute a 3D
        translation that maps the fixed pattern onto the scene.
      - Apply the fixed relative offsets from this anchor to get all 4 targets.

    A detection is "reasonable" if its depth is in [0.5, 5.0] meters
    (targets are at ~1.6–3.1m depth).
    """
    if not raw_targets:
        return raw_targets

    # Filter anchors: targets with reasonable depth
    anchors = [t for t in raw_targets
                if 0.5 < t["z"] < 5.0 and not np.isnan(t["z"])]

    if not anchors:
        print("    Target correction: no anchors with valid depth, using fixed targets")
        return CCD_TARGETS_FIXED

    # Compute the translation: average difference between anchor detections
    # and their corresponding positions in the fixed pattern.
    # Match by label (target_1 -> offset[0], etc.)
    shifts = []
    for anchor in anchors:
        # Find the matching fixed offset by label
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

    # Average shift from all anchors
    avg_shift = {
        "dx": np.mean([s["dx"] for s in shifts]),
        "dy": np.mean([s["dy"] for s in shifts]),
        "dz": np.mean([s["dz"] for s in shifts]),
    }

    # Apply fixed pattern + shift to get all 4 corrected targets
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
#  Target detection (YOLO)
# ─────────────────────────────────────────────────────────────
def detect_targets_yolo(trial_dir: Path, intrinsics: dict) -> list:
    """
    Run YOLO target detection on a trial directory.

    Detects cup-like targets using YOLO, then corrects positions using
    the known fixed relative geometry of the 4 CCD targets. Even if
    only 1-2 targets have good depth, the fixed arc pattern is shifted
    to match, giving reliable positions for all 4.

    Returns list of targets: [{"label": ..., "x": ..., "y": ..., "z": ...}, ...]
    """
    from target_detector import TargetDetector

    # Resolve trial layout (supports both old and new directory structure)
    layout = _resolve_trial_layout(trial_dir)
    if layout is None:
        print("    Cannot resolve trial layout for target detection")
        return []
    color_dir = layout["color_dir"]
    depth_dir = layout["depth_dir"]
    depth_format = layout["depth_format"]

    # Get color files (exclude macOS resource forks)
    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    if not color_files:
        print("    No color files for target detection")
        return []

    # Use middle frame for stable detection
    mid_idx = len(color_files) // 2
    color_path = color_files[mid_idx]
    color_img = cv2.imread(str(color_path))

    # Load corresponding depth
    depth_img = None
    h, w = color_img.shape[:2]

    if depth_format == "npy":
        # New layout: frame_NNNNNN.npy matching frame_NNNNNN.png
        depth_name = color_path.stem + ".npy"
        depth_path = depth_dir / depth_name
        if depth_path.exists():
            raw = np.load(str(depth_path))
            depth_img = raw.astype(np.float32) / 1000.0
            if depth_img.shape[:2] != (h, w):
                depth_img = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # Old layout: Depth_NNNNNN.raw matching Color_NNNNNN.png
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
                dw, dh = None, None

            if dw is not None:
                with open(depth_path, "rb") as f:
                    raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((dh, dw))
                if (dw, dh) != (w, h):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
                depth_img = raw.astype(np.float32) / 1000.0

    # Run YOLO detection
    detector = TargetDetector(model_path=str(YOLO_MODEL), confidence_threshold=0.3)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    detections = detector.detect(color_img, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)

    if not detections:
        # Try with lower confidence
        detector.confidence_threshold = 0.15
        detections = detector.detect(color_img, depth_img, fx=fx, fy=fy, cx=cx, cy=cy)

    if not detections:
        print(f"    YOLO: No targets detected in frame {color_path.name}")
        return []

    # Convert to target format (sorted left-to-right by pixel x)
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
            # Compute 3D from pixel center + depth
            cx_px, cy_px = det.center
            x = (cx_px - cx) * det.depth / fx
            y = (cy_px - cy) * det.depth / fy
            raw_targets.append({
                "label": f"target_{i}",
                "x": float(x), "y": float(y), "z": float(det.depth),
            })

    print(f"    YOLO: Detected {len(raw_targets)} raw targets")
    for t in raw_targets:
        print(f"      {t['label']}: x={t['x']:.3f}, y={t['y']:.3f}, z={t['z']:.3f}")

    # Correct using fixed relative geometry
    targets = _correct_targets_with_fixed_geometry(raw_targets)

    if targets is not raw_targets:
        print(f"    YOLO: Corrected targets (fixed geometry):")
        for t in targets:
            print(f"      {t['label']}: x={t['x']:.3f}, y={t['y']:.3f}, z={t['z']:.3f}")

    return targets


def save_target_coordinates_json(trial_dir: Path, targets: list):
    """Save targets as target_coordinates.json in the format expected by pose_visualize."""
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
#  SAM3 Segmentation (automatic, text-prompt)
# ─────────────────────────────────────────────────────────────
_sam3_video_predictor = None  # reuse across trials


def _get_sam3_video_predictor():
    """Lazy-load the SAM3 video predictor (reused across trials)."""
    global _sam3_video_predictor
    if _sam3_video_predictor is not None:
        return _sam3_video_predictor
    import torch
    from sam3.model_builder import build_sam3_video_predictor
    print("  Loading SAM3 video model...")
    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    _sam3_video_predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    print("  SAM3 video model loaded")
    return _sam3_video_predictor


def run_sam_segmentation_subprocess(trial_dir: Path) -> bool:
    """
    Run SAM3 segmentation in a separate subprocess to isolate crashes.

    SAM3's video predictor sometimes crashes with SIGABRT/double-free
    during session cleanup. By running in a subprocess, the crash doesn't
    kill the parent pipeline. The masked_video.mp4 is written to disk
    before cleanup, so it persists even if the subprocess crashes.
    """
    trial_dir_str = str(trial_dir)
    script = f'''
import sys
sys.path.insert(0, "{POINTING_DIR}")
sys.path.insert(0, "{POINTING_DIR / "step3_subject_extraction" / "dog_script"}")
sys.path.insert(0, "{POINTING_DIR / "step0_data_loading"}")
from pathlib import Path
from batch_reprocess_CCD import run_sam_segmentation
success = run_sam_segmentation(Path("{trial_dir_str}"))
print("SAM3_RESULT:" + str(success))
sys.stdout.flush()
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
        cwd=str(POINTING_DIR),
    )
    # Check for success in output (process may crash on cleanup but still succeed)
    stdout = result.stdout
    if "SAM3_RESULT:True" in stdout:
        return True
    # Even if process crashed, check if outputs were created
    seg_dir = trial_dir / "segmented_color"
    has_seg = seg_dir.is_dir() and any(seg_dir.iterdir())
    masked_video = trial_dir / "masked_video.mp4"
    if has_seg or (masked_video.exists() and masked_video.stat().st_size > 0):
        print(f"    SAM3 subprocess crashed but masked_video.mp4 exists — continuing")
        return True
    # Print stderr for debugging
    if result.returncode != 0:
        # Filter SAM3 output lines
        for line in stdout.split("\n"):
            if "SAM3" in line or "person" in line or "baby" in line or "Segmented" in line:
                print(f"    {line.strip()}")
        print(f"    SAM3 subprocess exited with code {result.returncode}")
    return False


def _select_baby_obj_id(outputs: dict) -> int:
    """
    From SAM3 add_prompt outputs with multiple detected persons,
    select the one most likely to be the baby (lowest center-of-mass
    in the frame, since the baby crawls on the floor).

    Returns the obj_id to keep, or -1 if no valid objects.
    """
    obj_ids = outputs.get("out_obj_ids", np.array([]))
    masks = outputs.get("out_binary_masks", np.array([]))

    if len(obj_ids) == 0 or masks.size == 0:
        return -1
    if len(obj_ids) == 1:
        return int(obj_ids[0])

    # Find object with highest center-of-mass Y (lowest in frame = baby on floor)
    best_id = -1
    best_y = -1
    for i, obj_id in enumerate(obj_ids):
        mask = masks[i].squeeze() if masks.ndim > 2 else masks.squeeze()
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        mean_y = float(ys.mean())
        if mean_y > best_y:
            best_y = mean_y
            best_id = int(obj_id)

    return best_id


def run_sam_segmentation(trial_dir: Path):
    """
    Run SAM3 automatic segmentation on a trial using text prompt.
    Creates masked_video.mp4 in the trial directory.

    Strategy:
      1. Try "baby" as text prompt first — if SAM3 recognizes it, this
         directly detects only the baby (no adult confusion).
      2. Fall back to "person" if "baby" yields no detections.
      3. When multiple persons are detected, propagate all and select the
         one with the most 2D motion (baby crawls, parent sits still).

    Pipeline:
      1. Write Color frames as numbered JPEGs to temp dir
      2. Start SAM3 video session
      3. Add text prompt on frame 0 (try "baby" then "person")
      4. Propagate segmentation through all frames
      5. Motion-based baby selection from tracked objects
      6. Apply masks to create masked_video.mp4
    """
    import tempfile
    import torch
    from sam3.visualization_utils import prepare_masks_for_visualization

    # Resolve color directory (supports both old and new layout)
    layout = _resolve_trial_layout(trial_dir)
    color_dir = layout["color_dir"] if layout else trial_dir / "Color"
    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    if not color_files:
        print(f"    No color files for SAM3 segmentation")
        return False

    predictor = _get_sam3_video_predictor()

    with tempfile.TemporaryDirectory() as tmpdir:
        jpeg_dir = Path(tmpdir) / "frames"
        jpeg_dir.mkdir()

        # Write color frames as numbered JPEGs (SAM3 expects this format)
        for i, cf in enumerate(color_files):
            img = cv2.imread(str(cf))
            if img is not None:
                cv2.imwrite(str(jpeg_dir / f"{i:06d}.jpg"), img)

        # Start video session
        resp = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(jpeg_dir),
        })
        session_id = resp["session_id"]

        # Try prompts in order: "baby" first, then "person"
        text_prompts = ["baby", "person"]
        baby_obj_id = None
        n_detected = 0

        for text_prompt in text_prompts:
            print(f"    Running SAM3 segmentation "
                  f"(text='{text_prompt}', {len(color_files)} frames)...")

            # Add text prompt on first frame
            prompt_resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
            })

            prompt_outputs = prompt_resp.get("outputs", {})
            obj_ids = prompt_outputs.get("out_obj_ids", np.array([]))
            n_detected = len(obj_ids)

            if n_detected > 1:
                baby_obj_id = _select_baby_obj_id(prompt_outputs)
                print(f"    SAM3: {n_detected} persons detected with "
                      f"'{text_prompt}', initial pick=object {baby_obj_id}")
                break
            elif n_detected == 1:
                baby_obj_id = int(obj_ids[0])
                print(f"    SAM3: 1 person detected with "
                      f"'{text_prompt}' (object {baby_obj_id})")
                break
            else:
                print(f"    SAM3: No detection with '{text_prompt}' on frame 0")
                # Reset session before trying next prompt
                predictor.handle_request({
                    "type": "reset_session", "session_id": session_id,
                })

        # If no detection on frame 0 with any prompt, try mid frame
        if baby_obj_id is None:
            mid_idx = len(color_files) // 2
            print(f"    SAM3: Trying 'person' on frame {mid_idx}...")
            prompt_resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": mid_idx,
                "text": "person",
            })
            prompt_outputs = prompt_resp.get("outputs", {})
            obj_ids = prompt_outputs.get("out_obj_ids", np.array([]))
            n_detected = len(obj_ids)
            if n_detected > 1:
                baby_obj_id = _select_baby_obj_id(prompt_outputs)
                print(f"    SAM3: Mid-frame: {n_detected} persons, "
                      f"initial pick=object {baby_obj_id}")
            elif n_detected == 1:
                baby_obj_id = int(obj_ids[0])
                print(f"    SAM3: Mid-frame: 1 person (object {baby_obj_id})")

        if baby_obj_id is None:
            try:
                predictor.handle_request({
                    "type": "close_session", "session_id": session_id,
                })
            except Exception:
                pass
            print(f"    SAM3: No persons detected, cannot segment")
            return False

        # Propagate through entire video (track ALL detected persons)
        raw_outputs = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            raw_outputs[response["frame_index"]] = response["outputs"]

        # Convert raw outputs to clean masks
        formatted = prepare_masks_for_visualization(raw_outputs)

        # ── Motion-based baby selection ──
        # Analyze centroid trajectories of ALL tracked objects.
        # The baby crawls toward targets (moves), the parent sits still.
        # Pick the object with the most 2D motion as the baby.
        if n_detected > 1:
            object_centroids = {}  # obj_id -> list of (cx, cy)
            for fidx, fdata in formatted.items():
                if not isinstance(fdata, dict):
                    continue
                for oid, m in fdata.items():
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    m = m.squeeze()
                    if m.ndim == 2 and m.sum() > 0:
                        ys, xs = np.where(m > 0)
                        if oid not in object_centroids:
                            object_centroids[oid] = []
                        object_centroids[oid].append((float(xs.mean()), float(ys.mean())))

            best_obj_id = baby_obj_id
            best_motion = -1
            for oid, centroids in object_centroids.items():
                if len(centroids) < 5:
                    continue
                cxs = [c[0] for c in centroids]
                cys = [c[1] for c in centroids]
                motion = (max(cxs) - min(cxs)) + (max(cys) - min(cys))
                if motion > best_motion:
                    best_motion = motion
                    best_obj_id = oid

            if best_obj_id != baby_obj_id:
                print(f"    SAM3: Motion analysis: switching from object "
                      f"{baby_obj_id} to {best_obj_id} "
                      f"(motion: {best_motion:.0f}px)")
                baby_obj_id = best_obj_id

        # Build per-frame binary masks, using ONLY the baby's object ID
        frame_masks = {}
        for fidx, fdata in formatted.items():
            if isinstance(fdata, dict):
                # Use only baby_obj_id if multiple objects, else merge all
                mask = fdata.get(baby_obj_id)
                if mask is None and len(fdata) == 1:
                    # Only one object tracked for this frame
                    mask = next(iter(fdata.values()))
                if mask is not None:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    mask = mask.squeeze()
                    if mask.ndim == 2 and mask.sum() > 0:
                        frame_masks[fidx] = mask.astype(np.uint8)

        # Close session to free GPU memory
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception:
            pass

    detected = len(frame_masks)
    total = len(color_files)
    print(f"    SAM3: Segmented baby in {detected}/{total} frames")

    if detected == 0:
        print(f"    SAM3: No masks after propagation, cannot create masked video")
        return False

    # Create masked_video.mp4 AND segmented_color/ frames
    masked_video = trial_dir / "masked_video.mp4"
    seg_out_dir = trial_dir / "segmented_color"
    seg_out_dir.mkdir(exist_ok=True)

    first_img = cv2.imread(str(color_files[0]))
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 5.0
    writer = cv2.VideoWriter(str(masked_video), fourcc, fps, (w, h))

    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is None:
            continue
        if i in frame_masks:
            mask = frame_masks[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            masked = img.copy()
            masked[mask == 0] = 0
            writer.write(masked)
        else:
            masked = np.zeros_like(img)
            writer.write(masked)
        # Save per-frame segmented image for mask-based depth
        cv2.imwrite(str(seg_out_dir / f"vis_frame_{i:06d}.png"), masked)

    writer.release()

    if masked_video.exists() and masked_video.stat().st_size > 0:
        print(f"    SAM3: Created {masked_video.name}")
        return True
    else:
        print(f"    SAM3: Failed to create masked_video.mp4")
        return False


# ─────────────────────────────────────────────────────────────
#  MediaPipe skeleton extraction
# ─────────────────────────────────────────────────────────────
def run_mediapipe(video_path: Path) -> Path:
    """Run MediaPipe on a video, return path to skeleton JSON."""
    from detect_dog_skeleton import run_mediapipe_json

    print(f"    Running MediaPipe on {video_path.name}...")
    run_mediapipe_json(str(video_path))

    skeleton_path = video_path.with_name(video_path.stem + "_skeleton.json")
    if skeleton_path.exists():
        print(f"    Skeleton: {skeleton_path.name}")
        return skeleton_path
    print(f"    ERROR: Skeleton JSON not created")
    return None


# ─────────────────────────────────────────────────────────────
#  pose_visualize
# ─────────────────────────────────────────────────────────────
def run_pose_visualize(skeleton_json: Path, side_view: bool = True):
    """Run pose_visualize on a skeleton JSON file."""
    # CWD must be pointing repo root for ./config/skeleton_config.json
    original_cwd = os.getcwd()
    os.chdir(str(POINTING_DIR))

    try:
        from dog_pose_visualize import pose_visualize
        print(f"    Running pose_visualize...")
        pose_visualize(str(skeleton_json), side_view=side_view, dog=False)
        print(f"    pose_visualize complete")
    except Exception as e:
        # cv2.destroyAllWindows() may fail in headless mode - that's OK
        if "destroyAllWindows" in str(e):
            print(f"    pose_visualize complete (GUI cleanup warning ignored)")
        else:
            raise
    finally:
        os.chdir(original_cwd)


# ─────────────────────────────────────────────────────────────
#  Video creation helper
# ─────────────────────────────────────────────────────────────
def create_video_from_frames(color_dir: Path, output_path: Path, fps: float = 5.0):
    """Create MP4 video from Color PNG frames (fallback when no SAM)."""
    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    if not color_files:
        return False

    first_img = cv2.imread(str(color_files[0]))
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for f in color_files:
        img = cv2.imread(str(f))
        if img is not None:
            out.write(img)
    out.release()
    return True


# ─────────────────────────────────────────────────────────────
#  Mask-based depth computation (no skeleton needed)
# ─────────────────────────────────────────────────────────────
def _load_depth_frame(depth_path: Path, depth_format: str,
                      expected_h: int, expected_w: int):
    """Load a single depth frame from .raw or .npy file. Returns depth in meters."""
    if depth_format == "npy":
        raw = np.load(str(depth_path))
        return raw.astype(np.float32) / 1000.0
    else:  # raw
        raw_size = depth_path.stat().st_size
        raw_pixels = raw_size // 2
        # Determine dimensions
        if raw_pixels == expected_w * expected_h:
            dw, dh = expected_w, expected_h
        elif raw_pixels == 640 * 480:
            dw, dh = 640, 480
        elif raw_pixels == 1280 * 720:
            dw, dh = 1280, 720
        else:
            for cw, ch in [(848, 480), (424, 240)]:
                if raw_pixels == cw * ch:
                    dw, dh = cw, ch
                    break
            else:
                return None
        with open(depth_path, "rb") as f:
            raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((dh, dw))
        return raw.astype(np.float32) / 1000.0


def compute_trace_from_mask_depth(trial_dir: Path, intrinsics: dict,
                                  targets: list,
                                  min_valid_ratio: float = 0.1,
                                  smooth_window: int = 5) -> dict:
    """
    Compute baby 3D trace directly from SAM mask + depth data.

    For each frame:
      1. Load segmented_color frame → binary mask (non-black pixels)
      2. Load depth file → depth map
      3. Compute 2D centroid of mask pixels
      4. Median valid depth within mask → avg_depth
      5. Convert (centroid_u, centroid_v, avg_depth) → 3D position

    Supports both old layout (Color/Depth .raw) and new layout (cam1/color/depth .npy).
    Returns dict with trace_3d, per_frame_target_metrics, fps, etc., or None if
    mask quality is too low (caller should fall back to MediaPipe).
    """
    seg_dir = trial_dir / "segmented_color"

    # Must have segmented_color/ from SAM
    if not seg_dir.is_dir() or not any(seg_dir.iterdir()):
        print(f"    No segmented_color/ directory — cannot use mask-based approach")
        return None

    # Resolve trial layout for depth files
    layout = _resolve_trial_layout(trial_dir)
    if layout is None:
        print(f"    Cannot determine trial data layout")
        return None

    color_dir = layout["color_dir"]
    depth_dir = layout["depth_dir"]
    depth_format = layout["depth_format"]

    # Get sorted frame lists
    seg_files = sorted([f for f in seg_dir.iterdir()
                        if f.suffix == ".png" and not f.name.startswith("._")])
    color_files = sorted([f for f in color_dir.iterdir()
                          if f.suffix == ".png" and not f.name.startswith("._")])
    depth_ext = ".npy" if depth_format == "npy" else ".raw"
    depth_files = sorted([f for f in depth_dir.iterdir()
                          if f.suffix == depth_ext and not f.name.startswith("._")])

    n_frames = min(len(seg_files), len(depth_files))
    if n_frames == 0:
        print(f"    No frames to process (seg={len(seg_files)}, depth={len(depth_files)})")
        return None

    # Check mask quality: sample up to 20 frames to count non-black
    sample_indices = np.linspace(0, n_frames - 1, min(20, n_frames), dtype=int)
    black_count = 0
    for idx in sample_indices:
        img = cv2.imread(str(seg_files[idx]))
        if img is None or np.max(img) == 0:
            black_count += 1
    mask_quality = 1.0 - black_count / len(sample_indices)
    print(f"    Mask quality: {mask_quality*100:.1f}% non-black (sampled {len(sample_indices)} frames)")

    if mask_quality < 0.15:
        print(f"    Mask quality too low — need MediaPipe fallback")
        return None

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    img_w, img_h = intrinsics["width"], intrinsics["height"]

    # Get actual image dimensions from first segmented frame
    first_seg = cv2.imread(str(seg_files[0]))
    seg_h, seg_w = first_seg.shape[:2]

    # Scale intrinsics if image dimensions differ from nominal
    scale_x = seg_w / img_w
    scale_y = seg_h / img_h
    if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
        fx_scaled = fx * scale_x
        fy_scaled = fy * scale_y
        cx_scaled = cx * scale_x
        cy_scaled = cy * scale_y
        print(f"    Scaling intrinsics: {img_w}x{img_h} -> {seg_w}x{seg_h}")
    else:
        fx_scaled, fy_scaled = fx, fy
        cx_scaled, cy_scaled = cx, cy

    # Determine FPS
    fps = 10.0
    masked_video = trial_dir / "masked_video.mp4"
    if masked_video.exists():
        temp_cap = cv2.VideoCapture(str(masked_video))
        fps = temp_cap.get(cv2.CAP_PROP_FPS) or fps
        temp_cap.release()
    else:
        # Try cam1 color video
        for vname in ["cam1_color.mp4", "Color.mp4"]:
            vpath = trial_dir / "cam1" / vname if "cam1" in vname else trial_dir / vname
            if vpath.exists():
                temp_cap = cv2.VideoCapture(str(vpath))
                fps = temp_cap.get(cv2.CAP_PROP_FPS) or fps
                temp_cap.release()
                break

    # Extract start frame index from first color file name
    start_frame_idx = 0
    if color_files:
        try:
            start_frame_idx = int(color_files[0].stem.split("_")[-1])
        except ValueError:
            pass

    # Process each frame
    trace_3d = []
    per_frame_target_metrics = []

    for i in range(n_frames):
        seg_img = cv2.imread(str(seg_files[i]))

        if seg_img is None:
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            continue

        gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        mask = gray > 5  # threshold above 0 for compression artifacts

        if np.sum(mask) < 10:
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            continue

        # 2D centroid of mask
        ys, xs = np.where(mask)
        u_center = float(np.mean(xs))
        v_center = float(np.mean(ys))

        # Load depth
        depth_m = _load_depth_frame(depth_files[i], depth_format, img_h, img_w)
        if depth_m is None:
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            continue

        # Always resize depth to match mask dimensions
        if depth_m.shape[:2] != (seg_h, seg_w):
            depth_m = cv2.resize(depth_m, (seg_w, seg_h),
                                 interpolation=cv2.INTER_NEAREST)

        # Safety: verify dimensions match
        if depth_m.shape[:2] != mask.shape:
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            continue

        # Apply mask to depth and compute median
        masked_depth = depth_m[mask]
        valid_depth = masked_depth[(masked_depth > 0.1) & (masked_depth < 10.0)]

        if len(valid_depth) < 5:
            trace_3d.append([np.nan, np.nan, np.nan])
            per_frame_target_metrics.append({})
            continue

        avg_depth = float(np.median(valid_depth))

        # Convert to 3D
        x_m = (u_center - cx_scaled) * avg_depth / fx_scaled
        y_m = (v_center - cy_scaled) * avg_depth / fy_scaled
        z_m = avg_depth

        trace_3d.append([x_m, y_m, z_m])

        # Compute target distances
        metrics = {}
        for t in targets:
            label = t["label"]
            dx = t["x"] - x_m
            dy = t["y"] - y_m
            dz = t["z"] - z_m
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            theta = np.arccos(dz / r) if r > 0 else 0.0
            phi = np.arctan2(dy, dx)
            metrics[f"{label}_r"] = r
            metrics[f"{label}_theta"] = theta
            metrics[f"{label}_phi"] = phi
        per_frame_target_metrics.append(metrics)

    # Report depth validity
    n_valid = sum(1 for pt in trace_3d if not np.isnan(pt[0]))
    print(f"    Depth validity: {n_valid}/{n_frames} frames ({100*n_valid/n_frames:.1f}%)")

    # ── Filter invalid 3D positions ──
    # Baby walks from near camera (~z=0.5-1.5m) toward targets (~z=1.6-3.1m).
    # Any z > 4m or extreme x values are depth noise / wrong detection.
    # Baby walks from camera (~z=0.5) toward targets (z=1.65-3.08m).
    # Z beyond max_target + margin is depth noise.
    max_target_z = max(t["z"] for t in targets) if targets else 3.1
    VALID_Z_MIN = 0.3
    VALID_Z_MAX = max_target_z + 0.5  # ~3.6m
    VALID_X_MIN, VALID_X_MAX = -3.0, 1.5
    MAX_STEP_M = 0.5  # max displacement per frame (baby crawling, not teleporting)

    n_filtered = 0
    for i in range(len(trace_3d)):
        pt = trace_3d[i]
        if np.isnan(pt[0]):
            continue
        x, y, z = pt[0], pt[1], pt[2]
        # Range filter
        if z < VALID_Z_MIN or z > VALID_Z_MAX or x < VALID_X_MIN or x > VALID_X_MAX:
            trace_3d[i] = [np.nan, np.nan, np.nan]
            per_frame_target_metrics[i] = {}
            n_filtered += 1
            continue
        # Velocity filter: reject jumps from previous valid point
        if i > 0:
            prev = trace_3d[i - 1]
            if not np.isnan(prev[0]):
                step = np.sqrt((x - prev[0])**2 + (z - prev[2])**2)
                if step > MAX_STEP_M:
                    trace_3d[i] = [np.nan, np.nan, np.nan]
                    per_frame_target_metrics[i] = {}
                    n_filtered += 1
                    continue

    n_after_filter = sum(1 for pt in trace_3d if not np.isnan(pt[0]))
    if n_filtered > 0:
        print(f"    Filtered {n_filtered} outlier points → {n_after_filter}/{n_frames} valid")

    # Interpolate gaps (only between valid points, require minimum valid ratio)
    from dog_pose_visualize import interpolate_trace, smooth_trace
    trace_3d = interpolate_trace(trace_3d, min_valid_ratio=min_valid_ratio)
    trace_3d = smooth_trace(trace_3d, window_size=smooth_window)

    # Recompute target metrics after smoothing
    for i in range(len(trace_3d)):
        pt = trace_3d[i]
        if not np.isnan(pt[0]):
            metrics = {}
            for t in targets:
                label = t["label"]
                dx = t["x"] - pt[0]
                dy = t["y"] - pt[1]
                dz = t["z"] - pt[2]
                r = np.sqrt(dx**2 + dy**2 + dz**2)
                theta = np.arccos(dz / r) if r > 0 else 0.0
                phi = np.arctan2(dy, dx)
                metrics[f"{label}_r"] = r
                metrics[f"{label}_theta"] = theta
                metrics[f"{label}_phi"] = phi
            per_frame_target_metrics[i] = metrics
        else:
            per_frame_target_metrics[i] = {}

    return {
        "trace_3d": trace_3d,
        "per_frame_target_metrics": per_frame_target_metrics,
        "fps": fps,
        "n_frames": n_frames,
        "start_frame_idx": start_frame_idx,
        "mask_quality": mask_quality,
    }


def save_mask_based_csv(result: dict, targets: list, output_path: Path):
    """Save mask-based trace results as CSV compatible with optimal path deviation."""
    import csv

    trace_3d = result["trace_3d"]
    metrics = result["per_frame_target_metrics"]
    fps = result["fps"]
    start_idx = result["start_frame_idx"]
    n_frames = result["n_frames"]

    # Build CSV header (compatible with existing pipeline)
    target_labels = sorted(set(
        k.rsplit("_", 1)[0] for m in metrics if m for k in m.keys()
    ))

    fieldnames = ["frame_index", "time_sec", "local_frame_index"]
    for tl in target_labels:
        fieldnames.extend([f"{tl}_r", f"{tl}_theta", f"{tl}_phi"])
    fieldnames.extend(["trace3d_x", "trace3d_y", "trace3d_z"])

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_frames):
            row = {
                "frame_index": start_idx + i,
                "time_sec": round(i / fps, 4) if fps > 0 else 0,
                "local_frame_index": i,
            }
            # Target metrics
            if i < len(metrics) and metrics[i]:
                for k, v in metrics[i].items():
                    if k in fieldnames:
                        row[k] = round(v, 6)
            # Trace 3D
            if i < len(trace_3d):
                pt = trace_3d[i]
                row["trace3d_x"] = round(pt[0], 6) if not np.isnan(pt[0]) else ""
                row["trace3d_y"] = round(pt[1], 6) if not np.isnan(pt[1]) else ""
                row["trace3d_z"] = round(pt[2], 6) if not np.isnan(pt[2]) else ""

            writer.writerow(row)

    print(f"    Saved CSV: {output_path.name} ({n_frames} rows)")


def save_mask_based_plots(result: dict, targets: list, output_dir: Path):
    """Generate trace and distance plots from mask-based results."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    trace_3d = result["trace_3d"]
    metrics = result["per_frame_target_metrics"]
    fps = result["fps"]
    n_frames = result["n_frames"]

    trace_arr = np.array(trace_3d, dtype=np.float32)

    # ── 3D trace plot ──
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    valid = ~np.isnan(trace_arr[:, 0])
    if np.any(valid):
        xs = trace_arr[valid, 0]
        ys = trace_arr[valid, 1]
        zs = trace_arr[valid, 2]
        colors = cm.viridis(np.linspace(0, 1, len(xs)))
        ax.scatter(xs, zs, ys, c=colors, s=8, alpha=0.7)
        ax.plot(xs, zs, ys, color="gray", alpha=0.3, linewidth=0.5)
        # Mark start/end
        ax.scatter([xs[0]], [zs[0]], [ys[0]], c="green", s=60, marker="^", label="Start")
        ax.scatter([xs[-1]], [zs[-1]], [ys[-1]], c="red", s=60, marker="v", label="End")
    # Plot targets
    for t in targets:
        ax.scatter([t["x"]], [t["z"]], [t["y"]], c="orange", s=100, marker="*")
        ax.text(t["x"], t["z"], t["y"], f"  {t['label']}", fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z / Depth (m)")
    ax.set_zlabel("Y (m)")
    ax.set_title("Baby 3D Trace (Mask-Based)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / "processed_subject_result_trace3d.png"), dpi=100)
    plt.close()

    # ── 2D trace plot (X-Z top-down) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    if np.any(valid):
        times = np.arange(n_frames)[valid] / fps if fps > 0 else np.arange(np.sum(valid))
        ax.scatter(trace_arr[valid, 0], trace_arr[valid, 2], c=times, cmap="viridis", s=8)
        ax.plot(trace_arr[valid, 0], trace_arr[valid, 2], "gray", alpha=0.3, linewidth=0.5)
    for t in targets:
        ax.plot(t["x"], t["z"], "r*", markersize=15)
        ax.annotate(t["label"], (t["x"], t["z"]), fontsize=8)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z / Depth (m)")
    ax.set_title("Baby Trace (Top-Down, Mask-Based)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(str(output_dir / "processed_subject_result_trace.png"), dpi=100)
    plt.close()

    # ── Distance comparison plot ──
    target_labels = sorted(set(
        k.rsplit("_", 1)[0] for m in metrics if m for k in m.keys()
        if k.endswith("_r")
    ))
    if target_labels:
        fig, ax = plt.subplots(figsize=(12, 5))
        times = np.arange(n_frames) / fps if fps > 0 else np.arange(n_frames)
        for tl in target_labels:
            vals = [m.get(f"{tl}_r", np.nan) if m else np.nan for m in metrics]
            ax.plot(times, vals, label=tl, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Distance to Targets")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(str(output_dir / "processed_subject_result_distance_comparison.png"),
                    dpi=100)
        plt.close()

    print(f"    Saved plots to {output_dir.name}")


# ─────────────────────────────────────────────────────────────
#  Zip extraction helpers
# ─────────────────────────────────────────────────────────────
def extract_subject_from_zip(zip_path: Path, extract_to: Path):
    """Extract a subject zip to a working directory."""
    print(f"  Extracting {zip_path.name}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    size_gb = sum(f.stat().st_size for f in extract_to.rglob("*") if f.is_file()) / (1024**3)
    print(f"  Extracted {size_gb:.1f} GB")


def _resolve_trial_layout(trial_dir: Path) -> dict:
    """
    Detect trial data layout and return normalized paths.

    Supports two layouts:
      Old: trial_dir/Color/ + trial_dir/Depth/ (.raw files)
      New: trial_dir/cam1/color/ + trial_dir/cam1/depth/ (.npy files)

    Returns dict with keys: color_dir, depth_dir, depth_format ("raw" or "npy"),
    or None if no valid layout found.
    """
    # Old layout: Color/ + Depth/ with .png + .raw
    color_dir = trial_dir / "Color"
    depth_dir = trial_dir / "Depth"
    if color_dir.is_dir() and depth_dir.is_dir():
        raw_files = [f for f in depth_dir.iterdir()
                     if f.suffix == ".raw" and not f.name.startswith("._")]
        if raw_files:
            return {"color_dir": color_dir, "depth_dir": depth_dir,
                    "depth_format": "raw"}

    # New layout: cam1/color/ + cam1/depth/ with .png + .npy
    cam1_dir = trial_dir / "cam1"
    if cam1_dir.is_dir():
        color_dir = cam1_dir / "color"
        depth_dir = cam1_dir / "depth"
        if color_dir.is_dir() and depth_dir.is_dir():
            npy_files = [f for f in depth_dir.iterdir()
                         if f.suffix == ".npy" and not f.name.startswith("._")]
            if npy_files:
                return {"color_dir": color_dir, "depth_dir": depth_dir,
                        "depth_format": "npy"}

    return None


def find_trial_dirs(data_dir: Path) -> list:
    """Find trial directories with color+depth frames.

    Supports:
      - Numbered dirs: 1, 2, 3, ... (old layout)
      - trial_N dirs: trial_0, trial_1, ... (new layout)
    """
    trials = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir():
            continue
        # Match either plain digits or trial_N naming
        is_trial = (d.name.isdigit() or
                    (d.name.startswith("trial_") and d.name[6:].isdigit()))
        if is_trial and _resolve_trial_layout(d) is not None:
            trials.append(d)
    return trials


def find_data_dir(subject_src: Path) -> tuple:
    """
    Find the data directory for a subject.
    Returns (zip_path, data_dir) - one or both may be set.

    Handles:
      - Old layout: subject_src/InnerDir/1/Color/ (zip-extracted, inner subdir)
      - New layout: subject_src/trial_0/cam1/color/ (direct trial folders)
    """
    zip_path = None
    data_dir = None

    if subject_src.is_file() and subject_src.suffix == ".zip":
        return subject_src, None

    zips = list(subject_src.glob("*.zip"))

    # Check for extracted data (inner folder with numbered trial dirs)
    for d in sorted(subject_src.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            if any(sd.name.isdigit() for sd in d.iterdir() if sd.is_dir()):
                data_dir = d
                break

    if data_dir is None and zips:
        zip_path = zips[0]
    elif data_dir is None:
        # Subject dir itself might have trial folders (old: digit names, new: trial_N)
        has_trials = any(
            d.is_dir() and (d.name.isdigit() or
                            (d.name.startswith("trial_") and d.name[6:].isdigit()))
            for d in subject_src.iterdir()
        )
        if has_trials:
            data_dir = subject_src

    return zip_path, data_dir


# ─────────────────────────────────────────────────────────────
#  Copy results to output
# ─────────────────────────────────────────────────────────────
def copy_results_to_output(trial_dir: Path, output_trial_dir: Path):
    """Copy processed results to output directory."""
    output_trial_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "processed_subject_result_table.csv",
        "processed_subject_result_trace.png",
        "processed_subject_result_trace3d.png",
        "processed_subject_result_distance_comparison.png",
        "target_coordinates.json",
    ]

    copied = 0
    for fname in files_to_copy:
        src = trial_dir / fname
        if src.exists():
            shutil.copy2(src, output_trial_dir / fname)
            copied += 1
    return copied


# ─────────────────────────────────────────────────────────────
#  Per-subject processing
# ─────────────────────────────────────────────────────────────
def process_subject(subject_name: str, subject_src: Path,
                    skip_sam: bool = False,
                    skip_mediapipe: bool = False,
                    fixed_targets: bool = False,
                    force_sam: bool = False):
    """
    Process one CCD subject through the full pipeline.

    Returns (ok_count, fail_count).
    """
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

    # Detect targets once per subject (targets are fixed across trials)
    subject_targets = None
    if not fixed_targets:
        print(f"\n  --- Target Detection ---")
        # Try YOLO on the first trial that has good data
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
        color_dir = layout["color_dir"] if layout else trial_dir / "Color"

        print(f"\n  --- Trial {trial_name} ---"
              + (f" [{layout['depth_format']}]" if layout else ""))

        # Step 1: Save target coordinates for this trial
        save_target_coordinates_json(trial_dir, subject_targets)

        # Step 2: SAM segmentation (run in subprocess to isolate crashes)
        seg_dir = trial_dir / "segmented_color"
        masked_video = trial_dir / "masked_video.mp4"
        if force_sam and not skip_sam:
            # Clear old segmentation results to force re-run
            if seg_dir.is_dir():
                shutil.rmtree(seg_dir)
            if masked_video.exists():
                masked_video.unlink()
        if not skip_sam and not (seg_dir.is_dir() and any(seg_dir.iterdir())):
            try:
                success = run_sam_segmentation_subprocess(trial_dir)
                if not success:
                    print(f"    SAM3 failed — will try MediaPipe fallback")
            except Exception as e:
                print(f"    SAM3 error: {e}")

        # Step 3: Try mask-based depth approach (primary — no skeleton needed)
        mask_result = None
        if seg_dir.is_dir() and any(seg_dir.iterdir()):
            mask_result = compute_trace_from_mask_depth(
                trial_dir, intrinsics, subject_targets)

        # Normalize output dir name: trial_1 for "1", trial_0 for "trial_0"
        if trial_name.startswith("trial_"):
            output_trial_name = trial_name
        else:
            output_trial_name = f"trial_{trial_name}"

        if mask_result is not None:
            # Mask-based approach succeeded
            output_trial_dir = output_subject_dir / output_trial_name
            output_trial_dir.mkdir(parents=True, exist_ok=True)

            csv_path = trial_dir / "processed_subject_result_table.csv"
            save_mask_based_csv(mask_result, subject_targets, csv_path)
            save_mask_based_plots(mask_result, subject_targets, trial_dir)

            copied = copy_results_to_output(trial_dir, output_trial_dir)
            print(f"    Copied {copied} result files to output")

            if (output_trial_dir / "processed_subject_result_table.csv").exists():
                ok += 1
            else:
                print(f"    WARNING: No CSV generated")
                fail += 1
        else:
            # Fallback: MediaPipe skeleton-based approach
            print(f"    Using MediaPipe fallback...")

            # Ensure video is available for MediaPipe
            if not masked_video.exists():
                color_mp4 = trial_dir / "Color.mp4"
                if not color_mp4.exists():
                    create_video_from_frames(color_dir, color_mp4)
                masked_video = color_mp4

            if not masked_video.exists():
                print(f"    ERROR: No video available for trial {trial_name}")
                fail += 1
                continue

            skeleton_json = None
            if not skip_mediapipe:
                skeleton_json = run_mediapipe(masked_video)
            else:
                for candidate in [
                    trial_dir / "masked_video_skeleton.json",
                    trial_dir / "Color_skeleton.json",
                ]:
                    if candidate.exists():
                        skeleton_json = candidate
                        print(f"    Reusing skeleton: {candidate.name}")
                        break
                if skeleton_json is None:
                    jsons = list(trial_dir.glob("*skeleton*.json"))
                    if jsons:
                        skeleton_json = jsons[0]
                        print(f"    Reusing skeleton: {skeleton_json.name}")

            if skeleton_json is None or not skeleton_json.exists():
                print(f"    ERROR: No skeleton JSON for trial {trial_name}")
                fail += 1
                continue

            try:
                run_pose_visualize(skeleton_json, side_view=True)
            except Exception as e:
                print(f"    ERROR in pose_visualize: {e}")
                traceback.print_exc()
                fail += 1
                continue

            output_trial_dir = output_subject_dir / output_trial_name
            copied = copy_results_to_output(trial_dir, output_trial_dir)
            print(f"    Copied {copied} result files to output")

            if (output_trial_dir / "processed_subject_result_table.csv").exists():
                ok += 1
            else:
                print(f"    WARNING: No CSV generated")
                fail += 1

    # Step 6: Run optimal path deviation
    if ok > 0:
        print(f"\n  Running optimal path deviation...")
        script = POINTING_DIR / "process_comprehension_optimal_path.py"
        cmd = [sys.executable, str(script), str(OUTPUT_DIR),
               "--subject", subject_name]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  Optimal path warning: {result.stderr[-300:]}")
            else:
                dev_count = sum(1 for d in output_subject_dir.iterdir()
                                if d.is_dir() and (d / "optimal_path_deviation.csv").exists())
                print(f"  Optimal path: {dev_count} trials processed")
        except Exception as e:
            print(f"  Optimal path error: {e}")

    # Step 7: Cleanup extracted data
    if work_dir is not None and work_dir.exists():
        print(f"\n  Cleaning up {work_dir}...")
        shutil.rmtree(work_dir)

    print(f"\n  {subject_name}: {ok} OK, {fail} failed / {len(trials)} trials")
    return ok, fail


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────
def main():
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Batch reprocess CCD Comprehension data through full pipeline")
    parser.add_argument("--subject", help="Process only this subject (prefix match)")
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip SAM segmentation, reuse existing masked_video.mp4")
    parser.add_argument("--skip-mediapipe", action="store_true",
                        help="Skip MediaPipe, reuse existing skeleton JSON")
    parser.add_argument("--fixed-targets", action="store_true",
                        help="Use fixed CCD_TARGETS instead of YOLO detection")
    parser.add_argument("--force-sam", action="store_true",
                        help="Force re-run SAM3, clearing old segmentation results")
    parser.add_argument("--source", type=str, default=str(SSD_SOURCE),
                        help=f"Source directory (default: {SSD_SOURCE})")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    source_dir = Path(args.source)
    OUTPUT_DIR = Path(args.output)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = OUTPUT_DIR

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Find all subjects
    subjects = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if args.subject:
        subjects = [d for d in subjects if d.name.startswith(args.subject)]

    mode_parts = []
    if args.skip_sam:
        mode_parts.append("skip-SAM")
    else:
        mode_parts.append("SAM3 (baby→person, motion-based)")
    if args.force_sam:
        mode_parts.append("force-SAM")
    if args.skip_mediapipe:
        mode_parts.append("skip-MediaPipe")
    else:
        mode_parts.append("MediaPipe")
    if args.fixed_targets:
        mode_parts.append("fixed targets")
    else:
        mode_parts.append("YOLO targets")

    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Subjects: {len(subjects)}")
    print(f"Mode: {' + '.join(mode_parts)}")
    print(f"YOLO model: {YOLO_MODEL}")
    print()

    total_ok = 0
    total_fail = 0

    for i, subject_dir in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] {subject_dir.name}")
        ok, fail = process_subject(
            subject_dir.name,
            subject_dir,
            skip_sam=args.skip_sam,
            skip_mediapipe=args.skip_mediapipe,
            fixed_targets=args.fixed_targets,
            force_sam=args.force_sam,
        )
        total_ok += ok
        total_fail += fail

    # Generate global CSVs
    if total_ok > 0:
        print(f"\n{'='*60}")
        print("Generating global CSVs...")
        script = POINTING_DIR / "process_comprehension_optimal_path.py"
        cmd = [sys.executable, str(script), str(output_dir)]
        try:
            subprocess.run(cmd, timeout=300)
        except Exception as e:
            print(f"Global CSV error: {e}")

    print(f"\n{'='*60}")
    print(f"DONE: {total_ok} trials OK, {total_fail} failed")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

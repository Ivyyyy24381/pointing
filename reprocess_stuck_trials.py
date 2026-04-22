#!/usr/bin/env python3
"""
Reprocess stuck CCD comprehension trials using manual point prompts.

Reads baby positions from stuck_trial_labels.json (created by label_stuck_trials.py)
and re-runs SAM3 segmentation with point prompts instead of text prompts.

Usage:
    python reprocess_stuck_trials.py
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

POINTING_DIR = Path("/home/tigerli/Documents/GitHub/pointing")
SSD_DIR = Path("/media/tigerli/Extreme SSD/pointing_data/point_comprehension_CCD")
OUTPUT_DIR = Path("/home/tigerli/Documents/pointing_data/point_comprehension_CCD_output")
WORK_DIR = Path("/home/tigerli/Documents/pointing_data/_ccd_work")
LABELS_FILE = POINTING_DIR / "stuck_trial_labels.json"

sys.path.insert(0, str(POINTING_DIR))
sys.path.insert(0, str(POINTING_DIR / "step3_subject_extraction" / "dog_script"))
sys.path.insert(0, str(POINTING_DIR / "step0_data_loading"))


def run_sam_with_box_prompt(trial_dir, box_x, box_y, box_w, box_h, img_w, img_h, frame_idx):
    """
    Run SAM3 segmentation using the SAM2-compatible tracker API with a box prompt.

    Uses build_sam3_video_model().tracker directly (as shown in the official
    sam3_for_sam2_video_task_example.ipynb), NOT the SAM3 detector handle_request API.
    Box prompt format: [[xmin, ymin, xmax, ymax]] in relative coords (0-1).
    """
    script = f'''
import sys, os
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, "{POINTING_DIR}")
sys.path.insert(0, "{POINTING_DIR / "step3_subject_extraction" / "dog_script"}")
sys.path.insert(0, "{POINTING_DIR / "step0_data_loading"}")

import tempfile
import cv2
import numpy as np
from pathlib import Path

trial_dir = Path("{trial_dir}")

# Resolve color directory
from batch_reprocess_CCD import _resolve_trial_layout
layout = _resolve_trial_layout(trial_dir)
color_dir = layout["color_dir"] if layout else trial_dir / "Color"
color_files = sorted([f for f in color_dir.iterdir()
                      if f.suffix == ".png" and not f.name.startswith("._")])
if not color_files:
    print("SAM3_RESULT:False")
    sys.exit(0)

print(f"  {{len(color_files)}} frames")

# Labeled bounding box in pixels (x, y, w, h)
box_x, box_y, box_w, box_h = {box_x}, {box_y}, {box_w}, {box_h}
label_frame = {frame_idx}

# Read image dimensions from actual frame
first_img = cv2.imread(str(color_files[0]))
img_h, img_w = first_img.shape[:2]

# Convert box from xywh pixels to xyxy relative coords (0-1)
rel_xmin = box_x / img_w
rel_ymin = box_y / img_h
rel_xmax = (box_x + box_w) / img_w
rel_ymax = (box_y + box_h) / img_h
# Clamp to [0, 1]
rel_xmin = max(0.0, min(rel_xmin, 1.0))
rel_ymin = max(0.0, min(rel_ymin, 1.0))
rel_xmax = max(0.0, min(rel_xmax, 1.0))
rel_ymax = max(0.0, min(rel_ymax, 1.0))
print(f"  Box: [{{rel_xmin:.3f}}, {{rel_ymin:.3f}}, {{rel_xmax:.3f}}, {{rel_ymax:.3f}}] (xyxy rel) on frame {{label_frame}}")

import torch
from sam3.model_builder import build_sam3_video_model

# Build SAM3 model and use the SAM2-compatible tracker directly
print("  Loading SAM3 model...")
sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone
print("  Model loaded")

with tempfile.TemporaryDirectory() as tmpdir:
    jpeg_dir = Path(tmpdir) / "frames"
    jpeg_dir.mkdir()

    print(f"  Converting {{len(color_files)}} frames to JPEG...")
    for i, cf in enumerate(color_files):
        img = cv2.imread(str(cf))
        if img is not None:
            cv2.imwrite(str(jpeg_dir / f"{{i:06d}}.jpg"), img)

    # Initialize tracker inference state
    print("  Initializing inference state...")
    inference_state = predictor.init_state(
        video_path=str(jpeg_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    )

    # Add box prompt on the labeled frame using SAM2 tracker API
    # Box format: [[xmin, ymin, xmax, ymax]] in relative coords
    rel_box = np.array([[rel_xmin, rel_ymin, rel_xmax, rel_ymax]], dtype=np.float32)
    ann_obj_id = 1

    print(f"  Adding box prompt on frame {{label_frame}}...")
    _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=label_frame,
        obj_id=ann_obj_id,
        box=rel_box,
    )
    print(f"  Box prompt: {{len(out_obj_ids)}} object(s), mask shape: {{video_res_masks.shape}}")

    # Check if initial mask is good
    init_mask = (video_res_masks[0] > 0.0).cpu().numpy().squeeze()
    print(f"  Initial mask: {{init_mask.sum()}} pixels, {{init_mask.shape}}")

    if init_mask.sum() == 0:
        print("  WARNING: Empty initial mask from box prompt")
        print("SAM3_RESULT:False")
        sys.exit(0)

    # Propagate through video
    print("  Propagating...")
    video_segments = {{}}
    for frame_idx_out, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state, start_frame_idx=0, max_frame_num_to_track=len(color_files),
        reverse=False, propagate_preflight=True
    ):
        video_segments[frame_idx_out] = {{
            out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(obj_ids)
        }}

    print(f"  Propagated {{len(video_segments)}} frames")
    non_empty = sum(1 for v in video_segments.values()
                    if any(m.sum() > 0 for m in v.values()))
    print(f"  Non-empty masks: {{non_empty}}")

# Build frame_masks dict
frame_masks = {{}}
for fidx, seg_data in video_segments.items():
    mask = seg_data.get(ann_obj_id)
    if mask is not None:
        mask = mask.squeeze()
        if mask.ndim == 2 and mask.sum() > 0:
            frame_masks[fidx] = mask.astype(np.uint8)

detected = len(frame_masks)
total = len(color_files)
print(f"  SAM3: Segmented in {{detected}}/{{total}} frames")

if detected == 0:
    print("SAM3_RESULT:False")
    sys.exit(0)

# Create masked_video.mp4 and segmented_color/
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
    cv2.imwrite(str(seg_out_dir / f"vis_frame_{{i:06d}}.png"), masked)

writer.release()
print("SAM3_RESULT:True")
sys.stdout.flush()
'''

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300,
        cwd=str(POINTING_DIR),
    )

    stdout = result.stdout
    stderr = result.stderr
    for line in stdout.split("\n"):
        if line.strip():
            print(f"    {line.strip()}")

    if stderr:
        # Print last few lines of stderr for debugging
        err_lines = [l for l in stderr.split("\n") if l.strip()]
        for line in err_lines[-10:]:
            print(f"    STDERR: {line.strip()}")

    if "SAM3_RESULT:True" in stdout:
        return True

    # Check if outputs exist even if process crashed
    seg_dir = trial_dir / "segmented_color"
    if seg_dir.is_dir() and any(seg_dir.iterdir()):
        print(f"    SAM3 subprocess crashed but segmented_color exists")
        return True

    return False


def find_trial_dir_on_ssd(subject, trial):
    """Find the trial directory in the SSD source data."""
    ssd_subj = SSD_DIR / subject
    if not ssd_subj.is_dir():
        return None

    trial_num = trial.replace("trial_", "")

    # Check various layouts
    candidates = [
        ssd_subj / trial_num,
        ssd_subj / trial,
    ]
    for inner in ssd_subj.iterdir():
        if inner.is_dir() and not inner.name.startswith("."):
            candidates.extend([
                inner / trial_num,
                inner / trial,
            ])

    for d in candidates:
        if d.is_dir():
            # Verify it has color frames
            from batch_reprocess_CCD import _resolve_trial_layout
            if _resolve_trial_layout(d) is not None:
                return d

    return None


def extract_trial_from_zip(subject, trial):
    """Extract just one trial from a subject's zip to work dir."""
    ssd_subj = SSD_DIR / subject
    zips = list(ssd_subj.glob("*.zip"))
    if not zips:
        return None

    trial_num = trial.replace("trial_", "")
    work_subj = WORK_DIR / subject
    work_subj.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zips[0]) as zf:
        # Find members for this trial
        trial_members = [m for m in zf.namelist()
                        if f"/{trial_num}/" in m or m.startswith(f"{trial_num}/")]
        if not trial_members:
            # Try inner folder
            all_members = zf.namelist()
            trial_members = [m for m in all_members
                           if f"/{trial_num}/" in m]

        if trial_members:
            print(f"    Extracting {len(trial_members)} files for trial {trial_num}")
            zf.extractall(work_subj, members=trial_members)

    # Find the extracted trial dir
    from batch_reprocess_CCD import _resolve_trial_layout
    for d in work_subj.rglob("*"):
        if d.is_dir() and d.name == trial_num:
            if _resolve_trial_layout(d) is not None:
                return d

    return None


def main():
    if not LABELS_FILE.exists():
        print(f"ERROR: No labels file found at {LABELS_FILE}")
        print(f"Run  python label_stuck_trials.py  first to label baby positions")
        sys.exit(1)

    with open(LABELS_FILE) as f:
        labels = json.load(f)

    print(f"Loaded {len(labels)} labels from {LABELS_FILE.name}")
    print(f"Output: {OUTPUT_DIR}\n")

    from batch_reprocess_CCD import (
        load_intrinsics, detect_targets_yolo,
        compute_trace_from_mask_depth, save_mask_based_csv,
        save_mask_based_plots, copy_results_to_output,
        CCD_TARGETS_FIXED, _resolve_trial_layout,
    )

    ok = 0
    fail = 0

    for key, label in labels.items():
        subject, trial = key.split("/")
        frame_idx = label["frame_idx"]

        # Support both old (click) and new (bbox) label formats
        if "box_x" in label:
            lbl_bx = label["box_x"]
            lbl_by = label["box_y"]
            lbl_bw = label["box_w"]
            lbl_bh = label["box_h"]
            lbl_img_w = label.get("img_w", 640)
            lbl_img_h = label.get("img_h", 480)
        else:
            # Old format: click point → create box around it
            cx, cy = label["click_x"], label["click_y"]
            lbl_img_w = label.get("img_w", 640)
            lbl_img_h = label.get("img_h", 480)
            lbl_bw = int(lbl_img_w * 0.20)
            lbl_bh = int(lbl_img_h * 0.25)
            lbl_bx = max(0, cx - lbl_bw // 2)
            lbl_by = max(0, cy - lbl_bh // 2)

        print(f"\n{'='*50}")
        print(f"{subject} / {trial}")
        print(f"  Box: ({lbl_bx},{lbl_by}) {lbl_bw}x{lbl_bh} on frame {frame_idx}")

        # Find trial data
        trial_dir = find_trial_dir_on_ssd(subject, trial)
        needs_cleanup = False

        if trial_dir is None:
            # Try extracting from zip
            trial_dir = extract_trial_from_zip(subject, trial)
            needs_cleanup = True

        if trial_dir is None:
            print(f"  ERROR: Cannot find trial data")
            fail += 1
            continue

        print(f"  Data: {trial_dir}")

        # Clear old segmentation
        seg_dir = trial_dir / "segmented_color"
        masked_video = trial_dir / "masked_video.mp4"
        if seg_dir.is_dir():
            shutil.rmtree(seg_dir)
        if masked_video.exists():
            masked_video.unlink()

        # Run SAM3 with box prompt
        success = run_sam_with_box_prompt(
            trial_dir, lbl_bx, lbl_by, lbl_bw, lbl_bh,
            lbl_img_w, lbl_img_h, frame_idx)

        if not success:
            print(f"  SAM3 box prompt FAILED")
            fail += 1
            if needs_cleanup:
                shutil.rmtree(WORK_DIR / subject, ignore_errors=True)
            continue

        # Load intrinsics and targets
        data_dir = trial_dir.parent
        intrinsics = load_intrinsics(data_dir)

        # Detect or use fixed targets
        targets = detect_targets_yolo(trial_dir, intrinsics)
        if not targets or len(targets) < 2:
            targets = CCD_TARGETS_FIXED

        # Compute mask-based trace
        mask_result = compute_trace_from_mask_depth(
            trial_dir, intrinsics, targets)

        if mask_result is None:
            print(f"  Mask-based trace FAILED")
            fail += 1
            if needs_cleanup:
                shutil.rmtree(WORK_DIR / subject, ignore_errors=True)
            continue

        # Save CSV and plots
        csv_path = trial_dir / "processed_subject_result_table.csv"
        save_mask_based_csv(mask_result, targets, csv_path)
        save_mask_based_plots(mask_result, targets, trial_dir)

        # Copy to output
        output_trial_dir = OUTPUT_DIR / subject / trial
        output_trial_dir.mkdir(parents=True, exist_ok=True)
        copied = copy_results_to_output(trial_dir, output_trial_dir)
        print(f"  Copied {copied} files to output")

        # Update consolidated folders
        traces_dir = OUTPUT_DIR / "_all_baby_traces"
        traces3d_dir = OUTPUT_DIR / "_all_traces"
        distance_dir = OUTPUT_DIR / "_all_distance_plots"
        prefix = f"{subject}__{trial}"

        trace_src = output_trial_dir / "processed_subject_result_trace.png"
        if trace_src.exists():
            shutil.copy2(trace_src, traces_dir / f"{prefix}.png")

        trace3d_src = output_trial_dir / "processed_subject_result_trace3d.png"
        if trace3d_src.exists():
            shutil.copy2(trace3d_src, traces3d_dir / f"{prefix}.png")

        dist_src = output_trial_dir / "processed_subject_result_distance_comparison.png"
        if dist_src.exists():
            shutil.copy2(dist_src, distance_dir / f"{prefix}.png")

        ok += 1
        print(f"  OK")

        # Cleanup extracted data
        if needs_cleanup:
            shutil.rmtree(WORK_DIR / subject, ignore_errors=True)

    # Regenerate global CSVs
    if ok > 0:
        print(f"\nRegenerating global CSVs...")
        script = POINTING_DIR / "process_comprehension_optimal_path.py"
        try:
            subprocess.run([sys.executable, str(script), str(OUTPUT_DIR)],
                         timeout=300)
        except Exception as e:
            print(f"Global CSV error: {e}")

    print(f"\n{'='*50}")
    print(f"DONE: {ok} OK, {fail} failed out of {len(labels)} labeled trials")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

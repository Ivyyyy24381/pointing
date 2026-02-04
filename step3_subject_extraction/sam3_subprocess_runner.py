"""
Standalone SAM3 detection script that runs in a separate process.

This avoids C-level double-free crashes caused by conflicts between
SAM3's CUDA operations and mediapipe/tensorflow already loaded in
the parent process.

Usage:
    python -m step3_subject_extraction.sam3_subprocess_runner \
        --cam-path /path/to/cam1 \
        --output-path /path/to/output \
        --subject-type dog \
        --fx 615 --fy 615 --cx 320 --cy 240
"""

import argparse
import json
import sys
import numpy as np
import cv2
from pathlib import Path


def run_sam3_detection(cam_path, output_path, subject_type='dog',
                       fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    """Run SAM3 detection in this (isolated) process."""
    cam_path = Path(cam_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    color_folder = cam_path / "color"
    depth_folder = cam_path / "depth"
    color_images = sorted(color_folder.glob("frame_*.png"))

    if not color_images:
        print(f"ERROR: No color frames found in {color_folder}")
        return False

    # Load SAM3 (no mediapipe/tf loaded in this process)
    from step3_subject_extraction.sam3_detector import SAM3Detector
    detector = SAM3Detector()

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
                if depth_npy.exists():
                    depth_img = np.load(str(depth_npy))
                if depth_img is not None and depth_img.dtype == np.uint16:
                    depth_img = depth_img.astype(np.float32) / 1000.0
            except Exception:
                pass

        all_frames.append(color_rgb)
        all_frame_numbers.append(frame_num)
        all_depth_images.append(depth_img)

    # Run detection
    results_dict = detector.process_batch(
        all_frames, all_frame_numbers,
        text_prompt=subject_type,
        crop_lower=True, crop_ratio=0.6,
        depth_images=all_depth_images,
        fx=fx, fy=fy, cx=cx, cy=cy,
    )

    detected_count = len(results_dict)
    print(f"SAM3 {subject_type} detected in {detected_count}/{len(all_frames)} frames")

    if not results_dict:
        return False

    # Save results JSON
    json_data = {}
    for frame_key, result in results_dict.items():
        json_data[frame_key] = result.to_dict()

    json_path = output_path / f"{subject_type}_detection_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved SAM3 {subject_type} results: {json_path.name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="SAM3 detection subprocess")
    parser.add_argument("--cam-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--subject-type", default="dog")
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    args = parser.parse_args()

    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    success = run_sam3_detection(
        args.cam_path, args.output_path,
        subject_type=args.subject_type,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
    )
    # Use os._exit() to skip Python cleanup — avoids CUDA double-free crash
    # that occurs when SAM3/torch tensors are garbage-collected at shutdown
    import os
    os._exit(0 if success else 1)


if __name__ == "__main__":
    main()

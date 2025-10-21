#!/usr/bin/env python3
"""
Batch dog detection processor for trial_input → trial_output workflow.

This module processes all frames from trial_input folder using DeepLabCut
batch processing and saves results to trial_output folder.

Usage:
    from step3_subject_extraction.batch_dog_detection import process_trial_dog_detection

    results = process_trial_dog_detection(
        trial_input_path="trial_input/1/single_camera",
        trial_output_path="trial_output/1/single_camera"
    )
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from subject_detector import SubjectDetector, SubjectDetectionResult


def process_trial_dog_detection(
    trial_input_path: str,
    trial_output_path: str,
    crop_ratio: float = 0.5,
    fx: float = 615.0,
    fy: float = 615.0,
    cx: float = 320.0,
    cy: float = 240.0
) -> Dict[str, SubjectDetectionResult]:
    """
    Process dog detection for entire trial using batch processing.

    Args:
        trial_input_path: Path to trial_input/X/single_camera folder
        trial_output_path: Path to trial_output/X/single_camera folder
        crop_ratio: Ratio of lower image to process (0.5 = bottom 50%)
        fx, fy: Focal lengths
        cx, cy: Principal point

    Returns:
        Dictionary mapping frame_keys to SubjectDetectionResult
    """
    trial_input = Path(trial_input_path)
    trial_output = Path(trial_output_path)

    print(f"\n{'='*70}")
    print(f"🐕 BATCH DOG DETECTION - Trial Processing")
    print(f"{'='*70}")
    print(f"Input:  {trial_input}")
    print(f"Output: {trial_output}")

    # Create output directory
    trial_output.mkdir(parents=True, exist_ok=True)

    # Find color frames
    color_dir = trial_input / "color"
    if not color_dir.exists():
        print(f"❌ No color directory found at: {color_dir}")
        return {}

    color_images = sorted(color_dir.glob("frame_*.png"))
    if not color_images:
        print(f"❌ No frame_*.png images found in: {color_dir}")
        return {}

    print(f"✅ Found {len(color_images)} frames to process")

    # Load all frames
    print(f"\n📂 Loading frames...")
    frames = []
    frame_numbers = []

    for i, img_path in enumerate(color_images):
        img = cv2.imread(str(img_path))
        if img is not None:
            frames.append(img)
            # Extract frame number from filename: frame_000123.png → 123
            frame_num = int(img_path.stem.split('_')[1])
            frame_numbers.append(frame_num)

        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(color_images)} frames...")

    print(f"✅ Loaded {len(frames)} frames")

    if len(frames) == 0:
        print(f"❌ No valid frames loaded")
        return {}

    # Initialize detector
    print(f"\n🐶 Initializing dog detector (crop_ratio={crop_ratio})...")
    detector = SubjectDetector(subject_type='dog', crop_ratio=crop_ratio)

    # Create output video path in trial_output
    output_video = trial_output / "dog_detection_cropped_video.mp4"

    # Load depth images if available
    depth_images = None
    depth_dir = trial_input / "depth"
    if depth_dir.exists():
        depth_files = sorted(depth_dir.glob("frame_*.png"))
        if len(depth_files) == len(frames):
            print(f"📊 Loading depth images...")
            depth_images = []
            for depth_file in depth_files:
                depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                depth_images.append(depth_img)
            print(f"✅ Loaded {len(depth_images)} depth images")

    # Process batch
    print(f"\n{'='*70}")
    print(f"🎬 Running batch dog detection...")
    print(f"{'='*70}")

    results = detector.process_batch_video(
        output_video_path=str(output_video),
        frames=frames,
        frame_numbers=frame_numbers,
        depth_images=depth_images,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    # Save results to JSON
    if results:
        output_json = trial_output / "dog_detection_results.json"
        results_dict = {k: v.to_dict() for k, v in results.items()}

        with open(output_json, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n💾 Saved results:")
        print(f"   JSON: {output_json}")
        print(f"   Video: {output_video}")
        print(f"   Detections: {len(results)}/{len(frames)} frames ({len(results)/len(frames)*100:.1f}%)")
    else:
        print(f"\n⚠️  No dogs detected in any frame")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch dog detection for trials")
    parser.add_argument("--input", required=True, help="Path to trial_input/X/single_camera")
    parser.add_argument("--output", required=True, help="Path to trial_output/X/single_camera")
    parser.add_argument("--crop-ratio", type=float, default=0.5, help="Crop ratio (default: 0.5 = bottom 50%%)")
    parser.add_argument("--fx", type=float, default=615.0, help="Focal length X")
    parser.add_argument("--fy", type=float, default=615.0, help="Focal length Y")
    parser.add_argument("--cx", type=float, default=320.0, help="Principal point X")
    parser.add_argument("--cy", type=float, default=240.0, help="Principal point Y")

    args = parser.parse_args()

    results = process_trial_dog_detection(
        trial_input_path=args.input,
        trial_output_path=args.output,
        crop_ratio=args.crop_ratio,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy
    )

    print(f"\n{'='*70}")
    print(f"✅ Processing complete!")
    print(f"{'='*70}\n")

#!/usr/bin/env python3
"""
Test batch dog detection using DeepLabCut on full trial.

Usage:
    /opt/anaconda3/envs/point_production/bin/python test_batch_dog_detection.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from step3_subject_extraction.subject_detector import SubjectDetector


def test_batch_dog_detection(trial_path: str):
    """Test batch dog detection on entire trial."""
    trial_path = Path(trial_path)

    print(f"\n{'='*70}")
    print(f"🐕 BATCH DOG DETECTION TEST")
    print(f"{'='*70}")
    print(f"Trial path: {trial_path}")

    # Find color images
    color_dir = trial_path / "color"
    if not color_dir.exists():
        color_dir = trial_path / "Color"

    if not color_dir.exists():
        print(f"❌ No color/Color directory found")
        return

    # Load all frames
    color_images = sorted(color_dir.glob("*.png"))
    if not color_images:
        print(f"❌ No PNG images found")
        return

    print(f"✅ Found {len(color_images)} frames")

    # Initialize detector
    print(f"\nInitializing SubjectDetector for 'dog'...")
    detector = SubjectDetector(subject_type='dog', crop_ratio=0.5)

    # Load all frames
    print(f"\nLoading frames...")
    frames = []
    frame_numbers = []

    for i, img_path in enumerate(color_images):
        img = cv2.imread(str(img_path))
        if img is not None:
            frames.append(img)
            frame_numbers.append(i + 1)

        if i % 50 == 0:
            print(f"  Loaded {i+1}/{len(color_images)} frames...")

    print(f"✅ Loaded {len(frames)} frames")

    # Process as batch
    output_video = trial_path / "temp_cropped_dog_video.mp4"

    print(f"\n{'='*70}")
    print(f"Running batch processing...")
    print(f"{'='*70}")

    results = detector.process_batch_video(
        output_video_path=str(output_video),
        frames=frames,
        frame_numbers=frame_numbers,
        depth_images=None,
        fx=922.5,  # For 1280x720
        fy=922.5,
        cx=640.0,
        cy=360.0
    )

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total frames processed: {len(frames)}")
    print(f"Frames with dog detected: {len(results)}")
    print(f"Detection rate: {len(results)/len(frames)*100:.1f}%")

    # Show sample results
    if results:
        print(f"\nSample detections:")
        for i, (frame_key, result) in enumerate(list(results.items())[:5]):
            print(f"\n  {frame_key}:")
            print(f"    Subject: {result.subject_type}")
            print(f"    BBox: {result.bbox}")
            print(f"    Keypoints: {len(result.keypoints_2d) if result.keypoints_2d else 0}")

        # Save results to JSON
        import json
        output_json = trial_path / "batch_dog_detection_results.json"
        results_dict = {k: v.to_dict() for k, v in results.items()}

        with open(output_json, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\n💾 Saved results to: {output_json}")
    else:
        print(f"\n⚠️ No dogs detected in any frame")


if __name__ == "__main__":
    # Test path
    test_path = "/Users/ivy/Downloads/dog_data/BDL049_Star_side_cam/1"

    test_batch_dog_detection(test_path)

    print(f"\n{'='*70}")
    print(f"Test complete!")
    print(f"{'='*70}\n")

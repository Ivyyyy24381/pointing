#!/usr/bin/env python3
"""
Test script for dog detection using DeepLabCut SuperAnimal.

Usage:
    python test_dog_detection.py
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from step3_subject_extraction.subject_detector import SubjectDetector


def test_dog_detection(trial_path: str):
    """Test dog detection on a trial."""
    trial_path = Path(trial_path)

    print(f"\n{'='*70}")
    print(f"🐕 TESTING DOG DETECTION")
    print(f"{'='*70}")
    print(f"Trial path: {trial_path}")

    # Check if path exists
    if not trial_path.exists():
        print(f"❌ Path does not exist: {trial_path}")
        return

    # Find color images (try both lowercase and uppercase)
    color_dir = trial_path / "color"
    if not color_dir.exists():
        color_dir = trial_path / "Color"

    if not color_dir.exists():
        print(f"❌ No color/Color directory found at: {trial_path}")
        return

    # Try different naming patterns
    color_images = sorted(color_dir.glob("frame_*.png"))
    if not color_images:
        color_images = sorted(color_dir.glob("_Color_*.png"))
    if not color_images:
        color_images = sorted(color_dir.glob("*.png"))

    if not color_images:
        print(f"❌ No PNG images found in: {color_dir}")
        return

    print(f"✅ Found {len(color_images)} frames")

    # Initialize dog detector
    print(f"\n{'='*70}")
    print(f"Initializing SubjectDetector for 'dog'...")
    print(f"{'='*70}")

    detector = SubjectDetector(subject_type='dog', crop_ratio=0.5)

    # Test on multiple frames to find where dog is visible
    test_frames = [0, 50, 100, 150, 200]  # Try frames throughout the video

    for test_frame_idx in test_frames:
        if test_frame_idx >= len(color_images):
            continue

        test_frame_path = color_images[test_frame_idx]

        print(f"\n{'='*70}")
        print(f"Testing frame {test_frame_idx}: {test_frame_path.name}")
        print(f"{'='*70}")

        # Load image
        image = cv2.imread(str(test_frame_path))
        if image is None:
            print(f"❌ Failed to load image: {test_frame_path}")
            continue

        h, w = image.shape[:2]
        print(f"Image size: {w}x{h}")

        # Run detection
        print(f"\nRunning detection...")
        result = detector.detect_frame(
            image=image,
            frame_number=test_frame_idx + 1,
            depth_image=None,
            fx=922.5,  # For 1280x720
            fy=922.5,
            cx=640.0,
            cy=360.0
        )

        print(f"\n{'='*70}")
        print(f"DETECTION RESULT - Frame {test_frame_idx}")
        print(f"{'='*70}")

        if result is None:
            print(f"❌ No detection result (returned None)")
        else:
            print(f"✅ Detection successful!")
            print(f"   Subject type: {result.subject_type}")
            print(f"   Detection region: {result.detection_region}")
            print(f"   Bounding box: {result.bbox}")
            print(f"   Number of 2D keypoints: {len(result.keypoints_2d) if result.keypoints_2d else 0}")
            print(f"   Number of 3D keypoints: {len(result.keypoints_3d) if result.keypoints_3d else 0}")

            # Show first few keypoints
            if result.keypoints_2d:
                print(f"\n   First 5 keypoints (2D):")
                for i, kp in enumerate(result.keypoints_2d[:5]):
                    print(f"     [{i}] x={kp[0]:.1f}, y={kp[1]:.1f}, conf={kp[2]:.3f}")

            print(f"\n🎉 Found dog! Stopping test.")
            break

        print(f"\n   Trying next frame...")


if __name__ == "__main__":
    # Test path - use trial 1
    test_path = "/Users/ivy/Downloads/dog_data/BDL049_Star_side_cam/1"

    test_dog_detection(test_path)

    print(f"\n{'='*70}")
    print(f"Test complete!")
    print(f"{'='*70}\n")

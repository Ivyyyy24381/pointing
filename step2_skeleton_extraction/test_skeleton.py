#!/usr/bin/env python3
"""
Quick test script for skeleton extraction.

Tests MediaPipe detector on sample trial data.
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector


def test_mediapipe():
    """Test MediaPipe detector."""
    print("="*60)
    print("Testing MediaPipe Human Pose Detector")
    print("="*60)

    # Check if trial_input exists
    trial_input = Path("trial_input")
    if not trial_input.exists():
        print("❌ trial_input/ not found")
        print("   Run Step 0 data loading first")
        return False

    # Find first trial with color images (works with both old and new structure)
    test_trial = None
    for trial_dir in sorted(trial_input.iterdir()):
        if not trial_dir.is_dir():
            continue

        # Check if this is directly a trial (old structure: trial_1_cam1/)
        color_folder = trial_dir / "color"
        if color_folder.exists() and list(color_folder.glob("*.png")):
            test_trial = trial_dir
            break

        # Check subdirectories (new structure: trial_1/cam1/)
        for camera_dir in sorted(trial_dir.iterdir()):
            if not camera_dir.is_dir():
                continue
            color_folder = camera_dir / "color"
            if color_folder.exists() and list(color_folder.glob("*.png")):
                test_trial = camera_dir
                break
        if test_trial:
            break

    if not test_trial:
        print("❌ No trial with color images found in trial_input/")
        return False

    print(f"\n✅ Found test trial: {test_trial}")

    # Initialize detector
    print("\n🤖 Initializing MediaPipe detector...")
    try:
        detector = MediaPipeHumanDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        print("✅ Detector initialized")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        return False

    # Process first 10 images as test
    color_folder = test_trial / "color"
    image_files = sorted(list(color_folder.glob("*.png")))[:10]

    if not image_files:
        print("❌ No PNG images found")
        return False

    print(f"\n📷 Processing {len(image_files)} test images...")

    import cv2
    results = []
    for i, img_path in enumerate(image_files, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = detector.detect_frame(img_rgb, i)
        if result:
            results.append(result)
            print(f"  Frame {i}: ✅ Detected {len(result.landmarks_2d)} landmarks, pointing_arm={result.metadata['pointing_arm']}")
        else:
            print(f"  Frame {i}: ❌ No pose detected")

    print(f"\n📊 Results:")
    print(f"  Total frames: {len(image_files)}")
    print(f"  Detected: {len(results)}")
    print(f"  Detection rate: {len(results)/len(image_files)*100:.1f}%")

    if results:
        # Save test output
        output_file = "test_skeleton_output.json"
        detector.save_results(results, output_file)
        print(f"\n💾 Saved test output to: {output_file}")
        print(f"\n✅ MediaPipe test PASSED")
        return True
    else:
        print(f"\n❌ MediaPipe test FAILED - no poses detected")
        return False


if __name__ == "__main__":
    success = test_mediapipe()
    sys.exit(0 if success else 1)

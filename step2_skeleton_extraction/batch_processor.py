#!/usr/bin/env python3
"""
Batch processor for skeleton extraction across multiple trials.

Processes trial_input/ folders and saves skeleton data to trial_output/.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
from step0_data_loading.load_trial_data_flexible import load_depth_flexible, detect_depth_shape


def determine_pointing_hand_whole_trial(results: List) -> str:
    """
    Determine which hand is used for pointing across the entire trial.

    Uses a voting mechanism with weighted scores based on:
    1. Per-frame detection confidence (motion, height, extension)
    2. Majority voting across frames
    3. Consistency across the trial

    Args:
        results: List of SkeletonResult objects

    Returns:
        "left", "right", or "unknown"
    """
    if not results:
        return "unknown"

    # Count votes from each frame
    left_votes = 0
    right_votes = 0
    auto_votes = 0

    for result in results:
        arm = result.metadata.get('pointing_arm', 'auto')
        if arm == 'left':
            left_votes += 1
        elif arm == 'right':
            right_votes += 1
        else:
            auto_votes += 1

    total_votes = len(results)
    left_pct = left_votes / total_votes if total_votes > 0 else 0
    right_pct = right_votes / total_votes if total_votes > 0 else 0

    # Require at least 30% confidence in one hand to make a decision
    confidence_threshold = 0.30

    if left_pct >= confidence_threshold and left_pct > right_pct * 1.5:
        return "left"
    elif right_pct >= confidence_threshold and right_pct > left_pct * 1.5:
        return "right"
    else:
        # Not enough confidence, return which one has more votes
        if left_votes > right_votes:
            return "left"
        elif right_votes > left_votes:
            return "right"
        else:
            return "unknown"


def process_trial(trial_path: str,
                  camera_id: Optional[str] = None,
                  output_dir: str = "trial_output",
                  detector_type: str = "mediapipe",
                  crop_upper_half: bool = False) -> None:
    """
    Process a single trial and extract skeleton data.

    Args:
        trial_path: Path to trial folder in trial_input/
        camera_id: Camera ID (e.g., 'cam1')
        output_dir: Base output directory
        detector_type: 'mediapipe' or 'deeplabcut'
        crop_upper_half: If True, crop upper half to focus on upper body only
    """
    trial_path = Path(trial_path)

    # Determine trial name and camera
    if camera_id:
        # Multi-camera: trial_input/trial_1/cam1/
        trial_name = trial_path.parent.name
        camera = camera_id
    else:
        # Single-camera: trial_input/1/single_camera/
        trial_name = trial_path.parent.name
        camera = "single_camera"

    print(f"\n{'='*60}")
    print(f"Processing Trial: {trial_name} / {camera}")
    print(f"{'='*60}")

    # Find color and depth folders
    color_folder = trial_path / "color"
    depth_folder = trial_path / "depth"

    if not color_folder.exists():
        print(f"❌ Color folder not found: {color_folder}")
        return

    has_depth = depth_folder.exists()
    if has_depth:
        print(f"✅ Found depth folder: {depth_folder}")
    else:
        print(f"⚠️ No depth folder found, processing 2D only")

    # Initialize detector
    if detector_type == "mediapipe":
        if crop_upper_half:
            print("🤖 Using MediaPipe human pose detector (UPPER HALF ONLY)")
        else:
            print("🤖 Using MediaPipe human pose detector")
        detector = MediaPipeHumanDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            lower_half_only=False,  # For baby detection (not used here)
            upper_half_only=crop_upper_half  # Focus on upper body for pointing gestures
        )
    else:
        print(f"❌ Detector type '{detector_type}' not implemented yet")
        return

    # Auto-detect camera intrinsics from image size
    sample_image = sorted(color_folder.glob("*.png"))[0]
    sample = cv2.imread(str(sample_image))
    h, w = sample.shape[:2]

    if w == 640 and h == 480:
        fx = fy = 615.0
        cx = 320.0
        cy = 240.0
    elif w == 1280 and h == 720:
        fx = fy = 922.5
        cx = 640.0
        cy = 360.0
    elif w == 1920 and h == 1080:
        fx = fy = 1383.75
        cx = 960.0
        cy = 540.0
    else:
        # Default fallback
        fx = fy = w * 0.9
        cx = w / 2.0
        cy = h / 2.0

    print(f"📷 Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")

    # Process all color images with depth if available
    print(f"📁 Processing images from: {color_folder}")
    results = []
    color_images = sorted(color_folder.glob("frame_*.png"))

    for i, color_path in enumerate(color_images, 1):
        # Extract frame number from filename
        frame_num = int(color_path.stem.split('_')[-1])

        # Load color image
        color_img = cv2.imread(str(color_path))
        color_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        # Load depth if available
        depth_img = None
        if has_depth:
            try:
                # Try .npy first, then .raw
                depth_npy = depth_folder / f"frame_{frame_num:06d}.npy"
                depth_raw = depth_folder / f"frame_{frame_num:06d}.raw"

                if depth_npy.exists():
                    depth_img = np.load(str(depth_npy))
                elif depth_raw.exists():
                    depth_img = load_depth_flexible(
                        str(trial_path.parent),
                        trial_path.name if camera_id else None,
                        frame_num
                    )
                    # Convert uint16 to meters if needed
                    if depth_img.dtype == np.uint16:
                        depth_img = depth_img.astype(np.float32) / 1000.0
            except Exception as e:
                print(f"⚠️ Could not load depth for frame {frame_num}: {e}")

        # Detect skeleton
        result = detector.detect_frame(
            color_rgb, frame_num,
            depth_image=depth_img,
            fx=fx, fy=fy, cx=cx, cy=cy
        )

        if result:
            results.append(result)

        # Progress indicator
        if i % 30 == 0:
            print(f"  Processed {i}/{len(color_images)} frames...", end='\r')

    print(f"\n✅ Processed {len(color_images)} frames")

    if not results:
        print("⚠️ No skeletons detected in any frames")
        return

    print(f"✅ Detected skeletons in {len(results)} frames")

    # Create output directory
    output_path = Path(output_dir) / trial_name / camera
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results
    output_file = output_path / "skeleton_2d.json"
    detector.save_results(results, str(output_file))

    # Determine whole-trial pointing hand
    pointing_hand = determine_pointing_hand_whole_trial(results)

    # Also save summary statistics
    summary_file = output_path / "skeleton_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Skeleton Extraction Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Trial: {trial_name}\n")
        f.write(f"Camera: {camera}\n")
        f.write(f"Total frames: {len(results)}\n")
        f.write(f"Detector: {detector_type}\n")
        f.write(f"\n{'='*40}\n")
        f.write(f"POINTING HAND (whole trial): {pointing_hand.upper()}\n")
        f.write(f"{'='*40}\n")
        f.write(f"\nPer-frame pointing arm distribution:\n")

        # Count pointing arms
        pointing_arms = {}
        for result in results:
            arm = result.metadata.get('pointing_arm', 'unknown')
            pointing_arms[arm] = pointing_arms.get(arm, 0) + 1

        for arm, count in sorted(pointing_arms.items()):
            f.write(f"  {arm}: {count} frames ({count/len(results)*100:.1f}%)\n")

    # Save pointing hand to separate JSON file for easy programmatic access
    import json
    pointing_hand_file = output_path / "pointing_hand.json"
    with open(pointing_hand_file, 'w') as f:
        json.dump({
            "trial": trial_name,
            "camera": camera,
            "pointing_hand": pointing_hand,
            "total_frames": len(results),
            "frame_distribution": pointing_arms
        }, f, indent=2)

    print(f"📊 Saved summary to: {summary_file}")
    print(f"👆 Pointing hand: {pointing_hand.upper()}")
    print(f"💾 Saved pointing hand to: {pointing_hand_file}")
    print(f"✅ Trial processing complete!")


def process_all_trials(trial_input_dir: str = "trial_input",
                       output_dir: str = "trial_output",
                       detector_type: str = "mediapipe",
                       crop_upper_half: bool = False) -> None:
    """
    Process all trials in trial_input/ directory.

    Args:
        trial_input_dir: Base trial input directory
        output_dir: Base output directory
        detector_type: Detector type to use
        crop_upper_half: If True, crop to upper half to focus on upper body
    """
    trial_input_path = Path(trial_input_dir)

    if not trial_input_path.exists():
        print(f"❌ Trial input directory not found: {trial_input_path}")
        return

    print(f"\n🔍 Scanning for trials in: {trial_input_path}")

    # Find all trial folders
    trials_found = []

    for trial_dir in sorted(trial_input_path.iterdir()):
        if not trial_dir.is_dir():
            continue

        # Check for camera subdirectories
        for camera_dir in sorted(trial_dir.iterdir()):
            if not camera_dir.is_dir():
                continue

            color_folder = camera_dir / "color"
            if color_folder.exists():
                trials_found.append((str(camera_dir), camera_dir.name))

    if not trials_found:
        print("❌ No trials found")
        return

    print(f"✅ Found {len(trials_found)} trials to process\n")

    # Process each trial
    for trial_path, camera_id in trials_found:
        try:
            process_trial(
                trial_path=trial_path,
                camera_id=camera_id if camera_id != "single_camera" else None,
                output_dir=output_dir,
                detector_type=detector_type,
                crop_upper_half=crop_upper_half
            )
        except Exception as e:
            print(f"❌ Error processing {trial_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"✅ Batch processing complete!")
    print(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch skeleton extraction")
    parser.add_argument(
        "--trial_input",
        type=str,
        default="trial_input",
        help="Trial input directory (default: trial_input)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trial_output",
        help="Output directory (default: trial_output)"
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["mediapipe", "deeplabcut"],
        default="mediapipe",
        help="Detector type (default: mediapipe)"
    )
    parser.add_argument(
        "--trial",
        type=str,
        help="Process specific trial only (e.g., trial_input/trial_1/cam1)"
    )
    parser.add_argument(
        "--crop-upper-half",
        action="store_true",
        help="Crop to upper 60%% of image to focus on upper body (helps when multiple people in frame)"
    )

    args = parser.parse_args()

    if args.trial:
        # Process single trial
        process_trial(
            trial_path=args.trial,
            output_dir=args.output,
            detector_type=args.detector,
            crop_upper_half=args.crop_upper_half
        )
    else:
        # Process all trials
        process_all_trials(
            trial_input_dir=args.trial_input,
            output_dir=args.output,
            detector_type=args.detector,
            crop_upper_half=args.crop_upper_half
        )


if __name__ == "__main__":
    main()

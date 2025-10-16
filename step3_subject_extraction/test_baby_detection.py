"""
Test script for baby detection with lower-half processing.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from step2_skeleton_extraction.mediapipe_human import MediaPipeHumanDetector
from step2_skeleton_extraction.image_utils import crop_to_lower_half


def visualize_crop(image, crop_ratio=0.5):
    """Visualize the crop line on the image."""
    vis = image.copy()
    h = vis.shape[0]
    y_offset = int(h * (1 - crop_ratio))

    # Draw red line showing crop boundary
    cv2.line(vis, (0, y_offset), (vis.shape[1], y_offset), (0, 0, 255), 3)
    cv2.putText(vis, f"Lower {int(crop_ratio*100)}% processed",
                (10, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    return vis


def test_baby_detection(image_path, output_dir=None):
    """
    Test baby detection on a single image.

    Args:
        image_path: Path to test image
        output_dir: Optional output directory for visualization
    """
    print(f"\n{'='*60}")
    print(f"TESTING BABY DETECTION (Lower-Half Only)")
    print(f"{'='*60}\n")

    # Load image
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        return

    print(f"📸 Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    print(f"   Image size: {w}x{h}")

    # Test 1: Standard human detection (full image)
    print(f"\n{'='*60}")
    print("TEST 1: Standard Human Detection (Full Image)")
    print(f"{'='*60}")

    detector_full = MediaPipeHumanDetector(
        min_detection_confidence=0.5,
        lower_half_only=False
    )

    result_full = detector_full.detect_frame(image_rgb, frame_number=1)

    if result_full:
        print(f"✅ Full image detection successful")
        print(f"   Landmarks detected: {len(result_full.landmarks_2d)}")

        # Check landmark distribution
        y_coords = [lm[1] for lm in result_full.landmarks_2d]
        upper_half = sum(1 for y in y_coords if y < h/2)
        lower_half = sum(1 for y in y_coords if y >= h/2)
        print(f"   Landmarks in upper half: {upper_half}")
        print(f"   Landmarks in lower half: {lower_half}")
    else:
        print(f"❌ No pose detected in full image")

    # Test 2: Baby detection (lower-half only)
    print(f"\n{'='*60}")
    print("TEST 2: Baby Detection (Lower-Half Only)")
    print(f"{'='*60}")

    detector_baby = MediaPipeHumanDetector(
        min_detection_confidence=0.5,
        lower_half_only=True
    )

    result_baby = detector_baby.detect_frame(image_rgb, frame_number=1)

    if result_baby:
        print(f"✅ Baby detection successful")
        print(f"   Landmarks detected: {len(result_baby.landmarks_2d)}")

        # Check if coordinates are in original image space
        y_coords = [lm[1] for lm in result_baby.landmarks_2d]
        min_y, max_y = min(y_coords), max(y_coords)
        print(f"   Y coordinate range: {min_y:.1f} - {max_y:.1f} (image height: {h})")

        # Check if landmarks are in lower half
        lower_half_only = all(y >= h/2 for y in y_coords if y > 0)
        if lower_half_only:
            print(f"   ✅ All landmarks in lower half (as expected)")
        else:
            upper_count = sum(1 for y in y_coords if y < h/2)
            print(f"   ⚠️  Some landmarks in upper half: {upper_count}/{len(y_coords)}")
            print(f"      (This is OK if person's head is in upper half but detected from lower-half crop)")
    else:
        print(f"❌ No pose detected with lower-half processing")

    # Test 3: Visualize crop region
    print(f"\n{'='*60}")
    print("TEST 3: Visualization")
    print(f"{'='*60}")

    # Create visualizations
    vis_crop = visualize_crop(image, crop_ratio=0.5)

    # Draw landmarks on both
    if result_full:
        vis_full = image.copy()
        for i, (x, y, conf) in enumerate(result_full.landmarks_2d):
            if conf > 0.5:
                cv2.circle(vis_full, (int(x), int(y)), 4, (0, 255, 0), -1)
        print(f"   ✓ Full image visualization created")
    else:
        vis_full = image.copy()

    if result_baby:
        vis_baby = vis_crop.copy()
        for i, (x, y, conf) in enumerate(result_baby.landmarks_2d):
            if conf > 0.5:
                cv2.circle(vis_baby, (int(x), int(y)), 4, (0, 255, 0), -1)
        print(f"   ✓ Baby detection visualization created")
    else:
        vis_baby = vis_crop

    # Save visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_dir / "test_full_detection.jpg"), vis_full)
        cv2.imwrite(str(output_dir / "test_baby_detection.jpg"), vis_baby)

        print(f"\n📁 Visualizations saved to: {output_dir}")
        print(f"   - test_full_detection.jpg")
        print(f"   - test_baby_detection.jpg")

    # Test 4: Verify coordinate mapping
    print(f"\n{'='*60}")
    print("TEST 4: Coordinate Mapping Verification")
    print(f"{'='*60}")

    if result_baby:
        # Manually test coordinate mapping
        cropped_image, y_offset = crop_to_lower_half(image_rgb, crop_ratio=0.5)
        print(f"   Original image size: {w}x{h}")
        print(f"   Cropped image size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
        print(f"   Y offset: {y_offset}")

        # Sample landmark should be >= y_offset
        sample_y = result_baby.landmarks_2d[0][1]
        if sample_y >= y_offset:
            print(f"   ✅ Coordinates correctly mapped (nose Y={sample_y:.1f} >= offset={y_offset})")
        else:
            print(f"   ❌ Coordinate mapping error (nose Y={sample_y:.1f} < offset={y_offset})")

    print(f"\n{'='*60}")
    print("TEST COMPLETE")
    print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test baby detection with lower-half processing")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--output", default="test_output", help="Output directory for visualizations")

    args = parser.parse_args()

    test_baby_detection(args.image, args.output)


if __name__ == "__main__":
    main()

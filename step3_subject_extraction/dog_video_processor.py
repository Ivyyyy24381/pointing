"""
Process dog detection by creating temporary video from frame sequence.
"""
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Optional
from tqdm import tqdm


def create_video_from_frames(color_dir: Path, output_video: Path, fps: float = 30.0) -> bool:
    """
    Create video from frame sequence.

    Args:
        color_dir: Directory containing frame_*.png files
        output_video: Path to output video file
        fps: Frames per second

    Returns:
        True if successful, False otherwise
    """
    # Get all PNG files
    frame_files = sorted(color_dir.glob("frame_*.png"))

    if not frame_files:
        print(f"❌ No frames found in {color_dir}")
        return False

    print(f"📹 Creating video from {len(frame_files)} frames...")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        print(f"❌ Failed to read first frame")
        return False

    h, w = first_frame.shape[:2]
    print(f"   Video size: {w}x{h} @ {fps} fps")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"❌ Failed to create video writer")
        return False

    # Write all frames
    for frame_path in tqdm(frame_files, desc="Writing frames"):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            writer.write(frame)

    writer.release()
    print(f"✅ Video created: {output_video}")
    return True


def process_dog_trial(trial_path: Path, output_dir: Path) -> Optional[Dict]:
    """
    Process dog detection for a trial.

    Args:
        trial_path: Path to trial_input/X/single_camera/
        output_dir: Path to trial_output/X/single_camera/

    Returns:
        Dictionary of results or None if failed
    """
    print(f"\n{'='*60}")
    print(f"🐶 DOG DETECTION - DEEPLABCUT")
    print(f"{'='*60}")
    print(f"Input: {trial_path}")
    print(f"Output: {output_dir}")

    color_dir = trial_path / "color"
    if not color_dir.exists():
        print(f"❌ Color directory not found: {color_dir}")
        return None

    # Create temporary video
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_video = Path(tmpdir) / "temp_video.mp4"

        print(f"\n📹 Step 1: Creating temporary video...")
        if not create_video_from_frames(color_dir, tmp_video):
            return None

        # Run DeepLabCut detection
        print(f"\n🔬 Step 2: Running DeepLabCut detection...")
        try:
            from step2_skeleton_extraction.deeplabcut_dog import DeepLabCutDogDetector

            detector = DeepLabCutDogDetector()

            # Set output directory for DLC results
            output_dir.mkdir(parents=True, exist_ok=True)

            # Run detection
            results = detector.detect_video(str(tmp_video), output_dir=str(output_dir))

            if not results:
                print(f"❌ DeepLabCut detection failed or returned no results")
                return None

            print(f"✅ Detection complete: {len(results)} frames processed")

            # Move DLC output files to output directory
            print(f"\n📦 Step 3: Organizing output files...")

            # Look for DLC output files in tmpdir
            for file_pattern in ["*.json", "*.mp4", "*.h5"]:
                for dlc_file in Path(tmpdir).glob(file_pattern):
                    if "DLC" in dlc_file.name or "skeleton" in dlc_file.name:
                        dest = output_dir / dlc_file.name
                        shutil.copy2(dlc_file, dest)
                        print(f"   Copied: {dlc_file.name}")

            return results

        except ImportError as e:
            print(f"❌ DeepLabCut not installed: {e}")
            print(f"   Install with: pip install deeplabcut")
            return None
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def dog_results_to_skeleton_format(dog_results: Dict, trial_path: Path) -> Dict:
    """
    Convert DeepLabCut dog results to standard skeleton format.

    Args:
        dog_results: Dictionary from DeepLabCutDogDetector
        trial_path: Path to trial for loading depth/metadata

    Returns:
        Dictionary in standard skeleton format
    """
    from step2_skeleton_extraction.skeleton_base import SkeletonResult

    skeleton_results = {}

    print(f"\n🔄 Converting {len(dog_results)} frames to skeleton format...")

    for frame_idx, frame_data in dog_results.items():
        # Get frame number (0-indexed from DLC)
        frame_num = frame_idx + 1  # Convert to 1-indexed
        frame_key = f"frame_{frame_num:06d}"

        # Extract keypoints from DLC format
        bodyparts = frame_data.get('bodyparts', [])
        bboxes = frame_data.get('bboxes', [])

        if not bodyparts:
            continue

        # DLC format: list of individuals, each with keypoints
        # We use max_individuals=1, so take first individual
        if len(bodyparts) > 0:
            keypoints = bodyparts[0]  # [[x, y, conf], ...]

            # Create skeleton result
            landmarks_2d = [(kp[0], kp[1], kp[2]) for kp in keypoints]

            # For dogs, we don't have hip center concept
            # Use midpoint between left and right hip if available
            hip_center_2d = None
            if len(landmarks_2d) >= 35:  # Check if hip keypoints exist
                left_hip_idx = 33  # Approximate indices
                right_hip_idx = 34
                left_hip = landmarks_2d[left_hip_idx]
                right_hip = landmarks_2d[right_hip_idx]
                hip_center_2d = (
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                )

            # Create result
            result = SkeletonResult(
                frame_number=frame_num,
                landmarks_2d=landmarks_2d,
                landmarks_3d=None,  # 3D not computed for dogs yet
                hip_center_2d=hip_center_2d,
                hip_center_3d=None,
                arm_vectors=None,  # Not applicable for dogs
                metadata={
                    'subject_type': 'dog',
                    'num_keypoints': len(landmarks_2d),
                    'bbox': bboxes[0] if bboxes else None
                }
            )

            skeleton_results[frame_key] = result

    print(f"✅ Converted {len(skeleton_results)} frames")
    return skeleton_results


if __name__ == "__main__":
    # Test video creation
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, help="Path to trial_input/X/single_camera/")
    parser.add_argument("--output", required=True, help="Path to trial_output/X/single_camera/")

    args = parser.parse_args()

    results = process_dog_trial(Path(args.trial), Path(args.output))

    if results:
        print(f"\n✅ Success! Processed {len(results)} frames")
    else:
        print(f"\n❌ Failed")

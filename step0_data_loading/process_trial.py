"""
Process Trial - Load all frames and save to trial_input folder

This script:
1. Detects trial folder structure (multi-camera or single-camera)
2. Finds all available frames
3. Loads color and depth for each frame in parallel
4. Saves organized output to trial_input/<trial_name>/

Usage:
    python process_trial.py <trial_path> [camera_id]

    Examples:
    python process_trial.py sample_raw_data/trial_1 cam1
    python process_trial.py sample_raw_data/1
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any
import glob
import shutil
from multiprocessing import Pool, cpu_count
from functools import partial

# Try to import tqdm, fallback to simple progress
try:
    from tqdm import tqdm as _tqdm
    tqdm = _tqdm
    tqdm_write = _tqdm.write
except ImportError:
    def tqdm(iterable, desc="Processing"):
        iterable_list = list(iterable)
        total = len(iterable_list)
        for i, item in enumerate(iterable_list):
            print(f"\r{desc}: {i+1}/{total}", end='', flush=True)
            yield item
        print()  # New line after completion

    def tqdm_write(msg):
        print(msg)

# Import our flexible loader
sys.path.insert(0, os.path.dirname(__file__))
from load_trial_data_flexible import (
    detect_folder_structure,
    load_color_flexible,
    load_depth_flexible,
    list_available_cameras
)


def precompute_metadata(trial_path: str, camera_id: Optional[str]) -> Dict[str, Any]:
    """
    Precompute expensive operations that would otherwise be repeated per frame.

    This function runs ONCE before parallel processing to detect:
    - Folder structure
    - True depth folder location
    - Depth shape for .raw files
    - File naming patterns

    Args:
        trial_path: Path to trial folder
        camera_id: Camera ID (None for single-camera structure)

    Returns:
        Dictionary containing precomputed metadata
    """
    from load_trial_data_flexible import (
        detect_folder_structure,
        detect_true_depth_folder,
        detect_depth_shape,
        find_depth_file
    )

    structure = detect_folder_structure(trial_path)
    depth_folder_name = detect_true_depth_folder(trial_path, camera_id)

    # Auto-detect depth shape from first available .raw file
    depth_shape = None
    if structure == 'multi_camera' and camera_id:
        for folder_name in ['depth', 'Depth']:
            depth_folder = os.path.join(trial_path, camera_id, folder_name)
            if os.path.isdir(depth_folder):
                raw_files = glob.glob(os.path.join(depth_folder, '*.raw'))
                if raw_files:
                    depth_shape = detect_depth_shape(raw_files[0])
                    break
    elif structure == 'single_camera':
        if depth_folder_name:
            depth_folder = os.path.join(trial_path, depth_folder_name)
            if os.path.isdir(depth_folder):
                raw_files = glob.glob(os.path.join(depth_folder, '*.raw'))
                if raw_files:
                    depth_shape = detect_depth_shape(raw_files[0])

    metadata = {
        'structure': structure,
        'depth_folder_name': depth_folder_name,
        'depth_shape': depth_shape
    }

    return metadata


def process_single_frame(
    frame_num: int,
    trial_path: str,
    camera_id: Optional[str],
    output_path: str,
    metadata: Dict[str, Any]
) -> Tuple[int, bool, Optional[str]]:
    """
    Process a single frame: load color and depth, save to output.

    This function is designed to run in parallel via multiprocessing.Pool.

    Args:
        frame_num: Frame number to process
        trial_path: Path to trial folder
        camera_id: Camera ID (None for single-camera)
        output_path: Output directory path
        metadata: Precomputed metadata (structure, depth_folder_name, depth_shape)

    Returns:
        Tuple of (frame_num, success, error_message)
    """
    try:
        from load_trial_data_flexible import (
            load_color_flexible,
            load_depth_flexible
        )

        # Load color
        color_success = False
        try:
            color_img = load_color_flexible(trial_path, camera_id, frame_num)
            color_out = os.path.join(output_path, "color", f"frame_{frame_num:06d}.png")
            cv2.imwrite(color_out, color_img)
            color_success = True
        except Exception as e:
            return (frame_num, False, f"Color failed: {e}")

        # Load depth (with precomputed metadata)
        depth_success = False
        try:
            depth_img = load_depth_flexible(
                trial_path,
                camera_id,
                frame_num,
                depth_shape=metadata.get('depth_shape')
            )
            depth_out = os.path.join(output_path, "depth", f"frame_{frame_num:06d}.npy")
            np.save(depth_out, depth_img)
            depth_success = True
        except Exception as e:
            return (frame_num, False, f"Depth failed: {e}")

        return (frame_num, True, None)

    except Exception as e:
        return (frame_num, False, str(e))


def find_all_frames(trial_path: str, camera_id: Optional[str] = None) -> List[int]:
    """
    Find all available frame numbers in a trial

    Args:
        trial_path: Path to trial folder
        camera_id: Camera ID (None for single-camera structure)

    Returns:
        List of frame numbers (sorted)
    """
    structure = detect_folder_structure(trial_path)
    frame_numbers = set()

    if structure == 'multi_camera':
        if camera_id is None:
            cameras = list_available_cameras(trial_path)
            if cameras:
                camera_id = cameras[0]
            else:
                return []

        # Check color folder
        for folder_name in ['color', 'Color']:
            color_folder = os.path.join(trial_path, camera_id, folder_name)
            if os.path.isdir(color_folder):
                # Try different patterns
                patterns = [
                    os.path.join(color_folder, 'frame_*.png'),
                    os.path.join(color_folder, 'frame_*.jpg'),
                    os.path.join(color_folder, 'Color_*.png'),
                ]

                for pattern in patterns:
                    files = glob.glob(pattern)
                    for f in files:
                        basename = os.path.basename(f)
                        # Extract number from filename
                        import re
                        match = re.search(r'_(\d+)\.', basename)
                        if match:
                            frame_numbers.add(int(match.group(1)))

    elif structure == 'single_camera':
        # Check Color/color folder
        for folder_name in ['Color', 'color']:
            color_folder = os.path.join(trial_path, folder_name)
            if os.path.isdir(color_folder):
                # Try different patterns
                patterns = [
                    os.path.join(color_folder, '_Color_*.png'),
                    os.path.join(color_folder, 'frame_*.png'),
                    os.path.join(color_folder, 'Color_*.png'),
                ]

                for pattern in patterns:
                    files = glob.glob(pattern)
                    for f in files:
                        basename = os.path.basename(f)
                        # Extract number from filename
                        import re
                        match = re.search(r'_(\d+)\.', basename)
                        if match:
                            frame_numbers.add(int(match.group(1)))

    return sorted(list(frame_numbers))


def process_trial(trial_path: str,
                  camera_id: Optional[str] = None,
                  output_base: str = "trial_input",
                  frame_range: Optional[Tuple[int, int]] = None,
                  num_workers: int = 8) -> str:
    """
    Process an entire trial and save to trial_input folder using parallel processing.

    Args:
        trial_path: Path to trial folder (e.g., 'sample_raw_data/trial_1')
        camera_id: Camera ID (e.g., 'cam1', or None for single-camera)
        output_base: Base output directory
        frame_range: Optional (start, end) frame range, None for all frames
        num_workers: Number of parallel workers (default: 8)

    Returns:
        output_path: Path to created output folder
    """
    import time
    import json

    # Get trial name
    trial_name = os.path.basename(os.path.normpath(trial_path))

    # Detect structure
    structure = detect_folder_structure(trial_path)
    print(f"\n📁 Detected structure: {structure}")

    # Auto-detect camera if needed
    if structure == 'multi_camera' and camera_id is None:
        cameras = list_available_cameras(trial_path)
        if cameras:
            camera_id = cameras[0]
            print(f"🎥 Auto-selected camera: {camera_id}")
        else:
            raise ValueError("No cameras found in multi-camera structure")

    # Find all frames
    print(f"\n🔍 Finding frames...")
    frame_numbers = find_all_frames(trial_path, camera_id)

    if not frame_numbers:
        raise ValueError(f"No frames found in {trial_path}")

    # Apply frame range filter
    if frame_range:
        frame_numbers = [f for f in frame_numbers if frame_range[0] <= f <= frame_range[1]]

    print(f"✅ Found {len(frame_numbers)} frames (range: {min(frame_numbers)}-{max(frame_numbers)})")

    # Create output directory (nested structure to match trial_output)
    if camera_id:
        output_path = os.path.join(output_base, trial_name, camera_id)
    else:
        # For single camera, use 'single_camera' as camera name for consistency
        output_path = os.path.join(output_base, trial_name, "single_camera")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "color"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "depth"), exist_ok=True)

    print(f"\n💾 Output directory: {output_path}")

    # Precompute expensive metadata ONCE (instead of per-frame)
    print(f"\n🔧 Precomputing metadata (folder structure, depth shape, etc.)...")
    precomputed_metadata = precompute_metadata(trial_path, camera_id)
    print(f"✅ Metadata precomputed:")
    print(f"   Structure: {precomputed_metadata['structure']}")
    print(f"   Depth folder: {precomputed_metadata['depth_folder_name']}")
    if precomputed_metadata['depth_shape']:
        print(f"   Depth shape: {precomputed_metadata['depth_shape']}")

    # Prepare worker function with fixed parameters
    worker_func = partial(
        process_single_frame,
        trial_path=trial_path,
        camera_id=camera_id,
        output_path=output_path,
        metadata=precomputed_metadata
    )

    # Process frames in parallel
    print(f"\n⚙️ Processing {len(frame_numbers)} frames with {num_workers} workers...")
    successful = 0
    failed = 0
    errors = []

    start_time = time.time()

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=num_workers) as pool:
        # Process frames with progress bar
        results = []
        for result in tqdm(
            pool.imap(worker_func, frame_numbers),
            total=len(frame_numbers),
            desc="Processing frames",
            unit="frame"
        ):
            results.append(result)
            frame_num, success, error_msg = result

            if success:
                successful += 1
            else:
                failed += 1
                errors.append((frame_num, error_msg))

    elapsed_time = time.time() - start_time
    fps = len(frame_numbers) / elapsed_time if elapsed_time > 0 else 0

    # Print errors if any
    if errors:
        print(f"\n⚠️ Errors encountered ({len(errors)} frames):")
        for frame_num, error_msg in errors[:10]:  # Show first 10 errors
            print(f"   Frame {frame_num}: {error_msg}")
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")

    # Save metadata
    metadata = {
        'trial_path': trial_path,
        'trial_name': trial_name,
        'camera_id': camera_id,
        'structure': structure,
        'total_frames': len(frame_numbers),
        'successful': successful,
        'failed': failed,
        'frame_range': [min(frame_numbers), max(frame_numbers)],
        'processing_time_seconds': elapsed_time,
        'processing_fps': fps,
        'num_workers': num_workers
    }

    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Processing complete!")
    print(f"   Successful: {successful}/{len(frame_numbers)}")
    print(f"   Failed: {failed}/{len(frame_numbers)}")
    print(f"   Time: {elapsed_time:.2f} seconds")
    print(f"   Speed: {fps:.1f} fps")
    print(f"   Workers: {num_workers}")
    print(f"   Output: {output_path}")

    return output_path


def process_all_cameras(trial_path: str, output_base: str = "trial_input") -> List[str]:
    """
    Process all cameras in a multi-camera trial

    Args:
        trial_path: Path to trial folder
        output_base: Base output directory

    Returns:
        List of output paths
    """
    structure = detect_folder_structure(trial_path)

    if structure != 'multi_camera':
        print(f"Trial is not multi-camera structure, processing as single camera")
        return [process_trial(trial_path, None, output_base)]

    cameras = list_available_cameras(trial_path)
    print(f"🎥 Found {len(cameras)} cameras: {', '.join(cameras)}")

    output_paths = []
    for camera in cameras:
        print(f"\n{'='*60}")
        print(f"Processing {camera}")
        print(f"{'='*60}")

        try:
            output_path = process_trial(trial_path, camera, output_base)
            output_paths.append(output_path)
        except Exception as e:
            print(f"❌ Failed to process {camera}: {e}")

    return output_paths


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Insufficient arguments")
        print("\nUsage:")
        print("  python process_trial.py <trial_path> [camera_id]")
        print("\nExamples:")
        print("  python process_trial.py sample_raw_data/trial_1 cam1")
        print("  python process_trial.py sample_raw_data/1")
        print("\nTo process all cameras:")
        print("  python process_trial.py sample_raw_data/trial_1 --all")
        sys.exit(1)

    trial_path = sys.argv[1]

    if not os.path.isdir(trial_path):
        print(f"❌ Error: Trial path does not exist: {trial_path}")
        sys.exit(1)

    # Check for --all flag
    if len(sys.argv) > 2 and sys.argv[2] == '--all':
        print("🎬 Processing all cameras...")
        output_paths = process_all_cameras(trial_path)
        print(f"\n🎉 Complete! Processed {len(output_paths)} cameras")
        sys.exit(0)

    # Single camera processing
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if len(sys.argv) < 3:
            cameras = list_available_cameras(trial_path)
            print(f"\n📋 Available cameras: {', '.join(cameras)}")
            print(f"\nUsage: python process_trial.py {trial_path} <camera_id>")
            print(f"   Or: python process_trial.py {trial_path} --all")
            sys.exit(1)

        camera_id = sys.argv[2]
    else:
        camera_id = None

    try:
        output_path = process_trial(trial_path, camera_id)
        print(f"\n✅ Success! Output saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

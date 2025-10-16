"""
Flexible Trial Data Loader

Automatically detects and loads data from different folder structures:
- Structure A: trial_1/cam1/color/frame_000001.png
- Structure B: 1/Color/_Color_0001.png (no camera subfolders)

Usage:
    python load_trial_data_flexible.py <trial_path> [camera_id] <frame_number>

    Examples:
    python load_trial_data_flexible.py sample_raw_data/trial_1 cam1 31
    python load_trial_data_flexible.py sample_raw_data/1 230
"""

import os
import sys
import numpy as np
import cv2
from typing import Tuple, Optional, List
import glob


def detect_folder_structure(trial_path: str) -> str:
    """
    Detect which folder structure is used

    Returns:
        'multi_camera': trial_path/cam1/color/frame_XXXXXX.png
        'single_camera': trial_path/Color/_Color_XXXX.png
        'unknown': Could not detect
    """
    # Check for multi-camera structure (cam1, cam2, cam3)
    cam_folders = [d for d in os.listdir(trial_path)
                   if os.path.isdir(os.path.join(trial_path, d)) and d.startswith('cam')]

    if cam_folders:
        # Check if cam1/color or cam1/Color exists
        for cam in cam_folders:
            color_path_lower = os.path.join(trial_path, cam, 'color')
            color_path_upper = os.path.join(trial_path, cam, 'Color')
            if os.path.isdir(color_path_lower) or os.path.isdir(color_path_upper):
                return 'multi_camera'

    # Check for single camera structure (Color, Depth folders directly)
    color_folder = os.path.join(trial_path, 'Color')
    color_folder_lower = os.path.join(trial_path, 'color')

    if os.path.isdir(color_folder) or os.path.isdir(color_folder_lower):
        return 'single_camera'

    return 'unknown'


def find_color_file(trial_path: str, camera_id: Optional[str], frame_number: int) -> Optional[str]:
    """
    Find color file with flexible naming

    Tries multiple naming patterns:
    - frame_XXXXXX.png (6 digits)
    - _Color_XXXX.png (4 digits)
    - Color_XXXXXX.png (6 digits)
    """
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if camera_id is None:
            raise ValueError("camera_id required for multi_camera structure")

        # Try both color and Color
        for folder_name in ['color', 'Color']:
            color_folder = os.path.join(trial_path, camera_id, folder_name)
            if not os.path.isdir(color_folder):
                continue

            # Try different naming patterns
            patterns = [
                f'frame_{frame_number:06d}',  # frame_000031
                f'Color_{frame_number:06d}',   # Color_000031
                f'{camera_id}_{frame_number:06d}',  # cam1_000031
            ]

            for pattern in patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    path = os.path.join(color_folder, f'{pattern}{ext}')
                    if os.path.exists(path):
                        return path

    elif structure == 'single_camera':
        # Try both Color and color
        for folder_name in ['Color', 'color']:
            color_folder = os.path.join(trial_path, folder_name)
            if not os.path.isdir(color_folder):
                continue

            # Try different naming patterns
            patterns = [
                f'_Color_{frame_number:04d}',  # _Color_0031
                f'_Color_{frame_number:06d}',  # _Color_000031
                f'frame_{frame_number:06d}',   # frame_000031
                f'frame_{frame_number:04d}',   # frame_0031
                f'Color_{frame_number:06d}',   # Color_000031
                f'Color_{frame_number:04d}',   # Color_0031
            ]

            for pattern in patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    path = os.path.join(color_folder, f'{pattern}{ext}')
                    if os.path.exists(path):
                        return path

    return None


def detect_true_depth_folder(trial_path: str, camera_id: Optional[str] = None) -> Optional[str]:
    """
    Detect which folder contains the true depth data.

    For single-camera structures, sometimes 'Depth' and 'Depth_Color' folders are swapped.
    The folder with .raw files contains the most accurate depth data.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera ID (for multi-camera structure)

    Returns:
        Name of the true depth folder ('Depth', 'depth', 'Depth_Color', etc.)
    """
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if camera_id:
            # Check standard depth folder
            for folder_name in ['depth', 'Depth']:
                depth_folder = os.path.join(trial_path, camera_id, folder_name)
                if os.path.isdir(depth_folder):
                    return folder_name
        return 'depth'  # default

    elif structure == 'single_camera':
        # Check for .raw files in different depth folders
        candidate_folders = ['Depth', 'depth', 'Depth_Color', 'depth_color']

        # Priority 1: Folder with .raw files (most accurate depth)
        for folder_name in candidate_folders:
            folder_path = os.path.join(trial_path, folder_name)
            if os.path.isdir(folder_path):
                # Check if this folder contains .raw files
                files = os.listdir(folder_path)
                raw_files = [f for f in files if f.endswith('.raw')]
                if raw_files:
                    print(f"🔍 Detected true depth folder: {folder_name} (contains {len(raw_files)} .raw files)")
                    return folder_name

        # Priority 2: Folder with .npy files
        for folder_name in candidate_folders:
            folder_path = os.path.join(trial_path, folder_name)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                npy_files = [f for f in files if f.endswith('.npy')]
                if npy_files:
                    print(f"🔍 Detected depth folder: {folder_name} (contains {len(npy_files)} .npy files)")
                    return folder_name

        # Priority 3: Any depth folder that exists
        for folder_name in candidate_folders:
            folder_path = os.path.join(trial_path, folder_name)
            if os.path.isdir(folder_path):
                print(f"🔍 Using depth folder: {folder_name}")
                return folder_name

    return None


def find_depth_file(trial_path: str, camera_id: Optional[str], frame_number: int) -> Optional[str]:
    """
    Find depth file with flexible naming

    Tries multiple naming patterns:
    - frame_XXXXXX.npy / frame_XXXXXX.raw
    - _Depth_XXXX.png (for single camera structure)

    Prioritizes .raw files for most accurate depth data.
    """
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if camera_id is None:
            raise ValueError("camera_id required for multi_camera structure")

        # Detect true depth folder
        depth_folder_name = detect_true_depth_folder(trial_path, camera_id)
        if depth_folder_name:
            depth_folder = os.path.join(trial_path, camera_id, depth_folder_name)

            # Try different naming patterns and formats (prioritize .raw)
            patterns = [
                f'frame_{frame_number:06d}',  # frame_000031
                f'Depth_{frame_number:06d}',   # Depth_000031
                f'{camera_id}_{frame_number:06d}',  # cam1_000031
            ]

            # Priority 1: .raw files (most accurate)
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.raw')
                if os.path.exists(path):
                    return path

            # Priority 2: .npy files
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.npy')
                if os.path.exists(path):
                    return path

            # Priority 3: .png files
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.png')
                if os.path.exists(path):
                    return path

    elif structure == 'single_camera':
        # Detect true depth folder (prioritizes folder with .raw files)
        depth_folder_name = detect_true_depth_folder(trial_path, None)

        if depth_folder_name:
            depth_folder = os.path.join(trial_path, depth_folder_name)

            # Try different naming patterns
            # Include patterns with folder name in filename (e.g., _Depth_Color_0230.raw)
            patterns = [
                f'_{depth_folder_name}_{frame_number:04d}',  # _Depth_Color_0031
                f'_{depth_folder_name}_{frame_number:06d}',  # _Depth_Color_000031
                f'_Depth_{frame_number:04d}',  # _Depth_0031
                f'_Depth_{frame_number:06d}',  # _Depth_000031
                f'frame_{frame_number:06d}',   # frame_000031
                f'frame_{frame_number:04d}',   # frame_0031
                f'Depth_{frame_number:06d}',   # Depth_000031
                f'Depth_{frame_number:04d}',   # Depth_0031
            ]

            # Priority 1: .raw files (most accurate)
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.raw')
                if os.path.exists(path):
                    return path

            # Priority 2: .npy files
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.npy')
                if os.path.exists(path):
                    return path

            # Priority 3: .png files
            for pattern in patterns:
                path = os.path.join(depth_folder, f'{pattern}.png')
                if os.path.exists(path):
                    return path

    return None


def load_color_flexible(trial_path: str, camera_id: Optional[str], frame_number: int) -> np.ndarray:
    """
    Load color image with flexible structure detection

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera structure)
        frame_number: Frame number

    Returns:
        color_img: BGR color image (H, W, 3)

    Raises:
        FileNotFoundError: If color file not found
        ValueError: If color file cannot be read
    """
    color_path = find_color_file(trial_path, camera_id, frame_number)

    if color_path is None:
        structure = detect_folder_structure(trial_path)
        raise FileNotFoundError(
            f"Color image not found for frame {frame_number}\n"
            f"Trial path: {trial_path}\n"
            f"Camera ID: {camera_id}\n"
            f"Detected structure: {structure}"
        )

    color_img = cv2.imread(color_path)
    if color_img is None:
        raise ValueError(f"Failed to read color image: {color_path}")

    print(f"✅ Loaded color ({color_img.shape}): {color_path}")
    return color_img


def load_depth_flexible(trial_path: str, camera_id: Optional[str], frame_number: int,
                       depth_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Load depth data with flexible structure detection

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera)
        frame_number: Frame number
        depth_shape: Shape for .raw files (height, width). If None, auto-detects from file size.

    Returns:
        depth_img: Depth in meters (H, W)

    Raises:
        FileNotFoundError: If depth file not found
        ValueError: If depth file cannot be read
    """
    depth_path = find_depth_file(trial_path, camera_id, frame_number)

    if depth_path is None:
        structure = detect_folder_structure(trial_path)
        raise FileNotFoundError(
            f"Depth file not found for frame {frame_number}\n"
            f"Trial path: {trial_path}\n"
            f"Camera ID: {camera_id}\n"
            f"Detected structure: {structure}"
        )

    # Load based on extension
    if depth_path.endswith('.npy'):
        depth_img = np.load(depth_path).astype(np.float32)
    elif depth_path.endswith('.raw'):
        # Auto-detect shape if not provided
        if depth_shape is None:
            depth_shape = detect_depth_shape(depth_path)
            print(f"   Auto-detected depth shape: {depth_shape}")
        with open(depth_path, 'rb') as f:
            depth_img = np.frombuffer(f.read(), dtype=np.uint16).reshape(depth_shape).astype(np.float32)
    elif depth_path.endswith('.png'):
        # Depth stored as PNG (common in single camera structure)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")

    # Convert to meters if needed
    if depth_img.max() > 100:
        depth_img = depth_img / 1000.0

    # Calculate statistics for verification
    valid_depths = depth_img[depth_img > 0]
    if len(valid_depths) > 0:
        median_depth = np.median(valid_depths)
        mean_depth = valid_depths.mean()
        # print(f"✅ Loaded depth ({depth_img.shape}): {depth_path}")
        # print(f"   Depth stats: median={median_depth:.3f}m, mean={mean_depth:.3f}m, range=[{valid_depths.min():.3f}, {depth_img.max():.3f}]m")
    else:
        # print(f"✅ Loaded depth ({depth_img.shape}): {depth_path}")
        print(f"   ⚠️ No valid depth values (all zeros)")

    return depth_img


def detect_depth_shape(depth_path: str) -> Tuple[int, int]:
    """
    Detect depth shape from .raw file size

    Returns:
        (height, width) tuple
    """
    file_size = os.path.getsize(depth_path)
    num_pixels = file_size // 2  # uint16 = 2 bytes per pixel

    # Common resolutions
    common_shapes = [
        (480, 640),    # VGA
        (720, 1280),   # 720p
        (1080, 1920),  # 1080p
    ]

    for h, w in common_shapes:
        if h * w == num_pixels:
            return (h, w)

    # If no match, try to infer square root
    import math
    side = int(math.sqrt(num_pixels))
    if side * side == num_pixels:
        return (side, side)

    # Default fallback
    print(f"⚠️ Unknown depth shape for {num_pixels} pixels, using 480x640")
    return (480, 640)


def load_trial_data_flexible(trial_path: str, camera_id: Optional[str] = None,
                            frame_number: int = 1,
                            depth_shape: Optional[Tuple[int, int]] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load both color and depth with automatic structure detection

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera, required for multi-camera)
        frame_number: Frame number
        depth_shape: Shape for .raw files

    Returns:
        color_img: BGR color image or None
        depth_img: Depth in meters or None
    """
    structure = detect_folder_structure(trial_path)
    print(f"📁 Detected structure: {structure}")

    if structure == 'multi_camera' and camera_id is None:
        # Auto-detect first available camera
        cam_folders = sorted([d for d in os.listdir(trial_path)
                            if os.path.isdir(os.path.join(trial_path, d)) and d.startswith('cam')])
        if cam_folders:
            camera_id = cam_folders[0]
            print(f"🎥 Auto-selected camera: {camera_id}")
        else:
            print("⚠️ No camera folders found")
            return None, None

    color_img = None
    depth_img = None

    # Load color
    try:
        color_img = load_color_flexible(trial_path, camera_id, frame_number)
    except Exception as e:
        print(f"⚠️ Could not load color: {e}")

    # Load depth
    try:
        depth_img = load_depth_flexible(trial_path, camera_id, frame_number, depth_shape)
    except Exception as e:
        print(f"⚠️ Could not load depth: {e}")

    return color_img, depth_img


def list_available_cameras(trial_path: str) -> List[str]:
    """List available cameras in a trial folder"""
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        cam_folders = sorted([d for d in os.listdir(trial_path)
                            if os.path.isdir(os.path.join(trial_path, d)) and d.startswith('cam')])
        return cam_folders
    elif structure == 'single_camera':
        return ['single_camera']
    else:
        return []


def visualize_rgbd(color_img: Optional[np.ndarray],
                   depth_img: Optional[np.ndarray],
                   window_name: str = "RGBD Data",
                   wait_key: bool = True):
    """Visualize color and depth side-by-side"""
    if color_img is None and depth_img is None:
        print("⚠️ No data to visualize")
        return

    viz_list = []

    if color_img is not None:
        viz_list.append(color_img)

    if depth_img is not None:
        depth_viz = depth_img.copy()
        valid_depth = depth_viz[depth_viz > 0]

        if len(valid_depth) > 0:
            depth_min = valid_depth.min()
            depth_max = np.percentile(valid_depth, 95)
            depth_viz = np.clip(depth_viz, depth_min, depth_max)
            depth_viz = (depth_viz - depth_min) / (depth_max - depth_min) * 255
        else:
            depth_viz = depth_viz * 0

        depth_viz = depth_viz.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
        viz_list.append(depth_colored)

    if len(viz_list) == 2:
        h1, h2 = viz_list[0].shape[0], viz_list[1].shape[0]
        if h1 != h2:
            target_h = min(h1, h2)
            viz_list[0] = cv2.resize(viz_list[0], (int(viz_list[0].shape[1] * target_h / h1), target_h))
            viz_list[1] = cv2.resize(viz_list[1], (int(viz_list[1].shape[1] * target_h / h2), target_h))
        combined = np.hstack(viz_list)
    else:
        combined = viz_list[0]

    cv2.imshow(window_name, combined)

    if wait_key:
        print("\n📺 Visualization displayed. Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Insufficient arguments")
        print("\nUsage:")
        print("  Multi-camera: python load_trial_data_flexible.py <trial_path> <camera_id> <frame_number>")
        print("  Single-camera: python load_trial_data_flexible.py <trial_path> <frame_number>")
        print("\nExamples:")
        print("  python load_trial_data_flexible.py sample_raw_data/trial_1 cam1 31")
        print("  python load_trial_data_flexible.py sample_raw_data/1 230")
        sys.exit(1)

    trial_path = sys.argv[1]

    if not os.path.isdir(trial_path):
        print(f"❌ Error: Trial path does not exist: {trial_path}")
        sys.exit(1)

    # Detect structure
    structure = detect_folder_structure(trial_path)
    print(f"\n📁 Detected structure: {structure}")

    # List available cameras
    cameras = list_available_cameras(trial_path)
    if cameras:
        print(f"🎥 Available cameras: {', '.join(cameras)}")

    # Parse arguments based on structure
    if structure == 'multi_camera':
        if len(sys.argv) < 4:
            print(f"\n❌ Error: Multi-camera structure requires camera_id")
            print(f"\nUsage: python load_trial_data_flexible.py {trial_path} <camera_id> <frame_number>")
            print(f"\nAvailable cameras: {', '.join(cameras)}")
            sys.exit(1)

        camera_id = sys.argv[2]
        try:
            frame_number = int(sys.argv[3])
        except ValueError:
            print(f"❌ Error: Frame number must be an integer, got '{sys.argv[3]}'")
            sys.exit(1)

    else:  # single_camera
        camera_id = None
        try:
            frame_number = int(sys.argv[2])
        except (ValueError, IndexError):
            if len(sys.argv) < 3:
                print(f"\n❌ Error: Frame number required")
            else:
                print(f"❌ Error: Frame number must be an integer, got '{sys.argv[2]}'")
            sys.exit(1)

    # Load data
    print(f"\n🔄 Loading data...")
    print(f"   Trial: {trial_path}")
    if camera_id:
        print(f"   Camera: {camera_id}")
    print(f"   Frame: {frame_number}")
    print()

    color_img, depth_img = load_trial_data_flexible(trial_path, camera_id, frame_number)

    # Print info
    print("\n" + "="*60)
    print("📊 LOADED DATA SUMMARY")
    print("="*60)

    if color_img is not None:
        print(f"\n🎨 COLOR IMAGE:")
        print(f"   Shape: {color_img.shape}")
        print(f"   Dtype: {color_img.dtype}")
        print(f"   Size: {color_img.nbytes / 1024:.2f} KB")
    else:
        print(f"\n🎨 COLOR IMAGE: Not loaded")

    if depth_img is not None:
        valid_depths = depth_img[depth_img > 0]
        print(f"\n📏 DEPTH IMAGE:")
        print(f"   Shape: {depth_img.shape}")
        print(f"   Dtype: {depth_img.dtype}")
        if len(valid_depths) > 0:
            print(f"   Range: [{valid_depths.min():.3f}, {depth_img.max():.3f}] meters")
            print(f"   Mean: {valid_depths.mean():.3f} meters")
        print(f"   Size: {depth_img.nbytes / 1024:.2f} KB")
    else:
        print(f"\n📏 DEPTH IMAGE: Not loaded")

    print("="*60 + "\n")

    # Visualize
    if color_img is not None or depth_img is not None:
        visualize_rgbd(color_img, depth_img,
                      window_name=f"Frame {frame_number}",
                      wait_key=True)
    else:
        print("❌ No data loaded to visualize")
        sys.exit(1)


if __name__ == "__main__":
    main()

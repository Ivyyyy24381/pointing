"""
Refactored Data Loading Module - Example Implementation

This is a modernized version of load_trial_data_flexible.py demonstrating:
- Complete type hints with type aliases
- Custom exception hierarchy
- Pathlib instead of os.path
- Async I/O for concurrent loading
- Dataclasses for configuration
- Context managers for resource management
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import aiofiles
import cv2
import numpy as np
import numpy.typing as npt

# Type aliases for clarity
FolderStructure = Literal['multi_camera', 'single_camera', 'unknown']
NDArrayFloat = npt.NDArray[np.float32]
NDArrayUInt8 = npt.NDArray[np.uint8]
DepthShape = Tuple[int, int]


# =============================================================================
# Custom Exceptions
# =============================================================================

class DataLoadingError(Exception):
    """Base exception for data loading errors."""
    pass


class ColorImageNotFoundError(DataLoadingError):
    """Color image file not found."""

    def __init__(self, trial_path: Path, frame_number: int, camera_id: Optional[str] = None):
        self.trial_path = trial_path
        self.frame_number = frame_number
        self.camera_id = camera_id
        msg = f"Color image not found: frame={frame_number}, camera={camera_id}, path={trial_path}"
        super().__init__(msg)


class DepthImageNotFoundError(DataLoadingError):
    """Depth image file not found."""

    def __init__(self, trial_path: Path, frame_number: int, camera_id: Optional[str] = None):
        self.trial_path = trial_path
        self.frame_number = frame_number
        self.camera_id = camera_id
        msg = f"Depth image not found: frame={frame_number}, camera={camera_id}, path={trial_path}"
        super().__init__(msg)


class InvalidImageError(DataLoadingError):
    """Image file is corrupted or cannot be read."""
    pass


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass(frozen=True)
class TrialConfig:
    """Configuration for a trial directory."""
    trial_path: Path
    camera_id: Optional[str] = None
    structure: FolderStructure = 'unknown'

    def __post_init__(self) -> None:
        """Validate trial configuration."""
        if not self.trial_path.exists():
            raise ValueError(f"Trial path does not exist: {self.trial_path}")
        if not self.trial_path.is_dir():
            raise ValueError(f"Trial path is not a directory: {self.trial_path}")


# =============================================================================
# Folder Structure Detection
# =============================================================================

def detect_folder_structure(trial_path: Path | str) -> FolderStructure:
    """
    Detect which folder structure is used.

    Args:
        trial_path: Path to trial folder

    Returns:
        'multi_camera': trial_path/cam1/color/frame_XXXXXX.png
        'single_camera': trial_path/Color/_Color_XXXX.png
        'unknown': Could not detect structure

    Example:
        >>> trial_path = Path("trial_input/1")
        >>> structure = detect_folder_structure(trial_path)
        >>> print(structure)
        'single_camera'
    """
    trial_path = Path(trial_path)

    # Check for multi-camera structure (cam1, cam2, cam3)
    cam_folders = [
        d.name for d in trial_path.iterdir()
        if d.is_dir() and d.name.startswith('cam')
    ]

    if cam_folders:
        # Verify at least one camera has color folder
        for cam in cam_folders:
            color_lower = trial_path / cam / 'color'
            color_upper = trial_path / cam / 'Color'
            if color_lower.is_dir() or color_upper.is_dir():
                return 'multi_camera'

    # Check for single camera structure
    color_folder = trial_path / 'Color'
    color_folder_lower = trial_path / 'color'

    if color_folder.is_dir() or color_folder_lower.is_dir():
        return 'single_camera'

    return 'unknown'


# =============================================================================
# File Path Finding (Synchronous)
# =============================================================================

def find_color_file(
    trial_path: Path,
    camera_id: Optional[str],
    frame_number: int
) -> Optional[Path]:
    """
    Find color file with flexible naming patterns.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera)
        frame_number: Frame number

    Returns:
        Path to color file, or None if not found

    Raises:
        ValueError: If camera_id required but not provided
    """
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if camera_id is None:
            raise ValueError("camera_id required for multi_camera structure")

        # Try both 'color' and 'Color' folders
        for folder_name in ['color', 'Color']:
            color_folder = trial_path / camera_id / folder_name
            if not color_folder.is_dir():
                continue

            # Try different naming patterns
            patterns = [
                f'frame_{frame_number:06d}',
                f'Color_{frame_number:06d}',
                f'{camera_id}_{frame_number:06d}',
            ]

            for pattern in patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    path = color_folder / f'{pattern}{ext}'
                    if path.exists():
                        return path

    elif structure == 'single_camera':
        # Try both 'Color' and 'color' folders
        for folder_name in ['Color', 'color']:
            color_folder = trial_path / folder_name
            if not color_folder.is_dir():
                continue

            # Try different naming patterns
            patterns = [
                f'_Color_{frame_number:04d}',
                f'_Color_{frame_number:06d}',
                f'frame_{frame_number:06d}',
                f'frame_{frame_number:04d}',
                f'Color_{frame_number:06d}',
                f'Color_{frame_number:04d}',
            ]

            for pattern in patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    path = color_folder / f'{pattern}{ext}'
                    if path.exists():
                        return path

    return None


def find_depth_file(
    trial_path: Path,
    camera_id: Optional[str],
    frame_number: int
) -> Optional[Path]:
    """
    Find depth file with flexible naming patterns.

    Prioritizes .raw files (most accurate), then .npy, then .png.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera)
        frame_number: Frame number

    Returns:
        Path to depth file, or None if not found
    """
    structure = detect_folder_structure(trial_path)

    if structure == 'multi_camera':
        if camera_id is None:
            raise ValueError("camera_id required for multi_camera structure")

        # Check depth folder
        for folder_name in ['depth', 'Depth']:
            depth_folder = trial_path / camera_id / folder_name
            if not depth_folder.is_dir():
                continue

            # Try different naming patterns
            patterns = [
                f'frame_{frame_number:06d}',
                f'Depth_{frame_number:06d}',
                f'{camera_id}_{frame_number:06d}',
            ]

            # Priority 1: .raw files (most accurate)
            for pattern in patterns:
                path = depth_folder / f'{pattern}.raw'
                if path.exists():
                    return path

            # Priority 2: .npy files
            for pattern in patterns:
                path = depth_folder / f'{pattern}.npy'
                if path.exists():
                    return path

            # Priority 3: .png files
            for pattern in patterns:
                path = depth_folder / f'{pattern}.png'
                if path.exists():
                    return path

    elif structure == 'single_camera':
        # Try both 'Depth' and 'depth' folders
        for folder_name in ['Depth', 'depth', 'Depth_Color']:
            depth_folder = trial_path / folder_name
            if not depth_folder.is_dir():
                continue

            # Try different naming patterns
            patterns = [
                f'_{folder_name}_{frame_number:04d}',
                f'_{folder_name}_{frame_number:06d}',
                f'_Depth_{frame_number:04d}',
                f'_Depth_{frame_number:06d}',
                f'frame_{frame_number:06d}',
                f'frame_{frame_number:04d}',
            ]

            # Priority order: .raw > .npy > .png
            for ext in ['.raw', '.npy', '.png']:
                for pattern in patterns:
                    path = depth_folder / f'{pattern}{ext}'
                    if path.exists():
                        return path

    return None


# =============================================================================
# Synchronous Data Loading
# =============================================================================

def load_color_flexible(
    trial_path: Path | str,
    camera_id: Optional[str],
    frame_number: int
) -> NDArrayUInt8:
    """
    Load color image with flexible structure detection.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera structure)
        frame_number: Frame number

    Returns:
        BGR color image (H, W, 3) as uint8

    Raises:
        ColorImageNotFoundError: If color file not found
        InvalidImageError: If color file cannot be read
    """
    trial_path = Path(trial_path)
    color_path = find_color_file(trial_path, camera_id, frame_number)

    if color_path is None:
        raise ColorImageNotFoundError(trial_path, frame_number, camera_id)

    color_img = cv2.imread(str(color_path))
    if color_img is None:
        raise InvalidImageError(f"Failed to read color image: {color_path}")

    print(f"✅ Loaded color ({color_img.shape}): {color_path}")
    return color_img


def detect_depth_shape(depth_path: Path) -> DepthShape:
    """
    Detect depth shape from .raw file size.

    Args:
        depth_path: Path to .raw depth file

    Returns:
        (height, width) tuple
    """
    file_size = depth_path.stat().st_size
    num_pixels = file_size // 2  # uint16 = 2 bytes per pixel

    # Common resolutions
    common_shapes: list[DepthShape] = [
        (480, 640),    # VGA
        (720, 1280),   # 720p
        (1080, 1920),  # 1080p
    ]

    for h, w in common_shapes:
        if h * w == num_pixels:
            return (h, w)

    # Try square root
    import math
    side = int(math.sqrt(num_pixels))
    if side * side == num_pixels:
        return (side, side)

    # Default fallback
    print(f"⚠️ Unknown depth shape for {num_pixels} pixels, using 480x640")
    return (480, 640)


def load_depth_flexible(
    trial_path: Path | str,
    camera_id: Optional[str],
    frame_number: int,
    depth_shape: Optional[DepthShape] = None
) -> NDArrayFloat:
    """
    Load depth data with flexible structure detection.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (None for single camera)
        frame_number: Frame number
        depth_shape: Shape for .raw files (height, width). Auto-detects if None.

    Returns:
        Depth in meters (H, W) as float32

    Raises:
        DepthImageNotFoundError: If depth file not found
        InvalidImageError: If depth file cannot be read
    """
    trial_path = Path(trial_path)
    depth_path = find_depth_file(trial_path, camera_id, frame_number)

    if depth_path is None:
        raise DepthImageNotFoundError(trial_path, frame_number, camera_id)

    # Load based on extension
    if depth_path.suffix == '.npy':
        depth_img = np.load(depth_path).astype(np.float32)

    elif depth_path.suffix == '.raw':
        # Auto-detect shape if not provided
        if depth_shape is None:
            depth_shape = detect_depth_shape(depth_path)
            print(f"   Auto-detected depth shape: {depth_shape}")

        with open(depth_path, 'rb') as f:
            depth_img = np.frombuffer(
                f.read(), dtype=np.uint16
            ).reshape(depth_shape).astype(np.float32)

    elif depth_path.suffix == '.png':
        depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise InvalidImageError(f"Failed to read depth PNG: {depth_path}")
        depth_img = depth_img.astype(np.float32)

    else:
        raise InvalidImageError(f"Unsupported depth format: {depth_path.suffix}")

    # Convert to meters if needed (check if values are in millimeters)
    if depth_img.max() > 100:
        depth_img = depth_img / 1000.0

    return depth_img


# =============================================================================
# Async Data Loading
# =============================================================================

async def load_color_async(
    trial_path: Path,
    camera_id: Optional[str],
    frame_number: int
) -> NDArrayUInt8:
    """
    Asynchronously load color image.

    Uses thread pool for I/O-bound operations to avoid blocking.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier
        frame_number: Frame number

    Returns:
        BGR color image (H, W, 3)

    Raises:
        ColorImageNotFoundError: If color file not found
        InvalidImageError: If image cannot be read
    """
    # Find file in thread pool (I/O operation)
    color_path = await asyncio.to_thread(
        find_color_file, trial_path, camera_id, frame_number
    )

    if color_path is None:
        raise ColorImageNotFoundError(trial_path, frame_number, camera_id)

    # Load image in thread pool (OpenCV is CPU-bound)
    color_img = await asyncio.to_thread(cv2.imread, str(color_path))

    if color_img is None:
        raise InvalidImageError(f"Failed to read color image: {color_path}")

    return color_img


async def load_depth_async(
    trial_path: Path,
    camera_id: Optional[str],
    frame_number: int,
    depth_shape: Optional[DepthShape] = None
) -> NDArrayFloat:
    """
    Asynchronously load depth data.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier
        frame_number: Frame number
        depth_shape: Shape for .raw files

    Returns:
        Depth in meters (H, W)

    Raises:
        DepthImageNotFoundError: If depth file not found
        InvalidImageError: If depth cannot be read
    """
    # Find file in thread pool
    depth_path = await asyncio.to_thread(
        find_depth_file, trial_path, camera_id, frame_number
    )

    if depth_path is None:
        raise DepthImageNotFoundError(trial_path, frame_number, camera_id)

    # Load based on extension
    if depth_path.suffix == '.npy':
        depth_img = await asyncio.to_thread(np.load, depth_path)
        depth_img = depth_img.astype(np.float32)

    elif depth_path.suffix == '.raw':
        # Auto-detect shape if needed
        if depth_shape is None:
            depth_shape = await asyncio.to_thread(detect_depth_shape, depth_path)

        # Read raw file asynchronously
        async with aiofiles.open(depth_path, 'rb') as f:
            raw_data = await f.read()

        depth_img = np.frombuffer(
            raw_data, dtype=np.uint16
        ).reshape(depth_shape).astype(np.float32)

    elif depth_path.suffix == '.png':
        depth_img = await asyncio.to_thread(
            cv2.imread, str(depth_path), cv2.IMREAD_UNCHANGED
        )
        if depth_img is None:
            raise InvalidImageError(f"Failed to read depth PNG: {depth_path}")
        depth_img = depth_img.astype(np.float32)

    else:
        raise InvalidImageError(f"Unsupported depth format: {depth_path.suffix}")

    # Convert to meters if needed
    if depth_img.max() > 100:
        depth_img = depth_img / 1000.0

    return depth_img


async def load_trial_data_async(
    trial_path: Path | str,
    camera_id: Optional[str] = None,
    frame_number: int = 1,
    depth_shape: Optional[DepthShape] = None
) -> Tuple[Optional[NDArrayUInt8], Optional[NDArrayFloat]]:
    """
    Load both color and depth concurrently using asyncio.

    This is approximately 2x faster than sequential loading for I/O-bound operations.

    Args:
        trial_path: Path to trial folder
        camera_id: Camera identifier (auto-detected if None for multi-camera)
        frame_number: Frame number
        depth_shape: Shape for .raw files

    Returns:
        (color_img, depth_img) tuple, either can be None if loading failed

    Example:
        >>> async def main():
        ...     color, depth = await load_trial_data_async(
        ...         Path("trial_input/1"), None, 1
        ...     )
        ...     print(f"Loaded: color={color.shape}, depth={depth.shape}")
        >>> asyncio.run(main())
    """
    trial_path = Path(trial_path)

    # Create concurrent tasks
    color_task = asyncio.create_task(
        load_color_async(trial_path, camera_id, frame_number)
    )
    depth_task = asyncio.create_task(
        load_depth_async(trial_path, camera_id, frame_number, depth_shape)
    )

    # Wait for both to complete, catching exceptions
    results = await asyncio.gather(
        color_task, depth_task, return_exceptions=True
    )

    # Process results
    color_img = results[0] if not isinstance(results[0], Exception) else None
    depth_img = results[1] if not isinstance(results[1], Exception) else None

    # Log exceptions
    if isinstance(results[0], Exception):
        print(f"⚠️ Could not load color: {results[0]}")
    if isinstance(results[1], Exception):
        print(f"⚠️ Could not load depth: {results[1]}")

    return color_img, depth_img


async def load_multiple_frames_async(
    trial_path: Path,
    frame_numbers: list[int],
    camera_id: Optional[str] = None,
    max_concurrent: int = 8
) -> list[Tuple[int, Optional[NDArrayUInt8], Optional[NDArrayFloat]]]:
    """
    Load multiple frames concurrently with controlled parallelism.

    Args:
        trial_path: Path to trial folder
        frame_numbers: List of frame numbers to load
        camera_id: Camera identifier
        max_concurrent: Maximum concurrent loads (to control memory usage)

    Returns:
        List of (frame_number, color_img, depth_img) tuples

    Example:
        >>> async def main():
        ...     frames = await load_multiple_frames_async(
        ...         Path("trial_input/1"),
        ...         frame_numbers=list(range(1, 101)),
        ...         max_concurrent=8
        ...     )
        ...     print(f"Loaded {len(frames)} frames")
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def load_with_semaphore(frame_num: int):
        async with semaphore:
            color, depth = await load_trial_data_async(trial_path, camera_id, frame_num)
            return (frame_num, color, depth)

    # Create tasks
    tasks = [load_with_semaphore(frame_num) for frame_num in frame_numbers]

    # Execute with progress
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        results.append(result)
        print(f"Loaded frame {result[0]}: {i}/{len(tasks)}", end='\r')

    print()  # New line
    return results


# =============================================================================
# Example Usage
# =============================================================================

async def example_async_usage() -> None:
    """Example of async data loading."""
    trial_path = Path("trial_input/1/single_camera")

    # Load single frame
    print("Loading single frame asynchronously...")
    color, depth = await load_trial_data_async(trial_path, None, 1)

    if color is not None:
        print(f"Color: {color.shape}, {color.dtype}")
    if depth is not None:
        print(f"Depth: {depth.shape}, {depth.dtype}, range=[{depth.min():.3f}, {depth.max():.3f}]m")

    # Load multiple frames
    print("\nLoading 10 frames concurrently...")
    frames = await load_multiple_frames_async(
        trial_path,
        frame_numbers=list(range(1, 11)),
        max_concurrent=4
    )

    print(f"\nSuccessfully loaded {sum(1 for _, c, d in frames if c is not None)} frames")


def example_sync_usage() -> None:
    """Example of synchronous data loading."""
    trial_path = Path("trial_input/1/single_camera")

    # Detect structure
    structure = detect_folder_structure(trial_path)
    print(f"Detected structure: {structure}")

    # Load frame
    try:
        color = load_color_flexible(trial_path, None, 1)
        depth = load_depth_flexible(trial_path, None, 1)

        print(f"Color: {color.shape}")
        print(f"Depth: {depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]m")

    except ColorImageNotFoundError as e:
        print(f"Color not found: {e}")
    except DepthImageNotFoundError as e:
        print(f"Depth not found: {e}")
    except DataLoadingError as e:
        print(f"Loading error: {e}")


if __name__ == "__main__":
    # Run sync example
    print("=" * 60)
    print("Synchronous Example")
    print("=" * 60)
    example_sync_usage()

    # Run async example
    print("\n" + "=" * 60)
    print("Asynchronous Example")
    print("=" * 60)
    asyncio.run(example_async_usage())

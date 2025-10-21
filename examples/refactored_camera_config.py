"""
Refactored Camera Configuration Module - Example Implementation

Demonstrates:
- Dataclasses for immutable configuration
- Type hints with validation
- Factory methods for common configurations
- Protocol-based interfaces
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Tuple
import numpy as np
import numpy.typing as npt


# =============================================================================
# Type Aliases
# =============================================================================

ImageShape = Tuple[int, int]  # (height, width)
ImageRGB = npt.NDArray[np.uint8]  # (H, W, 3)
DepthMap = npt.NDArray[np.float32]  # (H, W)


# =============================================================================
# Camera Intrinsics Protocol
# =============================================================================

class CameraIntrinsicsProtocol(Protocol):
    """Protocol for camera intrinsic parameters (duck typing)."""

    @property
    def fx(self) -> float:
        """Focal length in x-direction (pixels)."""
        ...

    @property
    def fy(self) -> float:
        """Focal length in y-direction (pixels)."""
        ...

    @property
    def cx(self) -> float:
        """Principal point x-coordinate (pixels)."""
        ...

    @property
    def cy(self) -> float:
        """Principal point y-coordinate (pixels)."""
        ...

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """Convert to 3x3 intrinsic matrix."""
        ...


# =============================================================================
# Camera Parameters Dataclass
# =============================================================================

@dataclass(frozen=True)
class CameraParams:
    """
    Immutable camera intrinsic parameters.

    Attributes:
        fx: Focal length in x-direction (pixels)
        fy: Focal length in y-direction (pixels)
        cx: Principal point x-coordinate (pixels)
        cy: Principal point y-coordinate (pixels)
    """
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        """Validate camera parameters."""
        if self.fx <= 0:
            raise ValueError(f"fx must be positive, got {self.fx}")
        if self.fy <= 0:
            raise ValueError(f"fy must be positive, got {self.fy}")
        if self.cx < 0:
            raise ValueError(f"cx must be non-negative, got {self.cx}")
        if self.cy < 0:
            raise ValueError(f"cy must be non-negative, got {self.cy}")

    @classmethod
    def from_resolution(cls, width: int, height: int) -> CameraParams:
        """
        Auto-detect camera parameters from image resolution.

        Uses known RealSense camera intrinsics for common resolutions.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            CameraParams for the given resolution

        Example:
            >>> params = CameraParams.from_resolution(640, 480)
            >>> print(params.fx, params.fy)
            615.0 615.0
        """
        # RealSense D435 typical intrinsics
        if width == 640 and height == 480:
            return cls(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
        elif width == 1280 and height == 720:
            return cls(fx=922.5, fy=922.5, cx=640.0, cy=360.0)
        elif width == 1920 and height == 1080:
            return cls(fx=1383.75, fy=1383.75, cx=960.0, cy=540.0)
        else:
            # Generic pinhole camera assumption
            # Typical FOV ~60 degrees -> f ≈ w / (2 * tan(30°)) ≈ 0.9 * w
            return cls(
                fx=width * 0.9,
                fy=width * 0.9,
                cx=width / 2.0,
                cy=height / 2.0
            )

    @classmethod
    def from_image(cls, image: ImageRGB | DepthMap) -> CameraParams:
        """
        Create camera parameters from image shape.

        Args:
            image: RGB or depth image

        Returns:
            CameraParams auto-detected from image dimensions

        Example:
            >>> import numpy as np
            >>> image = np.zeros((480, 640, 3), dtype=np.uint8)
            >>> params = CameraParams.from_image(image)
            >>> print(params.fx)
            615.0
        """
        h, w = image.shape[:2]
        return cls.from_resolution(w, h)

    def to_matrix(self) -> npt.NDArray[np.float64]:
        """
        Convert to 3x3 camera intrinsic matrix.

        Returns:
            K matrix in standard form:
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

        Example:
            >>> params = CameraParams(615.0, 615.0, 320.0, 240.0)
            >>> K = params.to_matrix()
            >>> print(K)
            [[615.   0. 320.]
             [  0. 615. 240.]
             [  0.   0.   1.]]
        """
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with camera parameters
        """
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy
        }

    @classmethod
    def from_dict(cls, data: dict) -> CameraParams:
        """
        Create from dictionary.

        Args:
            data: Dictionary with fx, fy, cx, cy keys

        Returns:
            CameraParams instance

        Raises:
            KeyError: If required keys missing
            ValueError: If values invalid
        """
        return cls(
            fx=float(data['fx']),
            fy=float(data['fy']),
            cx=float(data['cx']),
            cy=float(data['cy'])
        )

    def pixel_to_ray(self, u: float, v: float) -> npt.NDArray[np.float64]:
        """
        Convert pixel coordinates to normalized ray direction.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate

        Returns:
            Normalized ray direction [x, y, z] with z=1

        Example:
            >>> params = CameraParams(615.0, 615.0, 320.0, 240.0)
            >>> ray = params.pixel_to_ray(320, 240)  # Center pixel
            >>> print(ray)
            [0. 0. 1.]
        """
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        z = 1.0
        ray = np.array([x, y, z], dtype=np.float64)
        return ray / np.linalg.norm(ray)

    def backproject_pixel(
        self,
        u: float,
        v: float,
        depth: float
    ) -> npt.NDArray[np.float64]:
        """
        Backproject pixel with depth to 3D point in camera frame.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            depth: Depth in meters

        Returns:
            3D point [x, y, z] in camera frame (meters)

        Example:
            >>> params = CameraParams(615.0, 615.0, 320.0, 240.0)
            >>> point_3d = params.backproject_pixel(320, 240, 1.5)
            >>> print(point_3d)
            [0.  0.  1.5]
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z], dtype=np.float64)

    def project_point(
        self,
        point_3d: npt.NDArray[np.float64]
    ) -> Tuple[float, float]:
        """
        Project 3D point to pixel coordinates.

        Args:
            point_3d: 3D point [x, y, z] in camera frame (meters)

        Returns:
            (u, v) pixel coordinates

        Example:
            >>> params = CameraParams(615.0, 615.0, 320.0, 240.0)
            >>> u, v = params.project_point(np.array([0.0, 0.0, 1.5]))
            >>> print(u, v)
            320.0 240.0
        """
        x, y, z = point_3d
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return (u, v)

    def scale(self, factor: float) -> CameraParams:
        """
        Scale camera parameters (for image resizing).

        Args:
            factor: Scale factor (0.5 for half-size, 2.0 for double-size)

        Returns:
            New CameraParams with scaled values

        Example:
            >>> params = CameraParams(615.0, 615.0, 320.0, 240.0)
            >>> half_params = params.scale(0.5)
            >>> print(half_params.fx, half_params.cx)
            307.5 160.0
        """
        return CameraParams(
            fx=self.fx * factor,
            fy=self.fy * factor,
            cx=self.cx * factor,
            cy=self.cy * factor
        )


# =============================================================================
# Common Camera Presets
# =============================================================================

@dataclass(frozen=True)
class CameraPresets:
    """Pre-defined camera configurations for common devices."""

    REALSENSE_D435_VGA: CameraParams = field(
        default_factory=lambda: CameraParams(615.0, 615.0, 320.0, 240.0)
    )
    REALSENSE_D435_720P: CameraParams = field(
        default_factory=lambda: CameraParams(922.5, 922.5, 640.0, 360.0)
    )
    REALSENSE_D435_1080P: CameraParams = field(
        default_factory=lambda: CameraParams(1383.75, 1383.75, 960.0, 540.0)
    )

    @classmethod
    def get_preset(cls, name: str) -> CameraParams:
        """
        Get camera preset by name.

        Args:
            name: Preset name (e.g., 'REALSENSE_D435_VGA')

        Returns:
            CameraParams for the preset

        Raises:
            ValueError: If preset not found
        """
        presets = cls()
        if not hasattr(presets, name):
            raise ValueError(f"Unknown preset: {name}")
        return getattr(presets, name)


# =============================================================================
# Example Usage
# =============================================================================

def example_usage() -> None:
    """Example usage of camera configuration."""

    # Create from resolution
    print("=" * 60)
    print("Creating from resolution")
    print("=" * 60)
    params1 = CameraParams.from_resolution(640, 480)
    print(f"640x480: {params1}")
    print(f"Intrinsic matrix:\n{params1.to_matrix()}\n")

    # Create from image
    print("=" * 60)
    print("Creating from image")
    print("=" * 60)
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    params2 = CameraParams.from_image(image)
    print(f"Image shape {image.shape[:2]}: {params2}\n")

    # Use preset
    print("=" * 60)
    print("Using preset")
    print("=" * 60)
    params3 = CameraPresets.REALSENSE_D435_1080P
    print(f"RealSense D435 1080p: {params3}\n")

    # Backproject pixel to 3D
    print("=" * 60)
    print("Backprojection")
    print("=" * 60)
    u, v = 400, 300
    depth = 1.5
    point_3d = params1.backproject_pixel(u, v, depth)
    print(f"Pixel ({u}, {v}) at depth {depth}m → 3D point: {point_3d}")

    # Project 3D to pixel
    u_proj, v_proj = params1.project_point(point_3d)
    print(f"3D point {point_3d} → Pixel: ({u_proj:.1f}, {v_proj:.1f})\n")

    # Scale parameters
    print("=" * 60)
    print("Scaling parameters")
    print("=" * 60)
    params_half = params1.scale(0.5)
    print(f"Original: {params1}")
    print(f"Half-size: {params_half}\n")

    # Serialization
    print("=" * 60)
    print("Serialization")
    print("=" * 60)
    params_dict = params1.to_dict()
    print(f"Dict: {params_dict}")
    params_restored = CameraParams.from_dict(params_dict)
    print(f"Restored: {params_restored}")
    print(f"Equal: {params1 == params_restored}\n")


if __name__ == "__main__":
    example_usage()

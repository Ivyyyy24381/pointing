"""
Pointing vector calculation from 2D skeleton keypoints and depth.

Converts 2D keypoints to 3D and calculates pointing vectors.
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Point3D:
    """3D point in camera frame."""
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def to_dict(self) -> dict:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}


@dataclass
class PointingResult:
    """Pointing analysis result for a single frame."""
    frame_number: int
    wrist_3d: Optional[Point3D]
    elbow_3d: Optional[Point3D]
    shoulder_3d: Optional[Point3D]
    pointing_vector: Optional[np.ndarray]  # Unit vector from shoulder->wrist
    pointing_arm: str  # "left", "right", or "auto"

    def to_dict(self) -> dict:
        result = {
            "frame": self.frame_number,
            "pointing_arm": self.pointing_arm
        }

        if self.wrist_3d:
            result["wrist_3d"] = self.wrist_3d.to_dict()
        if self.elbow_3d:
            result["elbow_3d"] = self.elbow_3d.to_dict()
        if self.shoulder_3d:
            result["shoulder_3d"] = self.shoulder_3d.to_dict()
        if self.pointing_vector is not None:
            result["pointing_vector"] = self.pointing_vector.tolist()

        return result


class PointingCalculator:
    """
    Calculate 3D pointing vectors from 2D skeleton keypoints.

    Workflow:
    1. Load skeleton keypoints from Step 2
    2. Load depth maps from trial_input/
    3. Convert 2D keypoints to 3D using depth + camera intrinsics
    4. Calculate pointing vector from shoulder->elbow->wrist
    """

    # MediaPipe landmark indices
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self,
                 fx: float = 615.0,
                 fy: float = 615.0,
                 cx: float = 320.0,
                 cy: float = 240.0):
        """
        Initialize pointing calculator.

        Args:
            fx, fy: Focal lengths (pixels)
            cx, cy: Principal point (pixels)
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def pixel_to_3d(self, x: float, y: float, depth: float) -> Point3D:
        """
        Convert 2D pixel + depth to 3D point in camera frame.

        Args:
            x, y: Pixel coordinates
            depth: Depth in meters

        Returns:
            3D point in camera frame
        """
        z = depth
        x_3d = (x - self.cx) * z / self.fx
        y_3d = (y - self.cy) * z / self.fy

        return Point3D(x=x_3d, y=y_3d, z=z)

    def process_skeleton_frame(self,
                               landmarks_2d: List[Tuple[float, float, float]],
                               depth_map: np.ndarray,
                               pointing_arm: str = "auto") -> PointingResult:
        """
        Process a single skeleton frame to extract 3D pointing.

        Args:
            landmarks_2d: List of (x, y, visibility) tuples (33 MediaPipe landmarks)
            depth_map: Depth image (H, W) in meters
            pointing_arm: "left", "right", or "auto"

        Returns:
            PointingResult with 3D keypoints and pointing vector
        """
        frame_number = 0  # Will be set by caller

        # Determine which arm to use
        if pointing_arm == "auto":
            # Use arm that's higher (lower y value)
            left_wrist_y = landmarks_2d[self.LEFT_WRIST][1]
            right_wrist_y = landmarks_2d[self.RIGHT_WRIST][1]
            pointing_arm = "left" if left_wrist_y < right_wrist_y else "right"

        # Get keypoint indices based on pointing arm
        if pointing_arm == "left":
            wrist_idx = self.LEFT_WRIST
            elbow_idx = self.LEFT_ELBOW
            shoulder_idx = self.LEFT_SHOULDER
        else:
            wrist_idx = self.RIGHT_WRIST
            elbow_idx = self.RIGHT_ELBOW
            shoulder_idx = self.RIGHT_SHOULDER

        # Extract 2D keypoints
        wrist_2d = landmarks_2d[wrist_idx]
        elbow_2d = landmarks_2d[elbow_idx]
        shoulder_2d = landmarks_2d[shoulder_idx]

        # Get depth values
        h, w = depth_map.shape
        wrist_x, wrist_y = int(wrist_2d[0]), int(wrist_2d[1])
        elbow_x, elbow_y = int(elbow_2d[0]), int(elbow_2d[1])
        shoulder_x, shoulder_y = int(shoulder_2d[0]), int(shoulder_2d[1])

        # Check bounds and get depth
        wrist_3d = None
        elbow_3d = None
        shoulder_3d = None

        if 0 <= wrist_y < h and 0 <= wrist_x < w:
            wrist_depth = depth_map[wrist_y, wrist_x]
            if wrist_depth > 0:
                wrist_3d = self.pixel_to_3d(wrist_2d[0], wrist_2d[1], wrist_depth)

        if 0 <= elbow_y < h and 0 <= elbow_x < w:
            elbow_depth = depth_map[elbow_y, elbow_x]
            if elbow_depth > 0:
                elbow_3d = self.pixel_to_3d(elbow_2d[0], elbow_2d[1], elbow_depth)

        if 0 <= shoulder_y < h and 0 <= shoulder_x < w:
            shoulder_depth = depth_map[shoulder_y, shoulder_x]
            if shoulder_depth > 0:
                shoulder_3d = self.pixel_to_3d(shoulder_2d[0], shoulder_2d[1], shoulder_depth)

        # Calculate pointing vector (shoulder -> wrist)
        pointing_vector = None
        if wrist_3d and shoulder_3d:
            vec = wrist_3d.to_numpy() - shoulder_3d.to_numpy()
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                pointing_vector = vec / norm

        return PointingResult(
            frame_number=frame_number,
            wrist_3d=wrist_3d,
            elbow_3d=elbow_3d,
            shoulder_3d=shoulder_3d,
            pointing_vector=pointing_vector,
            pointing_arm=pointing_arm
        )

    def process_trial(self,
                     skeleton_file: str,
                     depth_folder: str,
                     output_file: str) -> List[PointingResult]:
        """
        Process entire trial: skeleton + depth -> pointing vectors.

        Args:
            skeleton_file: Path to skeleton_2d.json from Step 2
            depth_folder: Path to depth folder (trial_input/trial_1/cam1/depth/)
            output_file: Output JSON file for results

        Returns:
            List of PointingResult
        """
        print(f"\n{'='*60}")
        print(f"Processing Trial: Pointing Calculation")
        print(f"{'='*60}")

        # Load skeleton data
        print(f"📁 Loading skeleton: {skeleton_file}")
        with open(skeleton_file, 'r') as f:
            skeleton_data = json.load(f)

        results = []
        depth_folder = Path(depth_folder)

        print(f"📁 Depth folder: {depth_folder}")
        print(f"🔄 Processing frames...")

        for frame_key, frame_data in skeleton_data.items():
            frame_num = frame_data['frame']
            landmarks_2d = frame_data['landmarks_2d']
            pointing_arm = frame_data.get('metadata', {}).get('pointing_arm', 'auto')

            # Load corresponding depth map
            depth_file = depth_folder / f"frame_{frame_num:06d}.npy"
            if not depth_file.exists():
                print(f"  ⚠️ Frame {frame_num}: Depth not found")
                continue

            depth_map = np.load(depth_file)

            # Process frame
            result = self.process_skeleton_frame(landmarks_2d, depth_map, pointing_arm)
            result.frame_number = frame_num
            results.append(result)

            if frame_num % 10 == 0:
                print(f"  Processed frame {frame_num}...", end='\r')

        print(f"\n✅ Processed {len(results)} frames")

        # Save results
        output_data = {
            f"frame_{r.frame_number:06d}": r.to_dict()
            for r in results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Saved pointing results to: {output_file}")

        return results

"""
Base classes for skeleton detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class SkeletonResult:
    """Result from skeleton detection on a single frame."""
    frame_number: int
    landmarks_2d: List[Tuple[float, float, float]]  # [(x, y, visibility), ...]
    landmarks_3d: Optional[List[Tuple[float, float, float]]] = None  # [(x, y, z), ...] in camera frame
    keypoint_names: Optional[List[str]] = None
    metadata: Optional[Dict] = None  # Additional info (e.g., pointing_arm, confidence)
    arm_vectors: Optional[Dict[str, List[float]]] = None  # Arm vectors (shoulder_to_wrist, etc.)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "frame": self.frame_number,
            "landmarks_2d": [[float(x), float(y), float(v)] for x, y, v in self.landmarks_2d],
        }

        if self.landmarks_3d:
            result["landmarks_3d"] = [[float(x), float(y), float(z)] for x, y, z in self.landmarks_3d]

        if self.keypoint_names:
            result["keypoint_names"] = self.keypoint_names

        if self.metadata:
            # Deep copy metadata and convert tuples to lists for JSON serialization
            metadata_copy = {}
            for key, value in self.metadata.items():
                if isinstance(value, tuple):
                    metadata_copy[key] = list(value)
                else:
                    metadata_copy[key] = value
            result["metadata"] = metadata_copy

        if self.arm_vectors:
            result["arm_vectors"] = self.arm_vectors

        return result


class SkeletonDetector(ABC):
    """Abstract base class for skeleton detectors."""

    @abstractmethod
    def detect_frame(self, image: np.ndarray, frame_number: int) -> Optional[SkeletonResult]:
        """
        Detect skeleton in a single frame.

        Args:
            image: RGB image (H, W, 3)
            frame_number: Frame number for tracking

        Returns:
            SkeletonResult if detection successful, None otherwise
        """
        pass

    @abstractmethod
    def detect_video(self, video_path: str) -> List[SkeletonResult]:
        """
        Detect skeleton in entire video.

        Args:
            video_path: Path to video file

        Returns:
            List of SkeletonResult for each frame
        """
        pass

    def detect_image_folder(self, folder_path: str, frame_pattern: str = "frame_{:06d}.png") -> List[SkeletonResult]:
        """
        Detect skeleton in folder of images.

        Args:
            folder_path: Path to folder containing images
            frame_pattern: Pattern for frame filenames

        Returns:
            List of SkeletonResult for each frame
        """
        import os
        import cv2
        from pathlib import Path

        results = []
        folder = Path(folder_path)

        # Find all image files
        image_files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))

        for i, img_path in enumerate(image_files, start=1):
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = self.detect_frame(image_rgb, i)
            if result:
                results.append(result)

        return results

    def save_results(self, results: List[SkeletonResult], output_path: str) -> None:
        """
        Save detection results to JSON file.

        Args:
            results: List of SkeletonResult
            output_path: Path to output JSON file
        """
        import json

        output_data = {
            f"frame_{result.frame_number:06d}": result.to_dict()
            for result in results
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"💾 Saved {len(results)} skeleton detections to: {output_path}")

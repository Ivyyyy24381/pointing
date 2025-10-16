"""
Subject detection for dog and baby in lower half of image.

This module provides detection for subjects (dog/baby) that are separate from human detection.
It focuses on the lower half of the image to avoid detecting adult humans.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class SubjectDetectionResult:
    """Result from subject detection"""
    subject_type: str  # 'dog' or 'baby'
    detection_region: str = "lower_half"

    # 2D keypoints in original image coordinates
    keypoints_2d: Optional[List[List[float]]] = None  # [[x, y, confidence], ...]

    # Bounding box (for dog detection)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)

    # 3D positions if depth available
    keypoints_3d: Optional[List[List[float]]] = None  # [[x, y, z], ...]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "subject_type": self.subject_type,
            "detection_region": self.detection_region,
            "bbox": list(self.bbox) if self.bbox else None,
            "keypoints_2d": self.keypoints_2d,
            "keypoints_3d": self.keypoints_3d
        }


class SubjectDetector:
    """Detector for dog and baby subjects in lower half of image"""

    def __init__(self, subject_type: str = 'dog', crop_ratio: float = 0.5):
        """
        Initialize subject detector.

        Args:
            subject_type: 'dog' or 'baby'
            crop_ratio: Ratio of lower image to process (0.5 = bottom 50%)
        """
        if subject_type not in ['dog', 'baby']:
            raise ValueError("subject_type must be 'dog' or 'baby'")

        self.subject_type = subject_type
        self.crop_ratio = crop_ratio

        # Initialize detector based on type
        if subject_type == 'dog':
            self._init_dog_detector()
        else:  # baby
            self._init_baby_detector()

    def _init_dog_detector(self):
        """Initialize MediaPipe Pose for dog skeleton detection"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            # Use same settings as human detection
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✅ MediaPipe Pose loaded for dog skeleton detection")
        except ImportError:
            print("⚠️ mediapipe not installed. Run: pip install mediapipe")
            self.pose = None
        except Exception as e:
            print(f"⚠️ Failed to load MediaPipe: {e}")
            self.pose = None

    def _init_baby_detector(self):
        """Initialize MediaPipe detector for baby"""
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print(f"✅ MediaPipe loaded for baby detection")
        except ImportError:
            print("⚠️ mediapipe not installed. Run: pip install mediapipe")
            self.pose = None
        except Exception as e:
            print(f"⚠️ Failed to load MediaPipe: {e}")
            self.pose = None

    def crop_to_lower_half(self, image: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Crop image to lower half.

        Args:
            image: Input image (H, W, C)

        Returns:
            cropped: Lower portion of image
            y_offset: Y-offset where crop starts
        """
        height = image.shape[0]
        y_offset = int(height * (1 - self.crop_ratio))
        cropped = image[y_offset:, :, :]
        return cropped, y_offset

    def map_keypoints_to_original(self, keypoints: List[List[float]], y_offset: int) -> List[List[float]]:
        """
        Map keypoints from cropped image to original image coordinates.

        Args:
            keypoints: Keypoints in cropped image [[x, y, conf], ...]
            y_offset: Y-offset of crop

        Returns:
            Mapped keypoints in original image coordinates
        """
        mapped = []
        for kp in keypoints:
            if len(kp) >= 2:
                x, y = kp[0], kp[1]
                rest = kp[2:] if len(kp) > 2 else []
                mapped.append([x, y + y_offset] + rest)
            else:
                mapped.append(kp)
        return mapped

    def detect_frame(self, image: np.ndarray, frame_number: int,
                     depth_image: Optional[np.ndarray] = None,
                     fx: float = 615.0, fy: float = 615.0,
                     cx: float = 320.0, cy: float = 240.0) -> Optional[SubjectDetectionResult]:
        """
        Detect subject in frame.

        Args:
            image: RGB image (H, W, 3)
            frame_number: Frame number
            depth_image: Depth image in meters (optional)
            fx, fy: Focal lengths
            cx, cy: Principal point

        Returns:
            SubjectDetectionResult or None if no detection
        """
        # Crop to lower half
        cropped_img, y_offset = self.crop_to_lower_half(image)

        if self.subject_type == 'dog':
            return self._detect_dog(cropped_img, y_offset, depth_image, fx, fy, cx, cy)
        else:  # baby
            return self._detect_baby(cropped_img, y_offset, depth_image, fx, fy, cx, cy)

    def _detect_dog(self, cropped_img: np.ndarray, y_offset: int,
                    depth_image: Optional[np.ndarray],
                    fx: float, fy: float, cx: float, cy: float) -> Optional[SubjectDetectionResult]:
        """Detect dog skeleton using MediaPipe Pose"""
        if self.pose is None:
            return None

        # Convert to RGB for MediaPipe
        import cv2
        image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) if cropped_img.shape[2] == 3 else cropped_img

        # Run MediaPipe detection on cropped (lower half) image
        results = self.pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        # Extract 2D keypoints from cropped image
        h, w = cropped_img.shape[:2]
        keypoints_2d_cropped = []

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            visibility = landmark.visibility
            keypoints_2d_cropped.append([x, y, visibility])

        # Map to original image coordinates
        keypoints_2d = self.map_keypoints_to_original(keypoints_2d_cropped, y_offset)

        # Compute 3D keypoints if depth image available
        keypoints_3d = None
        if depth_image is not None:
            keypoints_3d = self._compute_3d_keypoints(
                keypoints_2d, depth_image, fx, fy, cx, cy
            )

        # Compute bounding box from keypoints
        bbox = self._compute_bbox_from_keypoints(keypoints_2d)

        result = SubjectDetectionResult(
            subject_type='dog',
            bbox=bbox,
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d
        )

        return result

    def _detect_baby(self, cropped_img: np.ndarray, y_offset: int,
                     depth_image: Optional[np.ndarray],
                     fx: float, fy: float, cx: float, cy: float) -> Optional[SubjectDetectionResult]:
        """Detect baby using MediaPipe"""
        if self.pose is None:
            return None

        # Convert to RGB for MediaPipe
        import cv2
        image_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) if cropped_img.shape[2] == 3 else cropped_img

        # Run MediaPipe detection
        results = self.pose.process(image_rgb)

        if results.pose_landmarks is None:
            return None

        # Extract 2D keypoints from cropped image
        h, w = cropped_img.shape[:2]
        keypoints_2d_cropped = []

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            visibility = landmark.visibility
            keypoints_2d_cropped.append([x, y, visibility])

        # Map to original image coordinates
        keypoints_2d = self.map_keypoints_to_original(keypoints_2d_cropped, y_offset)

        # Compute 3D keypoints if depth image available
        keypoints_3d = None
        if depth_image is not None:
            keypoints_3d = self._compute_3d_keypoints(
                keypoints_2d, depth_image, fx, fy, cx, cy
            )

        # Compute bounding box from keypoints
        bbox = self._compute_bbox_from_keypoints(keypoints_2d)

        result = SubjectDetectionResult(
            subject_type='baby',
            bbox=bbox,
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d
        )

        return result

    def _compute_3d_keypoints(self, keypoints_2d: List[List[float]],
                               depth_image: np.ndarray,
                               fx: float, fy: float, cx: float, cy: float) -> List[List[float]]:
        """
        Convert 2D keypoints to 3D using depth image and camera intrinsics.

        Args:
            keypoints_2d: [[x, y, conf], ...] in pixels
            depth_image: Depth in meters (H, W)
            fx, fy: Focal lengths
            cx, cy: Principal point

        Returns:
            [[x, y, z], ...] in meters (world coordinates)
        """
        keypoints_3d = []
        h, w = depth_image.shape

        for kp in keypoints_2d:
            if len(kp) < 2:
                keypoints_3d.append([0.0, 0.0, 0.0])
                continue

            x_pixel, y_pixel = int(kp[0]), int(kp[1])

            # Bounds check
            if x_pixel < 0 or x_pixel >= w or y_pixel < 0 or y_pixel >= h:
                keypoints_3d.append([0.0, 0.0, 0.0])
                continue

            # Get depth value
            z = depth_image[y_pixel, x_pixel]

            # Check for valid depth
            if z <= 0.0 or np.isnan(z) or np.isinf(z):
                keypoints_3d.append([0.0, 0.0, 0.0])
                continue

            # Pinhole camera model: backproject to 3D
            x_3d = (x_pixel - cx) * z / fx
            y_3d = (y_pixel - cy) * z / fy
            z_3d = z

            keypoints_3d.append([float(x_3d), float(y_3d), float(z_3d)])

        return keypoints_3d

    def _compute_bbox_from_keypoints(self, keypoints_2d: List[List[float]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute bounding box from 2D keypoints.

        Args:
            keypoints_2d: [[x, y, conf], ...]

        Returns:
            (x1, y1, x2, y2) or None
        """
        if not keypoints_2d:
            return None

        valid_points = []
        for kp in keypoints_2d:
            if len(kp) >= 2:
                x, y = kp[0], kp[1]
                # Only use points with reasonable confidence if available
                if len(kp) >= 3 and kp[2] > 0.3:
                    valid_points.append((x, y))
                elif len(kp) < 3:
                    valid_points.append((x, y))

        if not valid_points:
            return None

        xs = [p[0] for p in valid_points]
        ys = [p[1] for p in valid_points]

        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))

        return (x1, y1, x2, y2)

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose') and self.pose is not None:
            self.pose.close()


# Import cv2 at module level for baby detection
try:
    import cv2
except ImportError:
    print("⚠️ opencv not installed")
    cv2 = None

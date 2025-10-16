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
        """Initialize YOLO detector for dogs"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO('yolov8n.pt')
            print(f"✅ YOLO model loaded for dog detection")
        except ImportError:
            print("⚠️ ultralytics not installed. Run: pip install ultralytics")
            self.yolo_model = None
        except Exception as e:
            print(f"⚠️ Failed to load YOLO model: {e}")
            self.yolo_model = None

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
        """Detect dog using YOLO"""
        if self.yolo_model is None:
            return None

        # Run YOLO detection
        results = self.yolo_model(cropped_img, verbose=False)

        DOG_CLASS = 16  # COCO dataset dog class

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == DOG_CLASS and conf >= 0.5:
                    # Get bounding box in cropped image
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Map bbox to original image coordinates
                    bbox_original = (x1, y1 + y_offset, x2, y2 + y_offset)

                    # For now, just return bbox
                    # TODO: Add DeepLabCut skeleton detection
                    result = SubjectDetectionResult(
                        subject_type='dog',
                        bbox=bbox_original,
                        keypoints_2d=None,  # Will add skeleton later
                        keypoints_3d=None
                    )

                    return result

        return None

    def _detect_baby(self, cropped_img: np.ndarray, y_offset: int,
                     depth_image: Optional[np.ndarray],
                     fx: float, fy: float, cx: float, cy: float) -> Optional[SubjectDetectionResult]:
        """Detect baby using MediaPipe"""
        if self.pose is None:
            return None

        # Convert to RGB for MediaPipe
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

        # TODO: Add 3D computation using depth image

        result = SubjectDetectionResult(
            subject_type='baby',
            keypoints_2d=keypoints_2d,
            keypoints_3d=None  # Will add 3D later
        )

        return result

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

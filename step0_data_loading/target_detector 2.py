"""
Target detection wrapper for automatic cup detection using YOLOv8.

This module provides a simple interface to detect cups in color images.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Detection:
    """Single detection result"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str
    center_3d: Optional[Tuple[float, float, float]] = None  # (x, y, z) in camera frame
    depth: Optional[float] = None  # depth at center in meters
    avg_depth: Optional[float] = None  # average depth within bbox in meters
    median_depth: Optional[float] = None  # median depth within bbox in meters

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box"""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "bbox_2d": {
                "x1": int(self.x1),
                "y1": int(self.y1),
                "x2": int(self.x2),
                "y2": int(self.y2),
                "center": [int(self.center[0]), int(self.center[1])]
            },
            "confidence": float(self.confidence),
            "label": self.label,
            "depth_at_center_m": float(self.depth) if self.depth is not None else None,
            "avg_depth_m": float(self.avg_depth) if self.avg_depth is not None else None,
            "median_depth_m": float(self.median_depth) if self.median_depth is not None else None,
            "position_3d_camera_frame": {
                "x": float(self.center_3d[0]) if self.center_3d else None,
                "y": float(self.center_3d[1]) if self.center_3d else None,
                "z": float(self.center_3d[2]) if self.center_3d else None
            } if self.center_3d else None
        }


class TargetDetector:
    """Wrapper for YOLO-based cup detection"""

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize target detector.

        Args:
            model_path: Path to YOLO model weights. If None, uses default path.
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path

        # Try to load model
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"✅ Loaded YOLO model from: {model_path}")
        except ImportError:
            print("⚠️ ultralytics not installed. Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"⚠️ Failed to load model: {e}")
            self.model = None

    def detect(self, color_img: np.ndarray, depth_img: Optional[np.ndarray] = None,
               target_label: str = "cup",
               fx: float = 615.0, fy: float = 615.0,
               cx: float = 320.0, cy: float = 240.0) -> List[Detection]:
        """
        Detect targets in color image and compute 3D positions.

        Args:
            color_img: Color image (H, W, 3) BGR
            depth_img: Depth image (H, W) in meters (optional)
            target_label: Label to filter detections (default: "cup")
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point

        Returns:
            List of Detection objects with 3D positions if depth provided
        """
        if self.model is None:
            return []

        # Run detection
        results = self.model(color_img, verbose=False)

        # Parse detections
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                conf = float(box.conf[0])

                # Filter by label and confidence
                if label == target_label and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    det = Detection(x1, y1, x2, y2, conf, label)

                    # Compute 3D position if depth available
                    if depth_img is not None:
                        center_x, center_y = det.center

                        # Compute average and median depth within bbox
                        bbox_depth = depth_img[y1:y2, x1:x2]
                        valid_depth = bbox_depth[bbox_depth > 0]

                        if len(valid_depth) > 0:
                            det.avg_depth = float(valid_depth.mean())
                            det.median_depth = float(np.median(valid_depth))

                            # Use average non-zero depth for 3D position calculation
                            depth_value = det.avg_depth
                            det.depth = depth_value

                            # Convert to 3D camera frame coordinates using bbox center
                            z = depth_value
                            x = (center_x - cx) * z / fx
                            y = (center_y - cy) * z / fy
                            det.center_3d = (x, y, z)
                        else:
                            # Fallback: try depth at center pixel if no valid depth in bbox
                            if 0 <= center_y < depth_img.shape[0] and 0 <= center_x < depth_img.shape[1]:
                                depth_value = depth_img[center_y, center_x]
                                if depth_value > 0:
                                    z = depth_value
                                    x = (center_x - cx) * z / fx
                                    y = (center_y - cy) * z / fy
                                    det.center_3d = (x, y, z)
                                    det.depth = depth_value

                    detections.append(det)

        return detections

    def draw_detections(self, color_img: np.ndarray, detections: List[Detection],
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:
        """
        Draw detections on image.

        Args:
            color_img: Color image (H, W, 3) BGR
            detections: List of Detection objects
            color: BGR color for bounding boxes
            thickness: Line thickness

        Returns:
            Image with drawn detections
        """
        img = color_img.copy()

        for det in detections:
            # Draw bounding box
            cv2.rectangle(img, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

            # Draw label and confidence
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(img, label_text, (det.x1, det.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw center point
            center = det.center
            cv2.circle(img, center, 5, color, -1)

        return img


def get_default_model_path() -> Optional[str]:
    """Get default model path if it exists"""
    import os

    # Try to find best.pt in step0_data_loading folder
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "best.pt")

    if os.path.exists(model_path):
        return model_path

    # Fallback to old location (step1)
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path_old = os.path.join(base_path, "step1_calibration_process", "target_detection",
                                  "automatic_mode", "best.pt")

    if os.path.exists(model_path_old):
        return model_path_old

    return None

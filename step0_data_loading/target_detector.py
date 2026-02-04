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
               cx: float = 320.0, cy: float = 240.0,
               max_bbox_area_ratio: float = 0.15) -> List[Detection]:
        """
        Detect targets in color image and compute 3D positions.

        Args:
            color_img: Color image (H, W, 3) BGR
            depth_img: Depth image (H, W) in meters (optional)
            target_label: Label to filter detections (default: "cup")
            fx, fy: Focal lengths in pixels
            cx, cy: Principal point
            max_bbox_area_ratio: Maximum bbox area as ratio of image area (default: 0.15)
                               Targets larger than this are filtered out

        Returns:
            List of Detection objects with 3D positions if depth provided
        """
        if self.model is None:
            return []

        # Run detection
        results = self.model(color_img, verbose=False)

        # Get image dimensions for size filtering
        img_height, img_width = color_img.shape[:2]
        img_area = img_height * img_width

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

                    # Filter out targets that are too large
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    bbox_area_ratio = bbox_area / img_area

                    if bbox_area_ratio > max_bbox_area_ratio:
                        print(f"⚠️ Filtered out large target: {bbox_area_ratio:.2%} of image (max: {max_bbox_area_ratio:.2%})")
                        continue

                    det = Detection(x1, y1, x2, y2, conf, label)

                    # Compute 3D position if depth available
                    if depth_img is not None:
                        center_x, center_y = det.center

                        # Use CENTER REGION of bbox (inner 50%) to avoid background pixels
                        # This gives more accurate depth for the target itself
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        margin_x = int(bbox_w * 0.25)  # 25% margin on each side = inner 50%
                        margin_y = int(bbox_h * 0.25)
                        inner_x1 = max(x1 + margin_x, 0)
                        inner_x2 = min(x2 - margin_x, depth_img.shape[1])
                        inner_y1 = max(y1 + margin_y, 0)
                        inner_y2 = min(y2 - margin_y, depth_img.shape[0])

                        # Compute depth from inner region (excludes background at bbox edges)
                        inner_depth = depth_img[inner_y1:inner_y2, inner_x1:inner_x2]
                        valid_inner = inner_depth[(inner_depth > 0.1) & (inner_depth < 10.0)]

                        # Also compute full bbox depth for comparison/fallback
                        bbox_depth = depth_img[y1:y2, x1:x2]
                        valid_full = bbox_depth[(bbox_depth > 0.1) & (bbox_depth < 10.0)]

                        if len(valid_inner) > 10:
                            # Prefer inner region median (more robust against background)
                            det.avg_depth = float(np.mean(valid_inner))
                            det.median_depth = float(np.median(valid_inner))
                            depth_value = det.median_depth  # Use median for robustness
                            det.depth = depth_value
                        elif len(valid_full) > 0:
                            # Fallback to full bbox
                            det.avg_depth = float(np.mean(valid_full))
                            det.median_depth = float(np.median(valid_full))
                            depth_value = det.median_depth
                            det.depth = depth_value
                        else:
                            # Last resort: center pixel
                            if 0 <= center_y < depth_img.shape[0] and 0 <= center_x < depth_img.shape[1]:
                                depth_value = depth_img[center_y, center_x]
                                if depth_value > 0.1:
                                    det.depth = depth_value
                                    det.avg_depth = depth_value
                                    det.median_depth = depth_value

                        # Convert to 3D camera frame coordinates
                        if det.depth is not None and det.depth > 0:
                            z = det.depth
                            x = (center_x - cx) * z / fx
                            y = (center_y - cy) * z / fy
                            det.center_3d = (x, y, z)

                    detections.append(det)

        # Filter: If more than 4 targets detected, keep only the 4 smallest
        # (Large bounding boxes are likely humans misidentified as targets)
        if len(detections) > 4:
            print(f"⚠️ Detected {len(detections)} targets, filtering to keep only 4 smallest")

            # Sort by bounding box area (smallest first)
            detections_with_area = [(det, (det.x2 - det.x1) * (det.y2 - det.y1)) for det in detections]
            detections_with_area.sort(key=lambda x: x[1])  # Sort by area

            # Keep only the 4 smallest
            detections = [det for det, area in detections_with_area[:4]]

            print(f"✅ Kept 4 smallest targets (filtered out {len(detections_with_area) - 4} large ones)")

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


def apply_known_depth_constraint(detections: List[Detection],
                                  expected_depth: float,
                                  fx: float, fy: float,
                                  cx: float, cy: float,
                                  tolerance: float = 0.5) -> List[Detection]:
    """
    Apply a known depth constraint to target detections.

    If you know from your experiment setup that targets are at a specific depth,
    this function will:
    1. Warn about targets with detected depth far from expected
    2. Optionally override depth with expected value
    3. Recompute 3D positions with constrained depth

    Args:
        detections: List of Detection objects
        expected_depth: Known target depth in meters (e.g., 4.0m)
        fx, fy, cx, cy: Camera intrinsics
        tolerance: How far detected depth can be from expected before warning (meters)

    Returns:
        Updated detections with constrained depth
    """
    constrained = []

    for det in detections:
        det_copy = Detection(
            det.x1, det.y1, det.x2, det.y2,
            det.confidence, det.label,
            det.center_3d, det.depth, det.avg_depth, det.median_depth
        )

        original_depth = det.depth
        depth_diff = abs(original_depth - expected_depth) if original_depth else float('inf')

        if depth_diff > tolerance:
            print(f"⚠️ Target {det.label}: detected depth {original_depth:.2f}m "
                  f"differs from expected {expected_depth:.2f}m by {depth_diff:.2f}m")
            # Override with expected depth
            det_copy.depth = expected_depth

        # Recompute 3D position with (possibly constrained) depth
        center_x, center_y = det_copy.center
        z = det_copy.depth if det_copy.depth else expected_depth
        x = (center_x - cx) * z / fx
        y = (center_y - cy) * z / fy
        det_copy.center_3d = (x, y, z)

        constrained.append(det_copy)

    return constrained


def enforce_coplanarity(detections: List[Detection],
                        use_median_y: bool = True) -> List[Detection]:
    """
    Enforce that all targets lie on the same plane (same Y coordinate).

    This is useful when targets are placed on a table/floor and should
    have approximately the same height.

    Args:
        detections: List of Detection objects with center_3d
        use_median_y: If True, use median Y; otherwise use mean

    Returns:
        Updated detections with enforced coplanar Y
    """
    # Get all Y values
    y_values = [d.center_3d[1] for d in detections if d.center_3d]

    if not y_values:
        return detections

    if use_median_y:
        common_y = float(np.median(y_values))
    else:
        common_y = float(np.mean(y_values))

    print(f"📐 Enforcing coplanarity: all targets at Y={common_y:.3f}m")

    constrained = []
    for det in detections:
        det_copy = Detection(
            det.x1, det.y1, det.x2, det.y2,
            det.confidence, det.label,
            det.center_3d, det.depth, det.avg_depth, det.median_depth
        )

        if det_copy.center_3d:
            x, _, z = det_copy.center_3d
            det_copy.center_3d = (x, common_y, z)

        constrained.append(det_copy)

    return constrained


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

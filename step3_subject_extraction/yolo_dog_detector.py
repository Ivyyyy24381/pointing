"""
YOLO-based dog detector for bounding box detection.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


class YOLODogDetector:
    """Detect dogs using YOLO and return bounding boxes."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model (if None, uses ultralytics default)
        """
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO

            # Load YOLOv8 model (or use provided path)
            if model_path:
                self.model = YOLO(model_path)
            else:
                # Use default YOLOv8n (nano, fastest)
                self.model = YOLO('yolov8n.pt')

            print("✅ YOLO model loaded")
        except ImportError:
            raise ImportError(
                "Ultralytics YOLO not installed. Install with:\n"
                "pip install ultralytics"
            )

    def detect_dog(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect dog in image and return bounding box.

        Args:
            image: Input image (BGR format from cv2.imread)
            conf_threshold: Confidence threshold for detection

        Returns:
            (x1, y1, x2, y2) bounding box or None if no dog detected
            Coordinates are in original image space
        """
        # Run YOLO detection
        results = self.model(image, verbose=False)

        # Filter for 'dog' class (class 16 in COCO dataset)
        DOG_CLASS = 16

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == DOG_CLASS and conf >= conf_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    return (int(x1), int(y1), int(x2), int(y2))

        return None

    def crop_to_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                     padding: int = 20) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Crop image to bounding box with padding.

        Args:
            image: Input image
            bbox: (x1, y1, x2, y2) bounding box
            padding: Pixels to pad around bbox

        Returns:
            (cropped_image, (x_offset, y_offset))
            Offsets for mapping coordinates back to original
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        # Crop
        cropped = image[y1:y2, x1:x2]

        return cropped, (x1, y1)


if __name__ == "__main__":
    # Test YOLO detector
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--output", default="test_yolo_output.jpg", help="Output visualization")

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Failed to load image: {args.image}")
        exit(1)

    # Detect dog
    detector = YOLODogDetector()
    bbox = detector.detect_dog(image)

    if bbox:
        x1, y1, x2, y2 = bbox
        print(f"✅ Dog detected: bbox=({x1}, {y1}, {x2}, {y2})")

        # Draw bounding box
        vis = image.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "Dog", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                   0.9, (0, 255, 0), 2)

        # Save
        cv2.imwrite(args.output, vis)
        print(f"✅ Saved visualization: {args.output}")

        # Test cropping
        cropped, offset = detector.crop_to_bbox(image, bbox, padding=20)
        crop_path = args.output.replace('.jpg', '_cropped.jpg')
        cv2.imwrite(crop_path, cropped)
        print(f"✅ Saved cropped: {crop_path}")
        print(f"   Offset: {offset}")
    else:
        print(f"❌ No dog detected in image")

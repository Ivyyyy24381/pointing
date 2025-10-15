"""
SAM2 (Segment Anything Model 2) video segmentation wrapper.

Segments subjects across video frames with minimal user interaction.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

try:
    import torch
    from PIL import Image
    # Add SAM2 to path if needed
    sam2_path = Path(__file__).parent.parent / "thirdparty" / "sam2"
    if sam2_path.exists():
        sys.path.insert(0, str(sam2_path))

    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️ SAM2 not available. Install from: https://github.com/facebookresearch/segment-anything-2")


@dataclass
class SegmentationResult:
    """Segmentation result for a single frame."""
    frame_number: int
    mask: np.ndarray  # (H, W) boolean mask
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float


class SAM2VideoSegmenter:
    """
    SAM2-based video segmentation.

    Segments a subject across video frames using SAM2's video predictor.
    """

    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: Optional[str] = None):
        """
        Initialize SAM2 video segmenter.

        Args:
            checkpoint_path: Path to SAM2 checkpoint (.pt file)
            model_cfg: Model configuration file
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 not available. Install from: https://github.com/facebookresearch/segment-anything-2")

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent.parent / "legacy_code" / "sam2_b.pt")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

        print(f"🔧 Initializing SAM2 on device: {self.device}")
        print(f"📦 Loading checkpoint: {checkpoint_path}")

        # Build predictor
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=self.device)
        self.inference_state = None

        print("✅ SAM2 initialized successfully")

    def segment_video_folder(self,
                            video_dir: str,
                            init_points: List[Tuple[int, int]],
                            init_labels: List[int],
                            init_frame_idx: int = 0) -> Tuple[List[SegmentationResult], Dict]:
        """
        Segment subject across all frames in a video directory.

        Args:
            video_dir: Directory containing video frames (images)
            init_points: Initial prompt points [(x, y), ...]
            init_labels: Labels for points (1=foreground, 0=background)
            init_frame_idx: Frame index to initialize segmentation

        Returns:
            (results, metadata): List of SegmentationResult and metadata dict
        """
        video_dir = Path(video_dir)

        if not video_dir.exists():
            raise ValueError(f"Video directory not found: {video_dir}")

        print(f"\n🎬 Segmenting video: {video_dir}")

        # Initialize inference state
        print("🔄 Initializing inference state...")
        self.inference_state = self.predictor.init_state(video_path=str(video_dir))

        # Convert points to numpy
        points = np.array(init_points, dtype=np.float32)
        labels = np.array(init_labels, dtype=np.int32)

        # Add initial points
        print(f"🎯 Adding {len(init_points)} prompt points on frame {init_frame_idx}")
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=init_frame_idx,
            obj_id=1,
            points=points,
            labels=labels
        )

        # Propagate masks through video
        print("🔄 Propagating segmentation masks...")
        results = []

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            # Get mask for frame
            mask = (out_mask_logits[0] > 0.0).cpu().numpy()[0]  # (H, W)

            # Compute bounding box
            bbox = self._mask_to_bbox(mask)

            # Confidence (use mean logit value)
            confidence = float(torch.sigmoid(out_mask_logits[0]).mean().cpu().numpy())

            result = SegmentationResult(
                frame_number=out_frame_idx,
                mask=mask,
                bbox=bbox,
                confidence=confidence
            )
            results.append(result)

            if (out_frame_idx + 1) % 10 == 0:
                print(f"  Processed {out_frame_idx + 1} frames...", end='\r')

        print(f"\n✅ Segmented {len(results)} frames")

        # Create metadata
        metadata = {
            "num_frames": len(results),
            "init_frame_idx": init_frame_idx,
            "init_points": init_points,
            "init_labels": init_labels,
            "device": str(self.device),
            "model": "SAM2"
        }

        return results, metadata

    def _mask_to_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert binary mask to bounding box."""
        if not mask.any():
            return (0, 0, 0, 0)

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def save_masks(self, results: List[SegmentationResult], output_dir: str) -> None:
        """Save masks as PNG files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for result in results:
            mask_file = output_dir / f"mask_{result.frame_number:06d}.png"
            mask_img = (result.mask * 255).astype(np.uint8)
            cv2.imwrite(str(mask_file), mask_img)

        print(f"💾 Saved {len(results)} masks to: {output_dir}")

    def save_metadata(self, metadata: Dict, output_file: str) -> None:
        """Save segmentation metadata to JSON."""
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Saved metadata to: {output_file}")

    def create_masked_video(self,
                           video_dir: str,
                           results: List[SegmentationResult],
                           output_video: str,
                           fps: int = 30) -> None:
        """
        Create video with segmentation masks overlaid.

        Args:
            video_dir: Directory with original frames
            results: Segmentation results
            output_video: Output video path
            fps: Frames per second
        """
        video_dir = Path(video_dir)

        # Get first frame to determine size
        image_files = sorted(list(video_dir.glob("*.png")) + list(video_dir.glob("*.jpg")))
        if not image_files:
            raise ValueError(f"No images found in {video_dir}")

        first_frame = cv2.imread(str(image_files[0]))
        h, w = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        print(f"🎥 Creating masked video: {output_video}")

        for i, (img_path, result) in enumerate(zip(image_files, results)):
            frame = cv2.imread(str(img_path))

            # Apply mask overlay (semi-transparent)
            mask_overlay = np.zeros_like(frame)
            mask_overlay[result.mask] = [0, 255, 0]  # Green overlay

            masked_frame = cv2.addWeighted(frame, 0.7, mask_overlay, 0.3, 0)

            # Draw bounding box
            x, y, w_box, h_box = result.bbox
            cv2.rectangle(masked_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            out.write(masked_frame)

            if (i + 1) % 30 == 0:
                print(f"  Processed {i + 1}/{len(results)} frames...", end='\r')

        out.release()
        print(f"\n✅ Saved masked video to: {output_video}")

"""
SAM3-based subject detection for dog/baby/human segmentation.

Uses SAM3 text prompts to segment subjects in images, providing:
- Bounding box and mask for the subject
- Center of mass (2D and 3D) for localization
- Video mode for efficient tracking across frames
"""

import numpy as np
import tempfile
import cv2
from typing import Optional, Dict, List, Tuple
from pathlib import Path

try:
    import torch
    from PIL import Image as PILImage
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False

from .subject_detector import SubjectDetectionResult


class SAM3Detector:
    """SAM3-based detector for dog, baby, or human segmentation.

    Supports both per-frame image mode and efficient video tracking mode.
    """

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3 not available. Install from thirdparty/sam3/")

        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.video_predictor = None

    def _ensure_loaded(self):
        """Lazy-load the SAM3 image model."""
        if self.processor is not None:
            return
        print("Loading SAM3 model (this may take a moment)...")
        self.model = build_sam3_image_model(
            device=self.device,
            eval_mode=True,
            load_from_HF=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        self.processor = Sam3Processor(
            self.model,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )
        print(f"SAM3 image model loaded on {self.device}")

    def _ensure_video_loaded(self):
        """Lazy-load the SAM3 video predictor using the official builder."""
        if self.video_predictor is not None:
            return
        print("Loading SAM3 video model...")
        from sam3.model_builder import build_sam3_video_predictor
        gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        self.video_predictor = build_sam3_video_predictor(gpus_to_use=gpus)
        print("SAM3 video model loaded")

    def _to_pil(self, image: np.ndarray):
        """Convert RGB numpy array to PIL Image."""
        return PILImage.fromarray(image)

    @torch.inference_mode()
    def detect_subject_in_frame(
        self,
        image: np.ndarray,
        text_prompt: str = "dog",
        crop_lower: bool = True,
        crop_ratio: float = 0.6,
        depth_image: Optional[np.ndarray] = None,
        fx: float = 615.0, fy: float = 615.0,
        cx: float = 320.0, cy: float = 240.0,
    ) -> Optional[SubjectDetectionResult]:
        """Detect a subject in a single frame using SAM3 text prompt."""
        self._ensure_loaded()

        h_orig, w_orig = image.shape[:2]
        y_offset = 0

        if crop_lower:
            y_offset = int(h_orig * (1 - crop_ratio))
            work_img = image[y_offset:, :, :]
        else:
            work_img = image

        state = self.processor.set_image(self._to_pil(work_img))
        output = self.processor.set_text_prompt(state=state, prompt=text_prompt)

        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or len(masks) == 0:
            return None

        best_idx = scores.argmax().item()
        mask = masks[best_idx].cpu().numpy().squeeze()
        box = boxes[best_idx].cpu().numpy()
        score = scores[best_idx].item()

        if score < self.confidence_threshold:
            return None

        return self._mask_to_result(
            mask, box, score, y_offset, text_prompt,
            crop_lower, depth_image, w_orig, h_orig, fx, fy, cx, cy,
        )

    def _mask_to_result(
        self, mask, box, score, y_offset, text_prompt,
        crop_lower, depth_image, w_orig, h_orig, fx, fy, cx, cy,
    ) -> Optional[SubjectDetectionResult]:
        """Convert a mask + box + score to SubjectDetectionResult."""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None

        center_x = float(xs.mean())
        center_y = float(ys.mean()) + y_offset

        x1, y1, x2, y2 = box.tolist() if hasattr(box, 'tolist') else box
        bbox = (int(x1), int(y1 + y_offset), int(x2), int(y2 + y_offset))

        keypoints_2d = [[center_x, center_y, score]]

        keypoints_3d = None
        if depth_image is not None:
            px, py = int(center_x), int(center_y)
            if 0 <= px < w_orig and 0 <= py < h_orig:
                z = depth_image[py, px]
                if z > 0 and not np.isnan(z) and not np.isinf(z):
                    x_3d = (px - cx) * z / fx
                    y_3d = (py - cy) * z / fy
                    keypoints_3d = [[float(x_3d), float(y_3d), float(z)]]

        subject_type = "dog" if "dog" in text_prompt.lower() else (
            "baby" if "baby" in text_prompt.lower() else text_prompt
        )

        return SubjectDetectionResult(
            subject_type=subject_type,
            detection_region="lower_half" if crop_lower else "full",
            bbox=bbox,
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d,
        )

    def process_batch_video(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        text_prompt: str = "dog",
        crop_lower: bool = True,
        crop_ratio: float = 0.6,
        depth_images: Optional[List[np.ndarray]] = None,
        fx: float = 615.0, fy: float = 615.0,
        cx: float = 320.0, cy: float = 240.0,
    ) -> Dict[str, SubjectDetectionResult]:
        """
        Process frames using SAM3 video mode (track + propagate).

        Writes cropped frames as JPEGs to a temp directory, runs SAM3 video
        predictor with a text prompt on frame 0, then propagates tracking to
        all frames. Much faster than per-frame image inference.
        """
        self._ensure_video_loaded()

        if not frames:
            return {}

        h_orig = frames[0].shape[0]
        w_orig = frames[0].shape[1]
        y_offset = int(h_orig * (1 - crop_ratio)) if crop_lower else 0

        results = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            jpeg_dir = Path(tmpdir) / "frames"
            jpeg_dir.mkdir()

            # Write cropped frames as numbered JPEGs
            for i, frame in enumerate(frames):
                if crop_lower:
                    cropped = frame[y_offset:]
                else:
                    cropped = frame
                # SAM3 video loader expects BGR JPEGs or RGB -- save as JPEG
                bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(jpeg_dir / f"{i:06d}.jpg"), bgr)

            # Start video session
            resp = self.video_predictor.handle_request({
                "type": "start_session",
                "resource_path": str(jpeg_dir),
            })
            session_id = resp["session_id"]

            # Add text prompt on first frame
            self.video_predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": text_prompt,
            })

            # Propagate through video using handle_stream_request (official API)
            raw_outputs = {}
            for response in self.video_predictor.handle_stream_request(
                request=dict(
                    type="propagate_in_video",
                    session_id=session_id,
                )
            ):
                raw_outputs[response["frame_index"]] = response["outputs"]

            # Use SAM3's prepare_masks_for_visualization to get clean masks
            # formatted[fidx] = {obj_id: mask_array(H, W), ...}
            from sam3.visualization_utils import prepare_masks_for_visualization
            formatted = prepare_masks_for_visualization(raw_outputs)

            frame_masks = {}
            for fidx, fdata in formatted.items():
                if isinstance(fdata, dict):
                    # Merge all object masks into one binary mask
                    combined = None
                    for obj_id, mask in fdata.items():
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        mask = mask.squeeze()
                        if mask.ndim == 2:
                            if combined is None:
                                combined = mask.astype(bool)
                            else:
                                combined = combined | mask.astype(bool)
                    if combined is not None and combined.sum() > 0:
                        frame_masks[fidx] = combined.astype(np.uint8)

            # Close session
            try:
                self.video_predictor.handle_request({
                    "type": "close_session",
                    "session_id": session_id,
                })
            except Exception:
                pass

            # Convert masks to SubjectDetectionResults
            for fidx, mask in frame_masks.items():
                if fidx >= len(frame_numbers):
                    continue
                frame_num = frame_numbers[fidx]
                frame_key = f"frame_{frame_num:06d}"

                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    continue

                center_x = float(xs.mean())
                center_y = float(ys.mean()) + y_offset
                # Compute bbox from mask
                x1, y1_c, x2, y2_c = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                bbox = (x1, y1_c + y_offset, x2, y2_c + y_offset)

                keypoints_2d = [[center_x, center_y, 1.0]]

                keypoints_3d = None
                depth = depth_images[fidx] if depth_images and fidx < len(depth_images) else None
                if depth is not None:
                    px, py = int(center_x), int(center_y)
                    if 0 <= px < w_orig and 0 <= py < h_orig:
                        z = depth[py, px]
                        if z > 0 and not np.isnan(z) and not np.isinf(z):
                            x_3d = (px - cx) * z / fx
                            y_3d = (py - cy) * z / fy
                            keypoints_3d = [[float(x_3d), float(y_3d), float(z)]]

                subject_type = "dog" if "dog" in text_prompt.lower() else (
                    "baby" if "baby" in text_prompt.lower() else text_prompt
                )

                results[frame_key] = SubjectDetectionResult(
                    subject_type=subject_type,
                    detection_region="lower_half" if crop_lower else "full",
                    bbox=bbox,
                    keypoints_2d=keypoints_2d,
                    keypoints_3d=keypoints_3d,
                )

        print(f"    SAM3 video detected {text_prompt} in {len(results)}/{len(frames)} frames")
        return results

    def process_batch(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        text_prompt: str = "dog",
        crop_lower: bool = True,
        crop_ratio: float = 0.6,
        depth_images: Optional[List[np.ndarray]] = None,
        fx: float = 615.0, fy: float = 615.0,
        cx: float = 320.0, cy: float = 240.0,
        use_video_mode: bool = False,
    ) -> Dict[str, SubjectDetectionResult]:
        """
        Process frames for subject detection.

        Uses per-frame image mode by default (stable, ~25/25 detection).
        Video tracking mode (use_video_mode=True) is faster but currently
        crashes with a C-level double-free in SAM3 on longer sequences.
        """
        if use_video_mode:
            try:
                return self.process_batch_video(
                    frames, frame_numbers, text_prompt,
                    crop_lower, crop_ratio, depth_images,
                    fx, fy, cx, cy,
                )
            except Exception as e:
                print(f"    Video mode failed ({e}), falling back to per-frame...")

        return self._process_batch_per_frame(
            frames, frame_numbers, text_prompt,
            crop_lower, crop_ratio, depth_images,
            fx, fy, cx, cy,
        )

    def _process_batch_per_frame(
        self,
        frames: List[np.ndarray],
        frame_numbers: List[int],
        text_prompt: str = "dog",
        crop_lower: bool = True,
        crop_ratio: float = 0.6,
        depth_images: Optional[List[np.ndarray]] = None,
        fx: float = 615.0, fy: float = 615.0,
        cx: float = 320.0, cy: float = 240.0,
    ) -> Dict[str, SubjectDetectionResult]:
        """Per-frame fallback (slower but more robust)."""
        self._ensure_loaded()
        results = {}

        for i, (frame, frame_num) in enumerate(zip(frames, frame_numbers)):
            depth = depth_images[i] if depth_images and i < len(depth_images) else None
            result = self.detect_subject_in_frame(
                frame, text_prompt=text_prompt,
                crop_lower=crop_lower, crop_ratio=crop_ratio,
                depth_image=depth, fx=fx, fy=fy, cx=cx, cy=cy,
            )
            if result is not None:
                frame_key = f"frame_{frame_num:06d}"
                results[frame_key] = result

            if (i + 1) % 50 == 0 or i == len(frames) - 1:
                print(f"    SAM3 {text_prompt}: {i+1}/{len(frames)} frames...", end='\r')

        print(f"    SAM3 detected {text_prompt} in {len(results)}/{len(frames)} frames")
        return results

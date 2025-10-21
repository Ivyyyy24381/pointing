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

    def __init__(self, subject_type: str = 'dog', crop_ratio: float = 0.6):
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

        # For batch video processing (dog)
        self.cropped_frames_cache = []  # Store cropped frames
        self.frame_numbers_cache = []   # Store original frame numbers
        self.y_offsets_cache = []       # Store y_offsets for mapping back

        # Initialize detector based on type
        if subject_type == 'dog':
            self._init_dog_detector()
        else:  # baby
            self._init_baby_detector()

    def _init_dog_detector(self):
        """Initialize DeepLabCut for dog skeleton detection"""
        try:
            # Try to import DeepLabCut
            import deeplabcut
            self.dlc_predict = deeplabcut
            self.pose = None  # Not using MediaPipe for dog

            # SuperAnimal Quadruped keypoint names (39 keypoints)
            self.keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'throat', 'withers', 'tail_base', 'left_front_paw', 'right_front_paw',
                'left_back_paw', 'right_back_paw', 'left_front_knee', 'right_front_knee',
                'left_back_knee', 'right_back_knee', 'left_front_elbow', 'right_front_elbow',
                'left_back_elbow', 'right_back_elbow', 'neck', 'tail_mid', 'tail_tip',
                'left_front_wrist', 'right_front_wrist', 'left_back_ankle', 'right_back_ankle',
                'left_front_pad', 'right_front_pad', 'left_back_pad', 'right_back_pad',
                'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                'left_front_toe', 'right_front_toe', 'left_back_toe', 'right_back_toe'
            ]

            # DeepLabCut SuperAnimal Quadruped keypoint mapping to MediaPipe-like indices
            # Map DLC keypoints to 33 MediaPipe indices for compatibility
            self.dlc_to_mediapipe_map = {
                'nose': 0,
                'left_eye': 2,
                'right_eye': 5,
                'left_ear': 7,
                'right_ear': 8,
                'left_shoulder': 11,
                'right_shoulder': 12,
                'left_hip': 23,
                'right_hip': 24,
                'left_front_knee': 13,
                'right_front_knee': 14,
                'left_back_knee': 25,
                'right_back_knee': 26,
                'left_front_paw': 15,
                'right_front_paw': 16,
                'left_back_paw': 27,
                'right_back_paw': 28,
            }

            print(f"✅ DeepLabCut loaded for dog skeleton detection (SuperAnimal Quadruped)")
        except ImportError:
            print("⚠️ DeepLabCut not installed. Run: pip install deeplabcut")
            print("   For dog detection, DeepLabCut with SuperAnimal models is required")
            self.dlc_predict = None
            self.pose = None
        except Exception as e:
            print(f"⚠️ Failed to load DeepLabCut: {e}")
            self.dlc_predict = None
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
        """Detect dog skeleton using DeepLabCut SuperAnimal Quadruped

        Note: For single-frame dog detection, we use video_inference_superanimal
        which requires creating a temporary video file.
        """
        if self.dlc_predict is None:
            return None

        try:
            import cv2
            import tempfile
            import json
            from pathlib import Path

            # Create temporary directory for processing
            # Create debug directory for saving outputs
            debug_dir = Path("/tmp/dlc_debug")
            debug_dir.mkdir(exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                # Save cropped image as temporary PNG
                tmp_img = Path(tmpdir) / "frame.png"
                cv2.imwrite(str(tmp_img), cropped_img)

                # Create multi-frame video (DeepLabCut needs at least 2-3 frames)
                # We duplicate the same frame to create a valid video
                h, w = cropped_img.shape[:2]
                tmp_video = Path(tmpdir) / "temp_video.avi"

                # Use AVI with MJPEG codec (more reliable for short videos)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(str(tmp_video), fourcc, 30.0, (w, h))

                if not writer.isOpened():
                    print(f"⚠️ Failed to create video writer")
                    return None

                # Ensure image is in BGR format (OpenCV default)
                if len(cropped_img.shape) == 2:
                    # Grayscale - convert to BGR
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                elif cropped_img.shape[2] == 4:
                    # RGBA - convert to BGR
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGBA2BGR)

                # Write the same frame 5 times (ensure valid video)
                for i in range(5):
                    writer.write(cropped_img)

                writer.release()

                # Verify video was created and has frames
                import os
                if not os.path.exists(tmp_video) or os.path.getsize(tmp_video) == 0:
                    print(f"⚠️ Video file not created or empty")
                    return None

                # Test read the video
                test_cap = cv2.VideoCapture(str(tmp_video))
                frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Try to actually read frames
                read_frames = 0
                test_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ret, _ = test_cap.read()
                    if not ret:
                        break
                    read_frames += 1

                test_cap.release()

                print(f"   Video verification:")
                print(f"     - Reported frame count: {frame_count}")
                print(f"     - Actually readable frames: {read_frames}")

                if read_frames < 1:
                    print(f"⚠️ Video has {read_frames} readable frames - cannot process")
                    print(f"   This might be a codec issue. Trying to re-encode...")

                    # Try to re-encode the video using FFmpeg if available
                    # For now, just return None
                    return None

                # Run DeepLabCut SuperAnimal detection
                # API: video_inference_superanimal(videos, superanimal_name, **kwargs)
                import deeplabcut

                print(f"🔍 Running DeepLabCut on video: {tmp_video}")
                print(f"   Video info: {frame_count} frames, {w}x{h}")

                try:
                    output = deeplabcut.video_inference_superanimal(
                        [str(tmp_video)],
                        'superanimal_quadruped',  # superanimal_name (not model_type!)
                        model_name="hrnet_w32",
                        detector_name="fasterrcnn_resnet50_fpn_v2",
                        video_adapt=False,
                        pcutoff=0.3  # Confidence threshold
                    )
                    print(f"✅ DeepLabCut completed successfully")
                    print(f"   Output: {output}")
                    print(f"   Output type: {type(output)}")
                except Exception as dlc_error:
                    print(f"❌ DeepLabCut inference failed: {dlc_error}")
                    import traceback
                    traceback.print_exc()

                    # Debug: List all files in temp directory
                    print(f"\n📂 Files in temp directory:")
                    import os
                    for file in sorted(os.listdir(tmpdir)):
                        filepath = Path(tmpdir) / file
                        size = os.path.getsize(filepath)
                        print(f"   - {file} ({size} bytes)")

                    # If no dog detected, the detector returns empty predictions
                    # This is not necessarily an error - just means no dog in frame
                    print("\nℹ️ Likely cause: No dog detected in the cropped region (lower half)")
                    return None

                # DeepLabCut outputs pickle (.pkl), JSON, or HDF5 (.h5) files
                # Look for output files
                import pickle
                pkl_files = list(Path(tmpdir).glob("*.pkl"))
                json_files = list(Path(tmpdir).glob("*.json"))
                h5_files = list(Path(tmpdir).glob("*.h5"))
                # SuperAnimal output doesn't have 'DLC' in filename, look for any JSON
                dlc_jsons = json_files  # Use all JSON files

                print(f"\n📂 DeepLabCut output files:")
                print(f"   .pkl files: {[f.name for f in pkl_files]}")
                print(f"   .json files: {[f.name for f in json_files]}")
                print(f"   .h5 files: {[f.name for f in h5_files]}")
                print(f"   DLC JSON files: {[f.name for f in dlc_jsons]}")

                # Copy output files to debug directory for inspection
                import shutil
                for file in pkl_files + json_files + h5_files:
                    debug_file = debug_dir / file.name
                    shutil.copy2(file, debug_file)
                    print(f"   📋 Saved to: {debug_file}")

                # Copy video files too (labeled videos from DLC)
                for video_file in Path(tmpdir).glob("*.mp4"):
                    debug_video = debug_dir / video_file.name
                    shutil.copy2(video_file, debug_video)
                    print(f"   🎥 Saved labeled video: {debug_video}")

                # Copy the input video for reference
                debug_input = debug_dir / "input_video.avi"
                shutil.copy2(tmp_video, debug_input)
                print(f"   📹 Saved input video: {debug_input}")

                if pkl_files:
                    # Load pickle file (standard DLC output format)
                    with open(pkl_files[0], 'rb') as f:
                        dlc_predictions = pickle.load(f)

                    # DLC pickle format: list of dicts with 'bodyparts', 'bboxes', etc.
                    if not dlc_predictions or len(dlc_predictions) == 0:
                        print("⚠️ No predictions in DLC output")
                        return None

                    # Get first frame (we duplicated it 3 times, so all are identical)
                    frame_pred = dlc_predictions[0]
                    bodyparts_list = frame_pred.get('bodyparts', [])

                    if not bodyparts_list or len(bodyparts_list) == 0:
                        print("⚠️ No bodyparts detected in frame")
                        return None

                    # Take first individual
                    keypoints = bodyparts_list[0]  # [[x, y, conf], ...]

                    # Convert to dictionary format for compatibility
                    dlc_keypoints = {}
                    for i, kp in enumerate(keypoints):
                        if i < len(self.keypoint_names):
                            bodypart_name = self.keypoint_names[i]
                            dlc_keypoints[bodypart_name] = kp

                elif dlc_jsons:
                    # Parse JSON output (SuperAnimal format)
                    json_path = dlc_jsons[0]
                    print(f"   📄 Parsing JSON: {json_path.name}")

                    with open(json_path, 'r') as f:
                        dlc_data = json.load(f)

                    # SuperAnimal JSON format: list of frames, each with bodyparts list
                    if isinstance(dlc_data, list) and len(dlc_data) > 0:
                        frame_data = dlc_data[0]  # First frame
                    else:
                        print(f"   ⚠️ Unexpected JSON format")
                        return None

                    bodyparts_list = frame_data.get('bodyparts', [])
                    if not bodyparts_list or len(bodyparts_list) == 0:
                        print(f"   ⚠️ No bodyparts in JSON")
                        return None

                    # Each frame has bodyparts for multiple individuals
                    # bodyparts_list[0] = first individual's keypoints [[x, y, conf], ...]
                    keypoints = bodyparts_list[0]
                    print(f"   Found {len(keypoints)} keypoints for first individual")

                    # Convert to dictionary format for compatibility
                    dlc_keypoints = {}
                    for i, kp in enumerate(keypoints):
                        if i < len(self.keypoint_names):
                            bodypart_name = self.keypoint_names[i]
                            dlc_keypoints[bodypart_name] = kp
                else:
                    print("⚠️ No DLC output files found (.pkl or .json)")
                    return None

                # Convert DLC keypoints to MediaPipe format
                keypoints_2d_cropped = self._dlc_to_mediapipe_keypoints(dlc_keypoints, cropped_img.shape)

                if keypoints_2d_cropped is None or len(keypoints_2d_cropped) == 0:
                    return None

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

        except Exception as e:
            print(f"⚠️ DeepLabCut dog detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def _dlc_to_mediapipe_keypoints(self, dlc_keypoints: Dict, image_shape: Tuple) -> Optional[List[List[float]]]:
        """
        Convert DeepLabCut SuperAnimal Quadruped keypoints (39) to MediaPipe format (33).

        Args:
            dlc_keypoints: Dictionary with bodypart names as keys, [x, y, confidence] as values
            image_shape: (height, width, channels) of the image

        Returns:
            List of 33 keypoints in MediaPipe format [[x, y, confidence], ...]
        """
        # Initialize 33 MediaPipe keypoints with zeros
        mediapipe_keypoints = [[0.0, 0.0, 0.0] for _ in range(33)]

        # Map DLC bodypart names to MediaPipe indices (from _init_dog_detector)
        for bodypart_name, mp_idx in self.dlc_to_mediapipe_map.items():
            if bodypart_name in dlc_keypoints:
                kp = dlc_keypoints[bodypart_name]
                if len(kp) >= 3:  # [x, y, confidence]
                    mediapipe_keypoints[mp_idx] = [float(kp[0]), float(kp[1]), float(kp[2])]

        return mediapipe_keypoints

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

    def add_frame_to_batch(self, image: np.ndarray, frame_number: int):
        """
        Add frame to batch for later video processing (dog detection only).

        Args:
            image: RGB image (H, W, 3)
            frame_number: Frame number
        """
        if self.subject_type != 'dog':
            return

        # Crop to lower half and cache
        cropped_img, y_offset = self.crop_to_lower_half(image)
        self.cropped_frames_cache.append(cropped_img)
        self.frame_numbers_cache.append(frame_number)
        self.y_offsets_cache.append(y_offset)

    def process_batch_video(self, output_video_path: str,
                           frames: Optional[List[np.ndarray]] = None,
                           frame_numbers: Optional[List[int]] = None,
                           depth_images: Optional[List[np.ndarray]] = None,
                           fx: float = 615.0, fy: float = 615.0,
                           cx: float = 320.0, cy: float = 240.0) -> Dict[str, SubjectDetectionResult]:
        """
        Process frames as a video through DeepLabCut.

        Args:
            output_video_path: Path to save temporary cropped video
            frames: Optional list of frames to process (if not using cached frames)
            frame_numbers: Optional list of frame numbers corresponding to frames
            depth_images: List of depth images (same order as frames)
            fx, fy: Focal lengths
            cx, cy: Principal point

        Returns:
            Dictionary mapping frame_keys to SubjectDetectionResult
        """
        if self.subject_type != 'dog':
            return {}

        # Check if DeepLabCut is available
        if self.dlc_predict is None:
            print("❌ DeepLabCut not available. Cannot process batch video.")
            print("   Install with: pip install deeplabcut")
            return {}

        # Use provided frames or cached frames
        if frames is not None and frame_numbers is not None:
            # Crop all frames to lower half and cache
            self.cropped_frames_cache = []
            self.frame_numbers_cache = frame_numbers
            self.y_offsets_cache = []

            for frame in frames:
                cropped, y_offset = self.crop_to_lower_half(frame)
                self.cropped_frames_cache.append(cropped)
                self.y_offsets_cache.append(y_offset)

        if len(self.cropped_frames_cache) == 0:
            return {}

        import cv2
        from pathlib import Path

        try:
            # Create video from cropped frames
            print(f"\n🎬 Creating cropped video with {len(self.cropped_frames_cache)} frames...")
            h, w = self.cropped_frames_cache[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (w, h))

            for frame in self.cropped_frames_cache:
                # Convert RGB to BGR for OpenCV VideoWriter
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"✅ Saved cropped video: {output_video_path}")

            # Run DeepLabCut on the video
            print(f"\n🐶 Running DeepLabCut SuperAnimal Quadruped detection...")
            import deeplabcut
            deeplabcut.video_inference_superanimal(
                [output_video_path],
                'superanimal_quadruped',  # superanimal_name as positional arg
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",
                video_adapt=False,
                pcutoff=0.3
            )

            # Parse DeepLabCut output (SuperAnimal doesn't use 'DLC' in filename)
            output_dir = Path(output_video_path).parent
            video_name = Path(output_video_path).stem

            # Look for DLC output files matching the video name
            h5_files = list(output_dir.glob(f"{video_name}*.h5"))
            json_files = list(output_dir.glob(f"{video_name}*_before_adapt.json"))

            if not h5_files and not json_files:
                print("⚠️ No DeepLabCut output found (.h5 or .json)")
                print(f"   Looked for: {video_name}*.h5 or {video_name}*_before_adapt.json")
                return {}

            # Load DLC results - prefer JSON for easier parsing
            if json_files:
                import json
                json_file = json_files[0]
                print(f"📊 Loading results from: {json_file}")

                with open(json_file, 'r') as f:
                    dlc_data = json.load(f)

                # Process JSON format (list of frames)
                results = self._parse_dlc_json_batch(dlc_data, depth_images, fx, fy, cx, cy)
                return results

            # Fallback to H5 if JSON not available
            import pandas as pd
            h5_file = h5_files[0]
            print(f"📊 Loading results from: {h5_file}")
            df = pd.read_hdf(h5_file)

            # Convert DLC results to SubjectDetectionResult format
            results = {}
            for idx, (frame_num, y_offset) in enumerate(zip(self.frame_numbers_cache, self.y_offsets_cache)):
                frame_key = f"frame_{frame_num:06d}"

                # Extract keypoints for this frame
                dlc_keypoints = self._extract_dlc_keypoints_from_df(df, idx)

                if dlc_keypoints:
                    # Convert to MediaPipe format
                    keypoints_2d_cropped = self._dlc_to_mediapipe_keypoints(
                        dlc_keypoints, self.cropped_frames_cache[idx].shape
                    )

                    # Map to original image coordinates
                    keypoints_2d = self.map_keypoints_to_original(keypoints_2d_cropped, y_offset)

                    # Compute 3D if depth available
                    keypoints_3d = None
                    if depth_images and idx < len(depth_images):
                        keypoints_3d = self._compute_3d_keypoints(
                            keypoints_2d, depth_images[idx], fx, fy, cx, cy
                        )

                    # Compute bbox
                    bbox = self._compute_bbox_from_keypoints(keypoints_2d)

                    result = SubjectDetectionResult(
                        subject_type='dog',
                        bbox=bbox,
                        keypoints_2d=keypoints_2d,
                        keypoints_3d=keypoints_3d
                    )
                    results[frame_key] = result

            print(f"✅ Processed {len(results)} frames with dog detections")

            # Clear cache
            self.cropped_frames_cache = []
            self.frame_numbers_cache = []
            self.y_offsets_cache = []

            return results

        except Exception as e:
            print(f"❌ Batch video processing failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _parse_dlc_json_batch(self, dlc_data: list, depth_images: Optional[List[np.ndarray]],
                              fx: float, fy: float, cx: float, cy: float) -> Dict[str, SubjectDetectionResult]:
        """
        Parse DeepLabCut JSON output for batch processing.

        Args:
            dlc_data: List of frame data from DLC JSON
            depth_images: Optional list of depth images
            fx, fy: Focal lengths
            cx, cy: Principal point

        Returns:
            Dictionary mapping frame_keys to SubjectDetectionResult
        """
        results = {}

        # DLC JSON can be either a list or dict
        if isinstance(dlc_data, dict):
            # Convert dict to list indexed by frame number
            dlc_list = [dlc_data.get(i, {}) for i in range(len(self.frame_numbers_cache))]
        else:
            dlc_list = dlc_data

        for idx, (frame_num, y_offset) in enumerate(zip(self.frame_numbers_cache, self.y_offsets_cache)):
            if idx >= len(dlc_list):
                continue

            frame_key = f"frame_{frame_num:06d}"
            frame_data = dlc_list[idx]

            # Extract bodyparts for first individual
            bodyparts_list = frame_data.get('bodyparts', [])
            if not bodyparts_list or len(bodyparts_list) == 0:
                continue

            keypoints = bodyparts_list[0]  # First individual

            # Convert to dictionary format
            dlc_keypoints = {}
            for i, kp in enumerate(keypoints):
                if i < len(self.keypoint_names):
                    bodypart_name = self.keypoint_names[i]
                    dlc_keypoints[bodypart_name] = kp

            # Convert to MediaPipe format
            keypoints_2d_cropped = self._dlc_to_mediapipe_keypoints(
                dlc_keypoints, self.cropped_frames_cache[idx].shape
            )

            if not keypoints_2d_cropped:
                continue

            # Map to original image coordinates
            keypoints_2d = self.map_keypoints_to_original(keypoints_2d_cropped, y_offset)

            # Compute 3D if depth available
            keypoints_3d = None
            if depth_images and idx < len(depth_images):
                keypoints_3d = self._compute_3d_keypoints(
                    keypoints_2d, depth_images[idx], fx, fy, cx, cy
                )

            # Compute bbox
            bbox = self._compute_bbox_from_keypoints(keypoints_2d)

            result = SubjectDetectionResult(
                subject_type='dog',
                bbox=bbox,
                keypoints_2d=keypoints_2d,
                keypoints_3d=keypoints_3d
            )
            results[frame_key] = result

        print(f"✅ Processed {len(results)}/{len(self.frame_numbers_cache)} frames with dog detections")

        # Clear cache
        self.cropped_frames_cache = []
        self.frame_numbers_cache = []
        self.y_offsets_cache = []

        return results

    def _extract_dlc_keypoints_from_df(self, df, frame_idx: int) -> Optional[Dict]:
        """
        Extract keypoints from DeepLabCut DataFrame for a specific frame.

        Args:
            df: DeepLabCut output DataFrame
            frame_idx: Frame index

        Returns:
            Dictionary mapping bodypart names to [x, y, confidence]
        """
        try:
            keypoints = {}
            # DLC DataFrame has MultiIndex columns: (scorer, bodypart, coords)
            # Extract for individual=0 (first detected dog)
            for bodypart in df.columns.get_level_values(1).unique():
                if frame_idx < len(df):
                    x = df.iloc[frame_idx][(df.columns[0][0], bodypart, 'x')]
                    y = df.iloc[frame_idx][(df.columns[0][0], bodypart, 'y')]
                    conf = df.iloc[frame_idx][(df.columns[0][0], bodypart, 'likelihood')]
                    keypoints[bodypart] = [x, y, conf]
            return keypoints
        except Exception as e:
            print(f"⚠️ Could not extract keypoints for frame {frame_idx}: {e}")
            return None

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

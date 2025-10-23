"""
MediaPipe human pose detector for gesture analysis.

Detects 33 body landmarks using MediaPipe Pose.
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Tuple
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️ MediaPipe not installed. Run: pip install mediapipe")

from .skeleton_base import SkeletonDetector, SkeletonResult


class MediaPipeHumanDetector(SkeletonDetector):
    """MediaPipe-based human pose detector."""

    # MediaPipe Pose landmark names (33 landmarks)
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear",
        "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_pinky", "right_pinky",
        "left_index", "right_index",
        "left_thumb", "right_thumb",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
        "left_heel", "right_heel",
        "left_foot_index", "right_foot_index"
    ]

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 1,
                 enable_segmentation: bool = False,
                 smooth_landmarks: bool = True,
                 use_optical_flow: bool = True,
                 motion_history_frames: int = 5,
                 lower_half_only: bool = False):
        """
        Initialize MediaPipe Pose detector.

        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
            enable_segmentation: Enable person segmentation
            smooth_landmarks: Enable landmark smoothing
            use_optical_flow: Use optical flow for motion detection
            motion_history_frames: Number of frames to track for motion analysis
            lower_half_only: Process only lower half of image (for baby detection)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # False for video
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Optical flow settings
        self.use_optical_flow = use_optical_flow
        self.motion_history_frames = motion_history_frames

        # Lower-half processing (for baby detection)
        self.lower_half_only = lower_half_only
        self.crop_ratio = 0.6  # Keep lower 50% of image

        # History tracking for optical flow
        self.landmark_history = []  # List of past landmarks
        self.prev_gray = None  # Previous grayscale frame for optical flow

    def detect_frame(self, image: np.ndarray, frame_number: int,
                    depth_image: Optional[np.ndarray] = None,
                    fx: float = 615.0, fy: float = 615.0,
                    cx: float = 320.0, cy: float = 240.0) -> Optional[SkeletonResult]:
        """
        Detect pose in a single frame.

        Args:
            image: RGB image (H, W, 3)
            frame_number: Frame number
            depth_image: Optional depth image in meters (H, W)
            fx, fy: Camera focal lengths in pixels
            cx, cy: Camera principal point in pixels

        Returns:
            SkeletonResult or None if no pose detected
        """
        # Crop to lower half if enabled (for baby detection)
        y_offset = 0
        process_image = image
        process_depth = depth_image

        if self.lower_half_only:
            from .image_utils import crop_to_lower_half
            process_image, y_offset = crop_to_lower_half(image, self.crop_ratio)

            if depth_image is not None:
                process_depth, _ = crop_to_lower_half(depth_image, self.crop_ratio)

        # Process image
        results = self.pose.process(process_image)

        if not results.pose_landmarks:
            return None

        # Extract 2D landmarks
        landmarks_2d = []
        h, w = process_image.shape[:2]  # Use processed image dimensions

        for landmark in results.pose_landmarks.landmark:
            x = landmark.x * w  # Convert normalized to pixel coordinates
            y = landmark.y * h
            visibility = landmark.visibility
            landmarks_2d.append((x, y, visibility))

        # Map coordinates back to original image if cropped
        if y_offset > 0:
            from .image_utils import map_coordinates_from_crop
            # Convert to list format for mapping
            kp_list = [[x, y, vis] for x, y, vis in landmarks_2d]
            mapped_kp = map_coordinates_from_crop(kp_list, y_offset)
            landmarks_2d = [(x, y, vis) for x, y, vis in mapped_kp]

        # Compute 3D landmarks using hybrid approach:
        # - Depth image for accurate hip center position
        # - MediaPipe world landmarks for smooth relative limb positions
        landmarks_3d = None
        if depth_image is not None and results.pose_world_landmarks:
            # Use hybrid method: depth + MediaPipe world landmarks
            landmarks_3d = self._compute_3d_from_mediapipe_world(
                results.pose_world_landmarks,
                landmarks_2d,
                depth_image,
                fx, fy, cx, cy
            )

        # Update landmark history for motion tracking
        self._update_landmark_history(landmarks_2d, image)

        # Determine pointing arm (with optical flow if enabled)
        pointing_arm = self._determine_pointing_arm(landmarks_2d)

        # Compute hip center in 2D and 3D
        hip_center_2d = self._compute_hip_center_2d(landmarks_2d)
        hip_center_3d = None
        if landmarks_3d is not None:
            hip_center_3d = self._compute_hip_center_3d(landmarks_3d)

        # Compute arm vectors if we have 3D data
        arm_vectors = None
        if landmarks_3d is not None:
            arm_vectors = self._compute_arm_vectors(landmarks_3d, pointing_arm)

        metadata = {
            "pointing_arm": pointing_arm,
            "model": "mediapipe_pose",
            "num_landmarks": len(landmarks_2d),
            "has_depth": depth_image is not None,
            "hip_center_2d": hip_center_2d,
            "hip_center_3d": hip_center_3d
        }

        return SkeletonResult(
            frame_number=frame_number,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            keypoint_names=self.LANDMARK_NAMES,
            metadata=metadata,
            arm_vectors=arm_vectors
        )

    def _compute_3d_from_mediapipe_world(self,
                                        world_landmarks,
                                        landmarks_2d: List[tuple],
                                        depth_image: np.ndarray,
                                        fx: float, fy: float,
                                        cx: float, cy: float) -> List[tuple]:
        """
        Convert MediaPipe world landmarks to camera frame using depth for scale.

        MediaPipe world landmarks are:
        - Person-centric (origin at hip center)
        - X: person's right, Y: person's up, Z: person's back
        - Scale: metric (meters) but relative to body size

        We:
        1. Get hip center depth from depth image
        2. Use MediaPipe world landmarks (smoother, less noisy)
        3. Transform from person frame to camera frame
        4. Scale using actual depth

        Args:
            world_landmarks: MediaPipe pose_world_landmarks
            landmarks_2d: List of (x, y, visibility) tuples for depth lookup
            depth_image: Depth image in meters (H, W)
            fx, fy: Camera focal lengths
            cx, cy: Camera principal point

        Returns:
            List of (x, y, z) tuples in camera frame (meters)
        """
        LEFT_HIP = 23
        RIGHT_HIP = 24
        h, w = depth_image.shape[:2]

        # print(f"\n🔍 HYBRID 3D COMPUTATION DEBUG:")
        # print(f"   Depth image shape: {depth_image.shape}")

        # Step 1: Get hip center depth from depth image
        left_hip_2d = landmarks_2d[LEFT_HIP]
        right_hip_2d = landmarks_2d[RIGHT_HIP]

        # print(f"   Left hip 2D (pixel):  ({left_hip_2d[0]:.1f}, {left_hip_2d[1]:.1f})")
        # print(f"   Right hip 2D (pixel): ({right_hip_2d[0]:.1f}, {right_hip_2d[1]:.1f})")

        # Get depth at hip positions
        x_left = int(np.clip(left_hip_2d[0], 0, w - 1))
        y_left = int(np.clip(left_hip_2d[1], 0, h - 1))
        x_right = int(np.clip(right_hip_2d[0], 0, w - 1))
        y_right = int(np.clip(right_hip_2d[1], 0, h - 1))

        depth_left = depth_image[y_left, x_left]
        depth_right = depth_image[y_right, x_right]

        # print(f"   Left hip depth:  {depth_left:.4f} m at pixel ({x_left}, {y_left})")
        # print(f"   Right hip depth: {depth_right:.4f} m at pixel ({x_right}, {y_right})")

        # Average valid depths
        valid_depths = []
        if depth_left > 0 and np.isfinite(depth_left):
            valid_depths.append(depth_left)
        if depth_right > 0 and np.isfinite(depth_right):
            valid_depths.append(depth_right)

        if not valid_depths:
            print(f"   ⚠️  NO VALID HIP DEPTHS - falling back to per-landmark depth lookup")
            # No valid depth, fallback to depth lookup method
            return self._compute_3d_from_depth(landmarks_2d, depth_image, fx, fy, cx, cy)

        hip_center_depth = np.mean(valid_depths)
        # print(f"   Hip center depth: {hip_center_depth:.4f} m (average of {len(valid_depths)} valid depths)")

        # Step 2: Get hip center in 2D (for camera frame origin)
        hip_center_2d_x = (left_hip_2d[0] + right_hip_2d[0]) / 2
        hip_center_2d_y = (left_hip_2d[1] + right_hip_2d[1]) / 2

        # Hip center in camera frame
        hip_center_cam_x = (hip_center_2d_x - cx) * hip_center_depth / fx
        hip_center_cam_y = (hip_center_2d_y - cy) * hip_center_depth / fy
        hip_center_cam_z = hip_center_depth

        # Step 3: Transform each world landmark to camera frame
        landmarks_3d = []

        for idx, world_lm in enumerate(world_landmarks.landmark):
            # MediaPipe world coords (person frame, hip-centered)
            # Based on empirical data:
            # - X: person's right (+) / left (-)
            # - Y: person's DOWN (+) / up (-)  [hip=0, shoulder<0 (above), ankle>0 (below)]
            # - Z: person's BACK (+) / forward (-)  [nose<0 (front), ankle>0 (back)]
            x_person = world_lm.x
            y_person = world_lm.y
            z_person = world_lm.z

            # Transform to camera frame:
            # Camera: X=right, Y=down, Z=forward (depth into scene)
            # Person: X=right, Y=down, Z=back

            # Transformation:
            # - Person X → Camera X (same direction)
            # - Person Y → Camera Y (same direction, both down)
            # - Person Z → Camera Z (ADD person Z offset directly)
            #   Person Z: negative=front (face), positive=back (behind)
            #   Camera Z: positive=depth (away from camera)
            #   Example: nose at person Z=-0.35 (front) → offset -0.35 → closer to camera ✓
            #   Example: ankle at person Z=+0.13 (back) → offset +0.13 → further from camera ✓

            x_cam = hip_center_cam_x + x_person
            y_cam = hip_center_cam_y + y_person  # No flip: both Y-down
            z_cam = hip_center_cam_z + z_person  # No flip: add offset directly

            # Debug key landmarks (hips, shoulders, feet, nose)
            NOSE = 0
            LEFT_SHOULDER = 11
            RIGHT_SHOULDER = 12
            LEFT_ANKLE = 27
            RIGHT_ANKLE = 28
            LEFT_WRIST = 15
            RIGHT_WRIST = 16

            if idx in [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE, LEFT_WRIST, RIGHT_WRIST]:
                landmark_names = {
                    NOSE: "NOSE",
                    LEFT_SHOULDER: "LEFT_SHOULDER",
                    RIGHT_SHOULDER: "RIGHT_SHOULDER",
                    LEFT_HIP: "LEFT_HIP",
                    RIGHT_HIP: "RIGHT_HIP",
                    LEFT_ANKLE: "LEFT_ANKLE",
                    RIGHT_ANKLE: "RIGHT_ANKLE",
                    LEFT_WRIST: "LEFT_WRIST",
                    RIGHT_WRIST: "RIGHT_WRIST"
                }
                # print(f"  Landmark {idx:2d} ({landmark_names[idx]:15s}): MediaPipe ({x_person:+.4f}, {y_person:+.4f}, {z_person:+.4f}) → Camera ({x_cam:+.4f}, {y_cam:+.4f}, {z_cam:+.4f})")

            landmarks_3d.append((float(x_cam), float(y_cam), float(z_cam)))

        # Verify hip center
        computed_hip_center = (
            (landmarks_3d[LEFT_HIP][0] + landmarks_3d[RIGHT_HIP][0]) / 2,
            (landmarks_3d[LEFT_HIP][1] + landmarks_3d[RIGHT_HIP][1]) / 2,
            (landmarks_3d[LEFT_HIP][2] + landmarks_3d[RIGHT_HIP][2]) / 2
        )
        # print(f"  Hip center from depth: ({hip_center_cam_x:+.4f}, {hip_center_cam_y:+.4f}, {hip_center_cam_z:+.4f})")
        # print(f"  Hip center computed:   ({computed_hip_center[0]:+.4f}, {computed_hip_center[1]:+.4f}, {computed_hip_center[2]:+.4f})")

        return landmarks_3d

    def _compute_3d_from_depth(self, landmarks_2d: List[tuple],
                              depth_image: np.ndarray,
                              fx: float, fy: float,
                              cx: float, cy: float) -> List[tuple]:
        """
        Convert 2D landmarks to 3D using depth image.

        Args:
            landmarks_2d: List of (x, y, visibility) tuples
            depth_image: Depth image in meters (H, W)
            fx, fy: Camera focal lengths
            cx, cy: Camera principal point

        Returns:
            List of (x, y, z) tuples in camera frame (meters)
        """
        landmarks_3d = []
        h, w = depth_image.shape[:2]

        for x, y, visibility in landmarks_2d:
            # Get depth at landmark position
            x_int = int(np.clip(x, 0, w - 1))
            y_int = int(np.clip(y, 0, h - 1))

            z = depth_image[y_int, x_int]

            # Skip if no valid depth
            if z <= 0 or np.isnan(z) or np.isinf(z):
                landmarks_3d.append((0.0, 0.0, 0.0))
                continue

            # Convert pixel to 3D point in camera frame
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy
            z_3d = z

            landmarks_3d.append((float(x_3d), float(y_3d), float(z_3d)))

        return landmarks_3d

    def _update_landmark_history(self, landmarks_2d: List[tuple], image: np.ndarray):
        """
        Update the history of landmarks for motion tracking.

        Args:
            landmarks_2d: Current frame landmarks
            image: Current frame image (for optical flow)
        """
        # Convert to grayscale for optical flow
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Add current landmarks to history
        self.landmark_history.append(landmarks_2d)

        # Keep only recent frames
        if len(self.landmark_history) > self.motion_history_frames:
            self.landmark_history.pop(0)

        # Store current frame for next optical flow calculation
        self.prev_gray = gray

    def _calculate_wrist_motion(self, landmarks_2d: List[tuple]) -> Dict[str, float]:
        """
        Calculate motion magnitude for each wrist using landmark history.

        Args:
            landmarks_2d: Current frame landmarks

        Returns:
            Dictionary with 'left' and 'right' motion magnitudes
        """
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        motion = {'left': 0.0, 'right': 0.0}

        # Need at least 2 frames to calculate motion
        if len(self.landmark_history) < 2:
            return motion

        try:
            # Calculate motion over the history window
            left_motion_sum = 0.0
            right_motion_sum = 0.0

            for i in range(1, len(self.landmark_history)):
                prev_landmarks = self.landmark_history[i - 1]
                curr_landmarks = self.landmark_history[i]

                # Left wrist motion
                prev_left = prev_landmarks[LEFT_WRIST]
                curr_left = curr_landmarks[LEFT_WRIST]
                left_dx = curr_left[0] - prev_left[0]
                left_dy = curr_left[1] - prev_left[1]
                left_motion = np.sqrt(left_dx**2 + left_dy**2)
                left_motion_sum += left_motion

                # Right wrist motion
                prev_right = prev_landmarks[RIGHT_WRIST]
                curr_right = curr_landmarks[RIGHT_WRIST]
                right_dx = curr_right[0] - prev_right[0]
                right_dy = curr_right[1] - prev_right[1]
                right_motion = np.sqrt(right_dx**2 + right_dy**2)
                right_motion_sum += right_motion

            # Average motion over the window
            motion['left'] = left_motion_sum / (len(self.landmark_history) - 1)
            motion['right'] = right_motion_sum / (len(self.landmark_history) - 1)

        except (IndexError, TypeError):
            pass

        return motion

    def _compute_hip_center_2d(self, landmarks_2d: List[tuple]) -> Optional[Tuple[float, float]]:
        """
        Compute hip center in 2D from left and right hip landmarks.

        Args:
            landmarks_2d: List of (x, y, visibility) tuples

        Returns:
            Tuple of (x, y) in pixels, or None if hips not detected
        """
        LEFT_HIP = 23
        RIGHT_HIP = 24

        try:
            left_hip = landmarks_2d[LEFT_HIP]
            right_hip = landmarks_2d[RIGHT_HIP]

            # Check visibility
            if left_hip[2] < 0.5 or right_hip[2] < 0.5:
                return None

            # Average of left and right hip
            hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0

            return (hip_center_x, hip_center_y)

        except (IndexError, TypeError):
            return None

    def _compute_hip_center_3d(self, landmarks_3d: List[tuple]) -> Optional[Tuple[float, float, float]]:
        """
        Compute hip center in 3D from left and right hip landmarks.

        Args:
            landmarks_3d: List of (x, y, z) tuples in camera frame (meters)

        Returns:
            Tuple of (x, y, z) in meters, or None if hips not detected
        """
        LEFT_HIP = 23
        RIGHT_HIP = 24

        try:
            left_hip = landmarks_3d[LEFT_HIP]
            right_hip = landmarks_3d[RIGHT_HIP]

            # Check if depth is valid (not zero or NaN)
            if left_hip[2] <= 0 or right_hip[2] <= 0:
                return None
            if not (np.isfinite(left_hip[2]) and np.isfinite(right_hip[2])):
                return None

            # Average of left and right hip in 3D
            hip_center_x = (left_hip[0] + right_hip[0]) / 2.0
            hip_center_y = (left_hip[1] + right_hip[1]) / 2.0
            hip_center_z = (left_hip[2] + right_hip[2]) / 2.0

            return (hip_center_x, hip_center_y, hip_center_z)

        except (IndexError, TypeError):
            return None

    def _determine_pointing_arm(self, landmarks_2d: List[tuple]) -> str:
        """
        Determine which arm is pointing based on multiple criteria.
        Now uses optical flow motion as the PRIMARY criterion.

        Args:
            landmarks_2d: List of (x, y, visibility) tuples

        Returns:
            "left", "right", or "auto"
        """
        # Landmark indices
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24

        try:
            # Get landmark positions
            left_wrist = landmarks_2d[LEFT_WRIST]
            right_wrist = landmarks_2d[RIGHT_WRIST]
            left_elbow = landmarks_2d[LEFT_ELBOW]
            right_elbow = landmarks_2d[RIGHT_ELBOW]
            left_shoulder = landmarks_2d[LEFT_SHOULDER]
            right_shoulder = landmarks_2d[RIGHT_SHOULDER]
            left_hip = landmarks_2d[LEFT_HIP]
            right_hip = landmarks_2d[RIGHT_HIP]

            # Score each arm based on multiple criteria
            left_score = 0
            right_score = 0

            # Criterion 0: MOTION (NEW - HIGHEST PRIORITY - 5 points)
            # The wrist that is moving more is more likely pointing
            if self.use_optical_flow and len(self.landmark_history) >= 2:
                wrist_motion = self._calculate_wrist_motion(landmarks_2d)
                left_motion = wrist_motion['left']
                right_motion = wrist_motion['right']

                # Threshold: at least 2 pixels average motion to be considered "moving"
                motion_threshold = 2.0

                if left_motion > motion_threshold and right_motion > motion_threshold:
                    # Both moving - give points to the one moving MORE
                    if left_motion > right_motion * 1.3:  # Left moving 30% more
                        left_score += 5
                    elif right_motion > left_motion * 1.3:  # Right moving 30% more
                        right_score += 5
                    elif left_motion > right_motion * 1.1:  # Left moving 10% more
                        left_score += 3
                    elif right_motion > left_motion * 1.1:  # Right moving 10% more
                        right_score += 3
                elif left_motion > motion_threshold:  # Only left moving
                    left_score += 5
                elif right_motion > motion_threshold:  # Only right moving
                    right_score += 5

            # Criterion 1: Wrist height relative to shoulder
            # Higher wrist = more likely pointing
            left_height_diff = left_shoulder[1] - left_wrist[1]  # Positive if wrist above shoulder
            right_height_diff = right_shoulder[1] - right_wrist[1]

            if left_height_diff > 30:  # Left wrist significantly above shoulder
                left_score += 3
            elif left_height_diff > 10:
                left_score += 1

            if right_height_diff > 30:  # Right wrist significantly above shoulder
                right_score += 3
            elif right_height_diff > 10:
                right_score += 1

            # Criterion 2: Arm extension (straighter arm = more likely pointing)
            # Calculate angle at elbow (straighter = closer to 180 degrees)
            def calc_arm_extension(shoulder, elbow, wrist):
                # Vectors: shoulder->elbow and elbow->wrist
                v1 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
                v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

                # Normalize
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)

                if v1_norm > 0 and v2_norm > 0:
                    v1 = v1 / v1_norm
                    v2 = v2 / v2_norm
                    # Dot product gives cos(angle)
                    # cos(180°) = -1 (fully extended), cos(90°) = 0 (bent)
                    dot_product = np.dot(v1, v2)
                    return dot_product  # More negative = more extended
                return 0

            left_extension = calc_arm_extension(left_shoulder, left_elbow, left_wrist)
            right_extension = calc_arm_extension(right_shoulder, right_elbow, right_wrist)

            # More extended arm gets points
            if left_extension < -0.5:  # Fairly straight
                left_score += 2
            if right_extension < -0.5:
                right_score += 2

            # Criterion 3: Wrist distance from body (farther = more likely pointing)
            # Use hip center as body reference
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2

            left_dist = np.sqrt((left_wrist[0] - hip_center_x)**2 + (left_wrist[1] - hip_center_y)**2)
            right_dist = np.sqrt((right_wrist[0] - hip_center_x)**2 + (right_wrist[1] - hip_center_y)**2)

            # Farther from body gets points
            if left_dist > right_dist * 1.2:
                left_score += 1
            elif right_dist > left_dist * 1.2:
                right_score += 1

            # Criterion 4: Horizontal extension (wrist further from body horizontally)
            left_horiz_dist = abs(left_wrist[0] - hip_center_x)
            right_horiz_dist = abs(right_wrist[0] - hip_center_x)

            if left_horiz_dist > right_horiz_dist * 1.3:
                left_score += 1
            elif right_horiz_dist > left_horiz_dist * 1.3:
                right_score += 1

            # Make decision based on scores
            score_threshold = 2  # Need at least this much difference to be confident

            if left_score > right_score + score_threshold:
                return "left"
            elif right_score > left_score + score_threshold:
                return "right"
            else:
                # Not confident or both arms similar
                # Default to whichever wrist is higher
                if left_height_diff > right_height_diff and left_height_diff > 20:
                    return "left"
                elif right_height_diff > left_height_diff and right_height_diff > 20:
                    return "right"
                else:
                    return "auto"

        except (IndexError, TypeError, ValueError):
            return "auto"

    def _compute_arm_vectors(self, landmarks_3d: List[tuple],
                            pointing_arm: str) -> Dict[str, List[float]]:
        """
        Compute arm vectors matching legacy format.

        Calculates vectors like shoulder_to_wrist, elbow_to_wrist, etc.

        Args:
            landmarks_3d: List of (x, y, z) tuples
            pointing_arm: "left", "right", or "auto"

        Returns:
            Dictionary of arm vectors
        """
        # MediaPipe Pose landmark indices
        NOSE = 0
        LEFT_EYE = 2  # LEFT_EYE center (not inner)
        RIGHT_EYE = 5  # RIGHT_EYE center (not inner)
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        # Determine which arm to use
        if pointing_arm == "left":
            shoulder_idx = LEFT_SHOULDER
            elbow_idx = LEFT_ELBOW
            wrist_idx = LEFT_WRIST
            eye_idx = LEFT_EYE
        elif pointing_arm == "right":
            shoulder_idx = RIGHT_SHOULDER
            elbow_idx = RIGHT_ELBOW
            wrist_idx = RIGHT_WRIST
            eye_idx = RIGHT_EYE
        else:
            # For "auto", use the wrist that is HIGHER (more negative Y)
            left_wrist = np.array(landmarks_3d[LEFT_WRIST])
            right_wrist = np.array(landmarks_3d[RIGHT_WRIST])

            # Check which wrist is higher (lower Y value)
            if left_wrist[1] < right_wrist[1]:
                # Left wrist is higher
                shoulder_idx = LEFT_SHOULDER
                elbow_idx = LEFT_ELBOW
                wrist_idx = LEFT_WRIST
                eye_idx = LEFT_EYE
            else:
                # Right wrist is higher (or equal)
                shoulder_idx = RIGHT_SHOULDER
                elbow_idx = RIGHT_ELBOW
                wrist_idx = RIGHT_WRIST
                eye_idx = RIGHT_EYE

        # Helper function to compute vector
        def compute_vector(from_idx: int, to_idx: int) -> Optional[List[float]]:
            try:
                from_pt = np.array(landmarks_3d[from_idx])
                to_pt = np.array(landmarks_3d[to_idx])

                # Check for invalid points
                if np.all(from_pt == 0) or np.all(to_pt == 0):
                    return None

                # Compute normalized vector
                vec = to_pt - from_pt
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm

                return vec.tolist()
            except (IndexError, ValueError):
                return None

        # Compute all arm vectors (matching legacy format)
        vectors = {
            "eye_to_wrist": compute_vector(eye_idx, wrist_idx),
            "shoulder_to_wrist": compute_vector(shoulder_idx, wrist_idx),
            "elbow_to_wrist": compute_vector(elbow_idx, wrist_idx),
            "nose_to_wrist": compute_vector(NOSE, wrist_idx),
        }

        # Also store wrist location
        try:
            wrist_3d = landmarks_3d[wrist_idx]
            if not np.all(np.array(wrist_3d) == 0):
                vectors["wrist_location"] = list(wrist_3d)
            else:
                vectors["wrist_location"] = None
        except (IndexError, ValueError):
            vectors["wrist_location"] = None

        return vectors

    def detect_video(self, video_path: str) -> List[SkeletonResult]:
        """
        Detect pose in entire video.

        Args:
            video_path: Path to video file

        Returns:
            List of SkeletonResult for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        results = []
        frame_number = 0

        print(f"🎥 Processing video: {video_path}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_number += 1

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.detect_frame(image_rgb, frame_number)
            if result:
                results.append(result)

            # Progress indicator
            if frame_number % 30 == 0:
                print(f"  Processed {frame_number} frames...", end='\r')

        cap.release()
        print(f"\n✅ Processed {frame_number} frames, detected {len(results)} poses")

        return results

    def draw_skeleton(self, image: np.ndarray, result: SkeletonResult) -> np.ndarray:
        """
        Draw skeleton on image.

        Args:
            image: RGB image
            result: SkeletonResult

        Returns:
            Image with skeleton drawn
        """
        annotated_image = image.copy()

        # Convert landmarks back to MediaPipe format for drawing
        h, w = image.shape[:2]
        landmarks = self.mp_pose.PoseLandmark

        # Create MediaPipe landmarks object
        pose_landmarks = type('obj', (object,), {
            'landmark': [
                type('obj', (object,), {
                    'x': x / w,
                    'y': y / h,
                    'z': 0,
                    'visibility': v
                })()
                for x, y, v in result.landmarks_2d
            ]
        })()

        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            annotated_image,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Draw pointing arm indicator
        pointing_arm = result.metadata.get('pointing_arm', 'auto')
        cv2.putText(
            annotated_image,
            f"Pointing: {pointing_arm}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return annotated_image

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()

"""
Kalman Filter for 3D Landmark Trajectory Smoothing.

Implements a simple Kalman filter to smooth landmark positions over time,
reducing noise in position estimates while preserving actual movement.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class KalmanFilter3D:
    """
    Kalman filter for smoothing 3D point trajectories.

    Uses a constant velocity motion model:
    State: [x, y, z, vx, vy, vz]
    Observation: [x, y, z]
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1,
                 initial_covariance: float = 1.0):
        """
        Initialize Kalman filter.

        Args:
            process_noise: Process noise covariance (higher = trusts measurements more)
            measurement_noise: Measurement noise covariance (higher = smoother but laggier)
            initial_covariance: Initial state covariance
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_covariance = initial_covariance

        # State: [x, y, z, vx, vy, vz]
        self.state = None  # Will be initialized on first measurement
        self.P = None      # State covariance matrix

        # State transition matrix (constant velocity model)
        self.dt = 1.0 / 30.0  # Assume 30 FPS
        self.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Observation matrix (we only observe position)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance
        self.Q = np.eye(6) * process_noise
        # Increase velocity uncertainty
        self.Q[3:, 3:] *= 10

        # Measurement noise covariance
        self.R = np.eye(3) * measurement_noise

        self._initialized = False

    def reset(self):
        """Reset the filter state."""
        self.state = None
        self.P = None
        self._initialized = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update filter with new measurement and return filtered position.

        Args:
            measurement: [x, y, z] observed position

        Returns:
            Filtered [x, y, z] position
        """
        measurement = np.array(measurement).flatten()

        if not self._initialized or self.state is None:
            # Initialize state with first measurement
            self.state = np.zeros(6)
            self.state[:3] = measurement
            self.P = np.eye(6) * self.initial_covariance
            self._initialized = True
            return measurement.copy()

        # Predict step
        state_pred = self.F @ self.state
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Update step
        y = measurement - self.H @ state_pred  # Innovation
        S = self.H @ P_pred @ self.H.T + self.R  # Innovation covariance
        K = P_pred @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = state_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred

        return self.state[:3].copy()

    def predict(self) -> Optional[np.ndarray]:
        """
        Predict next position without measurement update.

        Returns:
            Predicted [x, y, z] position, or None if not initialized
        """
        if not self._initialized:
            return None

        state_pred = self.F @ self.state
        return state_pred[:3].copy()


class LandmarkKalmanFilter:
    """
    Multi-landmark Kalman filter for smoothing entire skeleton trajectories.

    Maintains separate Kalman filters for each landmark.
    """

    def __init__(self,
                 num_landmarks: int = 33,
                 process_noise: float = 0.005,
                 measurement_noise: float = 0.05):
        """
        Initialize multi-landmark filter.

        Args:
            num_landmarks: Number of landmarks (MediaPipe Pose = 33)
            process_noise: Process noise for each filter
            measurement_noise: Measurement noise for each filter
        """
        self.num_landmarks = num_landmarks
        self.filters = [
            KalmanFilter3D(process_noise, measurement_noise)
            for _ in range(num_landmarks)
        ]

    def reset(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()

    def update(self, landmarks_3d: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """
        Update all filters with new landmark positions.

        Args:
            landmarks_3d: List of (x, y, z) tuples for each landmark

        Returns:
            Filtered landmarks as list of (x, y, z) tuples
        """
        filtered = []

        for i, (lm, kf) in enumerate(zip(landmarks_3d, self.filters)):
            if lm is None or any(v is None for v in lm):
                # Skip invalid landmarks, use prediction if available
                pred = kf.predict()
                if pred is not None:
                    filtered.append(tuple(pred.tolist()))
                else:
                    filtered.append((0.0, 0.0, 0.0))
            else:
                pos = np.array(lm)
                filtered_pos = kf.update(pos)
                filtered.append(tuple(filtered_pos.tolist()))

        return filtered


class PointingTrajectoryFilter:
    """
    Filter specifically for pointing analysis trajectories.

    Filters ground intersection points and key landmarks.
    """

    def __init__(self,
                 process_noise: float = 0.01,
                 measurement_noise: float = 0.1):
        """
        Initialize pointing trajectory filter.

        Args:
            process_noise: Process noise (lower = smoother)
            measurement_noise: Measurement noise (higher = smoother)
        """
        # Filters for different pointing vectors
        self.intersection_filters = {
            'eye_to_wrist': KalmanFilter3D(process_noise, measurement_noise),
            'shoulder_to_wrist': KalmanFilter3D(process_noise, measurement_noise),
            'elbow_to_wrist': KalmanFilter3D(process_noise, measurement_noise),
            'nose_to_wrist': KalmanFilter3D(process_noise, measurement_noise),
            'head_orientation': KalmanFilter3D(process_noise, measurement_noise),
        }

        # Filter for wrist position
        self.wrist_filter = KalmanFilter3D(process_noise * 2, measurement_noise * 0.5)

    def reset(self):
        """Reset all filters."""
        for f in self.intersection_filters.values():
            f.reset()
        self.wrist_filter.reset()

    def filter_analysis(self, analysis: Dict) -> Dict:
        """
        Apply Kalman filtering to pointing analysis results.

        Args:
            analysis: Dictionary from analyze_pointing_frame()

        Returns:
            Analysis dict with filtered values (new keys with _filtered suffix)
        """
        if analysis is None:
            return None

        filtered = dict(analysis)  # Copy original

        # Filter wrist location
        if 'wrist_location' in analysis and analysis['wrist_location']:
            wrist = analysis['wrist_location']
            if wrist[0] is not None:
                filtered_wrist = self.wrist_filter.update(np.array(wrist))
                filtered['wrist_location_filtered'] = filtered_wrist.tolist()

        # Filter ground intersections
        for vec_name, kf in self.intersection_filters.items():
            key = f'{vec_name}_ground_intersection'
            if key in analysis:
                intersection = analysis[key]
                if intersection and intersection[0] is not None:
                    filtered_intersection = kf.update(np.array(intersection))
                    filtered[f'{key}_filtered'] = filtered_intersection.tolist()
                else:
                    # Use prediction if available
                    pred = kf.predict()
                    if pred is not None:
                        filtered[f'{key}_filtered'] = pred.tolist()
                    else:
                        filtered[f'{key}_filtered'] = [None, None, None]

        return filtered


def apply_kalman_smoothing(results: List,
                           process_noise: float = 0.005,
                           measurement_noise: float = 0.05) -> List:
    """
    Apply Kalman filtering to a list of detection results.

    Smooths 3D landmark positions across frames.

    Args:
        results: List of SkeletonResult objects with landmarks_3d
        process_noise: Kalman filter process noise
        measurement_noise: Kalman filter measurement noise

    Returns:
        Same results list with landmarks_3d_filtered added to each result
    """
    if not results:
        return results

    # Determine number of landmarks from first valid result
    num_landmarks = 33  # MediaPipe default
    for r in results:
        if hasattr(r, 'landmarks_3d') and r.landmarks_3d:
            num_landmarks = len(r.landmarks_3d)
            break

    # Create multi-landmark filter
    lm_filter = LandmarkKalmanFilter(
        num_landmarks=num_landmarks,
        process_noise=process_noise,
        measurement_noise=measurement_noise
    )

    # Process each frame
    for result in results:
        if hasattr(result, 'landmarks_3d') and result.landmarks_3d:
            filtered = lm_filter.update(result.landmarks_3d)
            result.landmarks_3d_filtered = filtered
        else:
            result.landmarks_3d_filtered = None

    return results


def smooth_pointing_analyses(analyses_dict: Dict,
                             process_noise: float = 0.01,
                             measurement_noise: float = 0.1) -> Dict:
    """
    Apply Kalman smoothing to a dictionary of pointing analyses.

    Args:
        analyses_dict: Dictionary of {frame_key: analysis_dict}
        process_noise: Kalman filter process noise
        measurement_noise: Kalman filter measurement noise

    Returns:
        New dictionary with filtered values added to each analysis
    """
    if not analyses_dict:
        return analyses_dict

    # Sort by frame number
    sorted_items = sorted(analyses_dict.items(),
                         key=lambda item: int(item[0].split('_')[-1]))

    # Create trajectory filter
    traj_filter = PointingTrajectoryFilter(process_noise, measurement_noise)

    # Process each frame in order
    filtered_dict = {}
    for frame_key, analysis in sorted_items:
        filtered_dict[frame_key] = traj_filter.filter_analysis(analysis)

    return filtered_dict

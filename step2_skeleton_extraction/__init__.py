"""
Step 2: Skeleton Extraction

Pose detection for humans (MediaPipe) and dogs (DeepLabCut).
"""

from .mediapipe_human import MediaPipeHumanDetector
from .skeleton_base import SkeletonDetector, SkeletonResult
from .kalman_filter import (
    KalmanFilter3D,
    LandmarkKalmanFilter,
    PointingTrajectoryFilter,
    apply_kalman_smoothing,
    smooth_pointing_analyses
)
from .plot_distance_to_targets import (
    plot_distance_to_targets,
    plot_distance_summary,
    plot_best_representation_analysis
)

__all__ = [
    'MediaPipeHumanDetector',
    'SkeletonDetector',
    'SkeletonResult',
    'KalmanFilter3D',
    'LandmarkKalmanFilter',
    'PointingTrajectoryFilter',
    'apply_kalman_smoothing',
    'smooth_pointing_analyses',
    'plot_distance_to_targets',
    'plot_distance_summary',
    'plot_best_representation_analysis',
]

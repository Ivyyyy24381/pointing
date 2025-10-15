"""
Step 2: Skeleton Extraction

Pose detection for humans (MediaPipe) and dogs (DeepLabCut).
"""

from .mediapipe_human import MediaPipeHumanDetector
from .skeleton_base import SkeletonDetector, SkeletonResult

__all__ = [
    'MediaPipeHumanDetector',
    'SkeletonDetector',
    'SkeletonResult',
]

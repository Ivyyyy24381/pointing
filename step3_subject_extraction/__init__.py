"""
Step 3: Subject Extraction (Optional)

Detect subjects (dog/baby) in lower half of image.
"""

from .subject_detector import SubjectDetector, SubjectDetectionResult
from .sam3_detector import SAM3Detector, SAM3_AVAILABLE

__all__ = ['SubjectDetector', 'SubjectDetectionResult', 'SAM3Detector', 'SAM3_AVAILABLE']
